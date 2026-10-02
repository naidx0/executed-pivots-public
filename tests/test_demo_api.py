"""The live demo's API: the one-click examples score as the audit did, and untrusted input is bounded."""
import json
import time

import pytest

from cleave.world.overlay import available

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytestmark = pytest.mark.skipif(not available(), reason="needs user namespaces + overlayfs")

from fastapi.testclient import TestClient  # noqa: E402

from pivots.demo.app import MAX_BODY, create_app  # noqa: E402
from pivots.demo.core import EXAMPLES, DemoCore  # noqa: E402
from pivots.demo.sandbox import Limits  # noqa: E402

TIMEOUT = 3
EXPECT = {"paraphrase": (0, 1, 1), "flipped": (1, 0, 0), "early": (1, 0, 0)}  # J, X, P


@pytest.fixture(scope="module")
def core(tmp_path_factory):
    return DemoCore(str(tmp_path_factory.mktemp("xp-demo-store")), limits=Limits(batch_timeout_s=TIMEOUT),
                    max_concurrent=2, max_queue=2, queue_wait_s=0.5)


@pytest.fixture(scope="module")
def client(core):
    with TestClient(create_app(core, warm=False)) as c:
        yield c


def run(client, **body):
    r = client.post("/api/run", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_page_and_pivots(client):
    page = client.get("/")
    assert page.status_code == 200 and "Executed Pivots" in page.text
    d = client.get("/api/pivots").json()
    assert len(d["pivots"]) == 17 and {e["id"] for e in d["examples"]} == set(EXPECT)
    det = client.get("/api/pivot/billing-invoice-bugfix:3").json()
    assert det["expert"]["commands"][0].startswith("sed -i ")
    assert "FAILED (failures=4)" in det["before"]
    ids = {c["id"] for c in det["candidates"]}
    assert {"ctrl:expert", "ctrl:flipped", "ctrl:early_complete", "student:opus:5"} <= ids


@pytest.mark.parametrize("ex", EXAMPLES, ids=[e["id"] for e in EXAMPLES])
def test_one_click_examples(client, core, ex):
    r = run(client, pivot=ex["pivot"], source=ex["source"])
    assert (r["J"]["reward"], r["X"]["reward"], r["P"]["P"]) == EXPECT[ex["id"]]
    audit = core.data["audit"][ex["pivot"]][ex["source"]]  # the live run agrees with the stored audit
    assert (r["J"]["reward"], r["X"]["reward"], r["P"]["P"]) == (audit["J"], audit["X"], audit["P"])
    assert r["J"]["similarity"] == pytest.approx(audit["J_sim"], abs=1e-3)
    assert r["X"]["self_agreement"] == 1.0
    if ex["id"] == "flipped":
        assert r["J"]["similarity"] > 0.99 and r["X"]["score"] < 0.1
        assert r["X"]["missing"] == ["/app/billing/invoice.py"] and r["X"]["your_changed"] == []
        assert (r["P"]["before"], r["P"]["expert"], r["P"]["after"]) == (6, 9, 6)
        assert "FAILED (failures=4)" in r["terminal"]["yours"] and "FAILED (failures=2)" in r["terminal"]["expert"]
    if ex["id"] == "early":
        assert r["X"]["penalties"] == {"premature_complete": 0.0}
        assert r["P"]["claims_complete"] and not r["P"]["task_passes"]
    if ex["id"] == "paraphrase":
        assert r["J"]["similarity"] == pytest.approx(0.70, abs=0.01)
        assert (r["P"]["expert"], r["P"]["after"]) == (12, 12)
        assert 'Revert "Speed up tax computation"' in r["terminal"]["yours"]


def test_edited_batch_scores_like_the_stored_one(client, core):
    det = client.get("/api/pivot/billing-invoice-bugfix:3").json()
    flipped = next(c for c in det["candidates"] if c["id"] == "ctrl:flipped")
    r = run(client, pivot="billing-invoice-bugfix:3", keystrokes=flipped["keystrokes"], verify=False)
    assert (r["J"]["reward"], r["X"]["reward"], r["P"]) == (1, 0, None)
    fixed = run(client, pivot="billing-invoice-bugfix:3", verify=False,
                keystrokes="python3 - <<'EOF'\nimport re\np='billing/invoice.py'\ns=open(p).read()\n"
                           "open(p,'w').write(s.replace('if qty > BULK_THRESHOLD:', 'if qty >= BULK_THRESHOLD:'))\nEOF\n"
                           "python3 -m unittest discover -s tests 2>&1 | tail -n 5")
    assert fixed["J"]["reward"] == 0 and fixed["X"]["state"] == 1.0


def test_timeout(client):
    t0 = time.time()
    r = run(client, pivot="billing-invoice-bugfix:3", keystrokes="echo started\nsleep 60\necho never")
    assert time.time() - t0 < TIMEOUT + 15
    assert r["X"]["timed_out"] and r["X"]["penalties"]["timeout"] == 0.5 and r["X"]["reward"] == 0
    assert r["terminal"]["timed_out"] and "started" in r["terminal"]["yours"]
    assert "never" not in r["terminal"]["yours"].replace("echo never", "")
    assert r["P"]["timed_out"] and (r["P"]["after"], r["P"]["P"]) == (6, 0)  # verifier reads the state it left


def test_input_limits(client, core):
    p = "billing-invoice-bugfix:3"
    too_long = client.post("/api/run", json={"pivot": p, "keystrokes": "x" * (core.inputs.max_keystroke_chars + 1)})
    assert too_long.status_code == 400 and "limit" in too_long.json()["error"]
    many = client.post("/api/run", json={"pivot": p, "keystrokes": "true\n" * (core.inputs.max_lines + 1)})
    assert many.status_code == 400
    raw = client.post("/api/run", json={"pivot": p, "action": "{" + " " * core.inputs.max_action_chars + "}"})
    assert raw.status_code == 400
    body = client.post("/api/run", content=json.dumps({"pivot": p, "keystrokes": "y" * MAX_BODY}),
                       headers={"Content-Type": "application/json"})
    assert body.status_code == 413
    assert client.post("/api/run", json={"pivot": "billing-invoice-bugfix:0", "keystrokes": "ls"}).status_code == 400
    assert client.post("/api/run", json={"pivot": p, "source": "student:nobody:9"}).status_code == 400
    assert client.post("/api/run", json={"pivot": p, "keystrokes": "a\x00b"}).status_code == 400
    assert client.get("/api/pivot/nope:1").status_code == 404


def test_sandbox_has_no_network_or_capabilities(client):
    r = run(client, pivot="billing-invoice-bugfix:3", verify=False,
            keystrokes="grep CapEff /proc/self/status\nchroot / true || echo chroot-denied\n"
                       "mount -t tmpfs none /mnt 2>/dev/null || echo mount-denied\n"
                       "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), 2)\" 2>&1 | tail -n 1\n"
                       "ls /home /root | wc -l\nenv | grep -c -i -E 'token|secret|proxy|key'")
    out = r["terminal"]["yours"]
    assert "CapEff:\t0000000000000000" in out
    assert "chroot-denied" in out and "mount-denied" in out
    assert "Network is unreachable" in out
    assert out.rstrip().splitlines()[-2].strip() == "0"  # the server's environment (tokens, proxies) stays out


def test_output_is_bounded(client):
    r = run(client, pivot="billing-invoice-bugfix:3", verify=False, keystrokes="head -c 30000000 /dev/zero | tr '\\0' y")
    assert len(r["terminal"]["yours"]) < 10000 and r["X"]["failure"] is None


def test_busy_returns_429(client, core):
    core._acquire()
    core._acquire()  # both slots taken
    try:
        r = client.post("/api/run", json={"pivot": "billing-invoice-bugfix:3", "keystrokes": "ls"})
        assert r.status_code == 429 and r.headers["Retry-After"] == "5"
    finally:
        core._release()
        core._release()
    assert client.post("/api/run", json={"pivot": "billing-invoice-bugfix:3", "keystrokes": "ls",
                                         "verify": False}).status_code == 200


def test_scratch_layers_are_reclaimed(client, core):
    import os

    from cleave.world import overlay

    run(client, pivot="billing-invoice-bugfix:3", source="ctrl:flipped")
    st = overlay._store(core.store)
    live = {layer for name in os.listdir(os.path.join(st.root, "checkpoints"))
            for layer in st.load_checkpoint("overlay:" + name)}
    left = set(os.path.join(st.root, "layers", n) for n in os.listdir(os.path.join(st.root, "layers")))
    assert left <= live, f"{len(left - live)} layers leaked"
