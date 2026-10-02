"""Token Factory student client, against a mock OpenAI-compatible server on localhost (no network, no key).

Covers: success, 429 then success, 5xx give-up, 402/401/403 and quota stops (never retried, sticky),
the request budget, the on-disk cache, egress refusal, the key never reaching disk, the Gym proxy, and
one audited pivot end to end through `pivots.audit.run --student tokenfactory:<model>`.
"""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from pivots.students import tokenfactory as tf
from pivots.students.mock_server import ACTIONS, MockTokenFactory
from pivots.students.sampler import LiveSampler, parse_spec

KEY = "sk-test-SENTINEL-4f1c9a"  # a fake key: the tests check it never leaves the Authorization header
MSGS = [{"role": "user", "content": "pivot prompt"}]


@pytest.fixture
def mock():
    servers = []

    def make(script=None, **kw):
        m = MockTokenFactory(script, **kw).start()
        servers.append(m)
        return m

    yield make
    for m in servers:
        m.stop()


def client(m, tmp_path, **kw):
    sleeps = []
    c = tf.TokenFactory("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", m.url, api_key=KEY,
                        cache_dir=str(tmp_path / "cache"), sleep=sleeps.append, **kw)
    return c, sleeps


def _no_key_on_disk(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            assert KEY not in p.read_text(errors="replace"), p


def test_success(mock, tmp_path):
    m = mock()
    c, _ = client(m, tmp_path)
    [s] = c.sample(MSGS, 1)
    assert s["text"] == ACTIONS[0] and s["reasoning"] == "mock reasoning 0" and s["cached"] is False
    assert m.calls[-1]["auth"] == f"Bearer {KEY}" and m.calls[-1]["model"] == c.model
    assert c.requests == 1
    _no_key_on_disk(tmp_path)


def test_429_then_success(mock, tmp_path):
    m = mock([429, 200])
    c, sleeps = client(m, tmp_path)
    out = c.chat(MSGS)
    assert out["choices"][0]["message"]["content"] == ACTIONS[0]
    assert [x["status"] for x in m.calls] == [429, 200] and c.requests == 2
    assert sleeps == [0.0]  # the mock's Retry-After: 0 is honoured
    ledger = [json.loads(line) for line in open(tmp_path / "cache" / "requests.jsonl")]
    assert [r["status"] for r in ledger] == [429, 200]


def test_5xx_retries_then_gives_up(mock, tmp_path):
    m = mock([503, 500, 503])
    c, sleeps = client(m, tmp_path, max_retries=2)
    with pytest.raises(tf.TokenFactoryError) as e:
        c.chat(MSGS)
    assert not isinstance(e.value, tf.TokenFactoryStop) and "gave up after 3 attempts" in str(e.value)
    assert len(m.calls) == 3 and len(sleeps) == 2
    assert c.chat(MSGS)["choices"]  # not sticky: the next call goes through


@pytest.mark.parametrize("status,kind", [(402, "billing_or_quota"), (401, "http_401"), (403, "http_403")])
def test_stop_statuses_are_final(mock, tmp_path, status, kind):
    m = mock([status, 200, 200])
    c, sleeps = client(m, tmp_path)
    with pytest.raises(tf.TokenFactoryStop) as e:
        c.chat(MSGS)
    assert e.value.kind == kind and sleeps == [] and len(m.calls) == 1
    with pytest.raises(tf.TokenFactoryStop):  # sticky: no further request reaches the server
        c.chat([{"role": "user", "content": "another"}])
    with pytest.raises(tf.TokenFactoryStop):
        c.list_models()
    assert len(m.calls) == 1 and c.requests == 1
    assert KEY not in str(e.value)


def test_402_stop_message(mock, tmp_path):
    m = mock([402])
    c, _ = client(m, tmp_path)
    with pytest.raises(tf.TokenFactoryStop) as e:
        c.chat(MSGS)
    assert e.value.kind == "billing_or_quota" and "HTTP 402" in str(e.value) and "Insufficient balance" in str(e.value)


def test_quota_429_stops_but_rate_limit_429_retries(mock, tmp_path):
    m = mock([(429, {"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}})])
    c, sleeps = client(m, tmp_path)
    with pytest.raises(tf.TokenFactoryStop) as e:
        c.chat(MSGS)
    assert e.value.kind == "billing_or_quota" and sleeps == [] and len(m.calls) == 1
    assert tf.classify(429, '{"error": {"message": "Too many requests"}}') == "retry"
    assert tf.classify(400, '{"error": {"message": "context too long"}}') == "fail"
    assert tf.classify(400, '{"error": {"message": "billing account suspended"}}') == "stop"


def test_cache(mock, tmp_path):
    m = mock()
    c, _ = client(m, tmp_path)
    a = c.sample(MSGS, 2)
    assert [s["cached"] for s in a] == [False, False] and a[0]["text"] != a[1]["text"] and len(m.calls) == 2
    c2, _ = client(m, tmp_path)  # a fresh process: same cache dir, no requests
    b = c2.sample(MSGS, 2)
    assert [s["cached"] for s in b] == [True, True] and [s["text"] for s in b] == [s["text"] for s in a]
    assert len(m.calls) == 2 and c2.requests == 0 and c2.cache_hits == 2
    c2.chat(MSGS, sample=0, temperature=0.2)  # other params: another key
    assert len(m.calls) == 3
    assert len(list((tmp_path / "cache").glob("*.json"))) == 3
    _no_key_on_disk(tmp_path)


def test_budget_is_hard(mock, tmp_path):
    m = mock([429, 200, 200])
    c, _ = client(m, tmp_path, max_requests=3)
    c.chat(MSGS, sample=0)  # 2 requests (429 then 200)
    c.chat(MSGS, sample=1)  # 3
    c.chat(MSGS, sample=0)  # cache hit: free
    with pytest.raises(tf.BudgetExhausted):
        c.chat(MSGS, sample=2)
    assert len(m.calls) == 3 and c.requests == 3


def test_egress_refusal_stops(monkeypatch, tmp_path):
    calls = []

    def refuse(req, timeout=None):
        calls.append(req.full_url)
        raise urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    c = tf.TokenFactory(api_key=KEY, cache_dir=str(tmp_path), sleep=lambda s: None)
    with pytest.raises(tf.TokenFactoryStop) as e:
        c.list_models()
    assert e.value.kind == "egress_blocked" and len(calls) == 1


def test_key_from_env_and_redaction(monkeypatch, tmp_path):
    monkeypatch.setenv("NEBIUS_API_KEY", KEY)
    assert tf.load_key(str(tmp_path / "absent")) == KEY
    monkeypatch.delenv("NEBIUS_API_KEY")
    (tmp_path / "k").write_text(KEY + "\n")
    assert tf.load_key(str(tmp_path / "k")) == KEY
    with pytest.raises(tf.TokenFactoryStop):
        tf.load_key(str(tmp_path / "absent"))
    c = tf.TokenFactory(api_key=KEY, cache_dir=None)
    assert KEY not in c.redact(f"echo {KEY} Authorization: Bearer {KEY}")


def test_list_models_and_cli(mock, tmp_path, monkeypatch, capsys):
    m = mock()
    monkeypatch.setenv("NEBIUS_API_KEY", KEY)
    rc = tf.main(["--list-models", "--base-url", m.url, "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / "m.json")])
    assert rc == 0 and "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B" in capsys.readouterr().out
    rc = tf.main(["--smoke", "--base-url", m.url, "--cache-dir", str(tmp_path / "c"), "--out", str(tmp_path / "s.json")])
    assert rc == 0 and json.loads((tmp_path / "s.json").read_text())["stats"]["requests"] == 1
    m.script[:] = [402]
    rc = tf.main(["--smoke", "--model", "other", "--base-url", m.url, "--cache-dir", str(tmp_path / "c")])
    assert rc == 3 and "STOP" in capsys.readouterr().err
    _no_key_on_disk(tmp_path)


def test_split_reasoning():
    assert tf.split_reasoning({"content": "<think>hmm</think>\n{\"a\": 1}"}) == ('{"a": 1}', "hmm")
    assert tf.split_reasoning({"content": "hmm</think>{}"}) == ("{}", "hmm")
    assert tf.split_reasoning({"content": "{}", "reasoning_content": "r"}) == ("{}", "r")


def test_spec_and_pool():
    assert parse_spec("tokenfactory:nvidia/X-1") == ("tokenfactory", "nvidia/X-1")
    assert parse_spec("tokenfactory") == ("tokenfactory", tf.DEFAULT_MODEL)
    with pytest.raises(ValueError):
        parse_spec("openai:gpt")
    assert tf.pool_name("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B") == "tf-nvidia-nemotron-3-nano-30b-a3b"


def test_sampler_writes_students_rows(mock, tmp_path):
    m = mock([429])
    c, _ = client(m, tmp_path)

    class Sp:
        name = "t"

    s = LiveSampler("tokenfactory:x", 3, str(tmp_path / "out" / "students.jsonl"), client=c, concurrency=2)
    got = s(Sp, [(0, MSGS), (2, [{"role": "user", "content": "other"}])])
    assert sorted(got) == [0, 2] and all(len(v) == 3 for v in got.values())
    rows = [json.loads(line) for line in open(tmp_path / "out" / "students.jsonl")]
    assert len(rows) == 6 and {r["pool"] for r in rows} == {s.pool} and all(r["text"] for r in rows)
    assert s.stats()["requests"] == 7 and s.stats()["samples"] == 6


def test_proxy_forwards_and_stops(mock, tmp_path):
    m = mock()
    c, _ = client(m, tmp_path, max_requests=3)
    srv = tf.make_proxy(c, port=0)
    import threading

    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v1/chat/completions"

    def post():
        req = urllib.request.Request(url, json.dumps({"model": "ignored", "messages": MSGS}).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        (s1, a), (s2, b) = post(), post()
        assert s1 == s2 == 200 and a["choices"][0]["message"]["content"] != b["choices"][0]["message"]["content"]
        assert m.calls[-1]["auth"] == f"Bearer {KEY}"  # the proxy holds the key, Gym sends a dummy
        m.script[:] = [402]
        s3, err = post()
        assert s3 == 429 and err["error"]["code"] == "budget_exceeded" and err["error"]["type"] == "billing_or_quota"
        n = len(m.calls)
        assert post()[0] == 429 and len(m.calls) == n  # sticky
    finally:
        srv.shutdown()
        srv.server_close()


def test_audit_with_live_students(mock, tmp_path, capsys):
    """One pivot through the whole audit: sample 2 students from the mock, label J/X/E and P."""
    from cleave.world.overlay import available

    if not available():
        pytest.skip("needs user namespaces + overlayfs")
    from pivots.audit import run

    m = mock()
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "audit"
    store = f"/tmp/cleave-tf-audit-test-{os.getpid()}"
    os.environ["NEBIUS_API_KEY"] = KEY
    try:
        run.main([str(root / "specimens" / "billing-invoice-bugfix"), "--pivots", "billing-invoice-bugfix:3",
                  "--student", "tokenfactory:nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "--n-students", "2",
                  "--tf-base-url", m.url, "--tf-cache", str(tmp_path / "cache"), "--out", str(out),
                  "--store", store, "--workers", "2", "--progress"])
    finally:
        os.environ.pop("NEBIUS_API_KEY", None)
        import shutil

        shutil.rmtree(store, ignore_errors=True)
    rows = [json.loads(line) for line in open(out / "rows.jsonl")]
    students = [r for r in rows if r["action"].startswith("student:tf-nvidia-nemotron-3-nano-30b-a3b:")]
    assert len(students) == 2 and {r["turn"] for r in rows} == {3}
    assert all({"J", "X", "E"} <= set(r) for r in students)
    prog = [json.loads(line) for line in open(out / "progress.jsonl")]
    assert len(prog) == len(rows) and all("P" in r for r in prog)
    summ = json.loads((out / "summary.json").read_text())
    assert summ["live_students"]["requests"] == 2 and "student:tf-nvidia-nemotron-3-nano-30b-a3b" in summ
    assert len(m.calls) == 2
    _no_key_on_disk(tmp_path)
