"""End to end with no model: a fake local model server (canned Terminus-2 JSON, shaped the way thinking models
answer) -> the local-policy proxy -> pivots.gym.rollouts -> ExecutedPivotCore.verify (what executed_pivot's
/verify runs) at two pivots, then pivots.gym.labels adds E and P.

Checks that the expert's action earns reward 1 however the model wrapped it (think + fence, reasoning-only),
that a string-close action with the opposite effect earns X 0 while J pays it, that malformed JSON is paid 0
unmasked, and that a policy failure after the retries is masked as policy_error and never verified.
Needs the overlay backend (Linux: user namespaces + overlayfs); it runs in h24-runner:local, privileged.
"""
import json
import os
import shutil
import threading
from pathlib import Path

import pytest

from _fake_openai import FakeOpenAI
from cleave.world.overlay import OverlayWorld, available
from pivots.gym import labels as L
from pivots.gym.core import Config, ExecutedPivotCore
from pivots.gym.rollouts import collect
from pivots.students import local_policy as lp
from resources_servers.executed_pivot.scripts.prepare import pivot_rows

pytestmark = pytest.mark.skipif(not available(), reason="needs user namespaces + overlayfs")

REPO_ROOT = Path(__file__).resolve().parents[1]
PIVOTS = [("billing-invoice-bugfix", 3), ("ledger-git-revert", 3)]
FLIPPED = json.dumps({"analysis": "fix the threshold", "plan": "sed", "task_complete": False, "commands": [
    {"keystrokes": "sed -i 's/if qty > BULK_THRESHOLD:/if qty < BULK_THRESHOLD:/' billing/invoice.py\n", "duration": 0.1},
    {"keystrokes": "grep -n BULK_THRESHOLD billing/invoice.py\n", "duration": 0.1},
    {"keystrokes": "python3 -m unittest discover -s tests 2>&1 | tail -n 5\n", "duration": 0.1}]})
NOOP = json.dumps({"analysis": "wait", "plan": "nothing", "commands": [], "task_complete": False})


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("gym_local_rollouts")
    store = f"/tmp/gym-local-rollouts-test-{os.getpid()}"
    rows, anchors = [], []
    for task, turn in PIVOTS:
        r, a = pivot_rows(REPO_ROOT / "specimens" / task, store, turns=[turn])
        rows += r
        anchors += a
    (tmp / "anchors.jsonl").write_text("".join(json.dumps(a) + "\n" for a in anchors))
    billing, ledger = (r["expected_answer"] for r in rows)
    fake = FakeOpenAI([
        ("think", billing), ("reasoning", f"Flip it.\n{FLIPPED}"), ("content", billing[: len(billing) // 2]),
        ("reasoning_content", f"Commit the revert.\n{ledger}"), ("content", NOOP), ("status", 503), ("status", 503),
    ]).start()
    proxy = lp.make_proxy(lp.LocalPolicy(fake.url, "fake-local", retries=1, sleep=lambda s: None), port=0,
                          log_path=str(tmp / "policy.jsonl"))
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    core = ExecutedPivotCore.from_files(OverlayWorld(store=store, workdir=None), str(tmp / "anchors.jsonl"), Config())
    recs = collect(rows, 3, f"http://127.0.0.1:{proxy.server_port}/v1", core.verify, tmp / "rollouts.jsonl",
                   concurrency=1)
    summary = L.run(str(tmp / "rollouts.jsonl"), str(tmp / "labels"), f"/tmp/gym-local-labels-{os.getpid()}", workers=2)
    yield {"recs": recs, "summary": summary, "tmp": tmp}
    proxy.shutdown()
    fake.stop()
    for p in Path("/tmp").glob(f"gym-local-*-{os.getpid()}*"):
        shutil.rmtree(p, ignore_errors=True)


def test_rewards_through_verify(run):
    r = run["recs"]
    got = [(x["uuid"], x["k"], x["reward"], x["j_reward"], x["mask_sample"], x["failure_kind"]) for x in r]
    assert got[0] == ("billing-invoice-bugfix:3", 0, 1.0, 1.0, False, None)  # think + fenced expert
    assert got[1] == ("billing-invoice-bugfix:3", 1, 0.0, 1.0, False, None)  # flipped: J pays, X does not
    assert got[2][2:] == (0.0, 0.0, False, "executed_pivot:model_output_invalid")  # malformed JSON, not masked
    assert got[3] == ("ledger-git-revert:3", 0, 1.0, 1.0, False, None)  # reasoning-only expert
    assert got[4][2] == 0.0 and got[4][4] is False  # doing nothing where the expert commits
    assert got[5][4:] == (True, "executed_pivot:policy_error") and r[5]["text"] is None
    assert [x["policy"].get("source") for x in r[:5]] == ["content", "reasoning", "invalid", "reasoning", "content"]
    assert all(x["usage"]["completion_tokens"] == 20 for x in r[:5])


def test_labels_and_h31_summary(run):
    s = run["summary"]
    assert s["rollouts"] == 6 and s["labelled"] == 5 and s["masked"] == {"executed_pivot:policy_error": 1}
    lab = {(x["uuid"], x["i"]): x for x in map(json.loads, (run["tmp"] / "labels" / "labels.jsonl").read_text().splitlines())}
    assert lab[("billing-invoice-bugfix:3", 0)]["E"] == 1 and lab[("billing-invoice-bugfix:3", 1)]["E"] == 0
    assert lab[("billing-invoice-bugfix:3", 2)]["E"] == 0 and lab[("ledger-git-revert:3", 3)]["E"] == 1
    assert lab[("billing-invoice-bugfix:3", 0)]["P"] == 1 and lab[("billing-invoice-bugfix:3", 1)]["P"] == 0
    assert s["X_vs_E"]["false_credit"] == 0 and s["J_vs_E"]["false_credit"] >= 1
    assert s["X_vs_E"]["agree"] >= s["J_vs_E"]["agree"]
    assert s["h31"]["verdict"].startswith("inconclusive")  # 5 rollouts is far below the pre-registered 100
