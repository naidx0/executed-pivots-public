"""Red team (H11) and the H10 run: a directory whose ctime moved only because __pycache__ appeared is no effect.

Importing a module creates tests/__pycache__ (pruned by the probe), which moves tests/'s ctime. Both the
teacher and a wrong candidate that ran python then "changed" tests/, and that shared entry lifted the
wrong candidate's state score to 0.75 (0.8 overall with matching output).
"""
import json
import os

import pytest

from cleave.world.overlay import OverlayWorld, available
from pivots.effect import EffectJudge, Pivot
from pivots.effect.probe import Effects

H = {c: c * 64 for c in "abc"}


def eff(changed, listing):
    return Effects("ok", 0, "/app", dict(changed), set(listing), roots=["/app"])


def test_directory_with_same_entries_is_not_an_effect():
    pre = {"/app", "/app/tests", "/app/tests/test_a.py", "/app/report.csv"}
    now = pre | {"/app/tests/__pycache__"}
    e = eff({"/app/tests": "D", "/app/report.csv": H["a"]}, now)
    assert e.structural_changes(pre) == {"/app/report.csv": H["a"]}
    # a directory that lost an entry, or gained an empty subdirectory, still counts
    assert "/app/tests" in eff({"/app/tests": "D"}, pre - {"/app/tests/test_a.py"}).structural_changes(pre)
    assert "/app" in eff({"/app": "D"}, pre | {"/app/new"}).structural_changes(pre)


def test_shared_pycache_parent_no_longer_lifts_a_wrong_edit():
    j = EffectJudge(world=None)
    pre = {"/app", "/app/tests", "/app/tests/test_a.py", "/app/report.csv"}
    now = pre | {"/app/tests/__pycache__"}
    ref = eff({"/app/tests": "D", "/app/report.csv": H["a"]}, now)
    wrong = eff({"/app/tests": "D", "/app/report.csv": H["b"]}, now)
    c = j._compare(ref, wrong, pre, {}, {})
    assert c["state"] == 0.5 and c["reward"] < 0.75  # was state 0.75, reward 0.8
    assert j._compare(ref, eff(ref.changed, now), pre, {}, {})["reward"] == 1.0


def test_a_teacher_that_only_ran_python_keeps_its_directory_trace():
    # expert step: run the test suite; its only file-level trace is tests/ (tests/__pycache__ appeared)
    j = EffectJudge(world=None)
    pre = {"/app", "/app/tests", "/app/tests/test_a.py"}
    now = pre | {"/app/tests/__pycache__"}
    ref = eff({"/app/tests": "D"}, now)
    ran_tests = j._compare(ref, eff({"/app/tests": "D"}, now), pre, {}, {})
    assert ran_tests["reward"] == 1.0 and ran_tests["state"] == 1.0
    idle = j._compare(ref, eff({}, pre), pre, {}, {})
    assert idle["state"] == 0.0 and idle["reward"] < 0.75


@pytest.mark.skipif(not available(), reason="needs user namespaces + overlayfs")
def test_wrong_output_after_running_the_tests_is_not_credited():
    w = OverlayWorld(store=f"/tmp/cleave-redteam-test-{os.getpid()}", workdir="/app")
    assert w.run(["bash", "-c", "mkdir -p /app/tests && cd /app && printf 'X = 1\\n' > tests/helper.py"
                  " && printf 'import helper\\n' > tests/test_a.py"]).ok
    wid = w.checkpoint()

    def act(*cmds):
        return json.dumps({"analysis": "a", "plan": "p", "task_complete": False,
                           "commands": [{"keystrokes": c, "duration": 0.1} for c in cmds]})

    work = "cd tests && python3 test_a.py && cd .. && echo 42 > report.csv\n"
    pv = Pivot("rt-pycache", wid, "/app", act(work))
    j = EffectJudge(w)
    ref = j.prepare(pv)["ref"]
    assert "/app/tests" in ref.meaningful_changes()  # the raw probe still sees the ctime move
    assert j.score(pv, act(work)).binary == 1
    wrong = j.score(pv, act(work, "echo 41 > report.csv\n"))
    assert wrong.binary == 0 and wrong.state == 0.5
