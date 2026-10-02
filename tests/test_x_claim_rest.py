"""H49: an early completion claim is checked against the state the expert's rest of the episode leaves.

X zeroes a step that sets task_complete where the expert kept working (premature_complete). With the expert's later
actions on the pivot (Pivot.rest_answers), the penalty is dropped when the candidate's run already holds the final
state: every path the expert's step and later steps change or delete, with the same bytes (directories, generated
artifacts, /tmp scratch and bytes that differ between two rest runs left out), and every process they leave running.

Pure unit test: probe.execute is replaced by a small model of a project, no Linux needed.
"""
import hashlib
import itertools
import json

import pytest

from pivots.effect import EffectJudge, Pivot
from pivots.effect import xreward
from pivots.effect.probe import Effects

OUT = "/app/reports/out.csv"
BRANCH = "/app/.git/refs/heads/release"
STAMP = "/app/reports/stamp"
PRE = {"/app", "/app/run.py", "/app/reports", "/app/.git", "/app/old.log"}
_tick = itertools.count()


def h(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def fake_execute(world, anchor, script, *, cwd, cmd_timeout=30, network=False, sweep=True, extra_roots=()):
    out, changed, listing, procs = "", {}, set(PRE), []
    for line in script.splitlines():
        part = line.strip()
        if part == "python3 run.py":
            changed[OUT] = h("good")
            changed["/app/reports"] = "D"
            listing.add(OUT)
            out += "wrote 3 rows\n"
        elif part == "python3 run.py --fix":
            changed[OUT] = h("fixed")
        elif part == "python3 run.py --bad":
            changed[OUT] = h("bad")
            listing.add(OUT)
        elif part == "git branch release":
            changed[BRANCH] = h("sha")
        elif part == "cat reports/out.csv":
            out += "a,b\n1,2\n"
        elif part == "python3 -m check":
            changed["/app/__pycache__/check.cpython-311.pyc"] = h(f"pyc{next(_tick)}")
            changed["/tmp/check.txt"] = h(f"tmp{next(_tick)}")
            out += "ok\n"
        elif part == "date > reports/stamp":
            changed[STAMP] = h(f"stamp{next(_tick)}")
        elif part == "rm -f old.log":
            listing.discard("/app/old.log")
        elif part == "serve &":
            procs.append("python3 -m http.server 8000")
    return Effects(out, 0, cwd, changed, listing, roots=["/app"], procs=procs)


@pytest.fixture(autouse=True)
def fake_sandbox(monkeypatch):
    monkeypatch.setattr(xreward, "execute", fake_execute)
    monkeypatch.setattr(xreward, "listing_of", lambda world, anchor, cwd, roots: set(PRE))


def act(*cmds, tc=False):
    return json.dumps({"analysis": "a", "plan": "p", "task_complete": tc,
                       "commands": [{"keystrokes": c, "duration": 0.5} for c in cmds]})


RUN = act("python3 run.py\n")                 # the expert's step: it does not claim
BARE = act(tc=True)


def pivot(*rest):
    return Pivot("p", "anchor", "/app", RUN, next_answer=rest[0] if rest else None, final=not rest,
                 rest_answers=tuple(rest))


def score(pv, text, **kw):
    return EffectJudge(world=None, check_determinism=False, **kw).score(pv, text)


@pytest.mark.parametrize("rest", [
    (act("cat reports/out.csv\n"), BARE),                        # later turns only read
    (act("python3 -m check\n"), BARE),                           # a check leaves a .pyc and /tmp scratch
    (BARE,),                                                     # the next turn is the bare claim
], ids=["reads", "check_scratch", "bare_next"])
def test_an_early_claim_that_already_holds_the_final_state_keeps_its_credit(rest):
    s = score(pivot(*rest), act("python3 run.py\n", tc=True))
    assert s.binary == 1.0 and "premature_complete" not in s.penalties
    assert s.detail["claim_rest"]["ok"] is True


@pytest.mark.parametrize("rest,cand,blocker", [
    ((act("git branch release\n"), BARE), act("python3 run.py\n", tc=True), "missing"),
    ((act("python3 run.py --fix\n"), BARE), act("python3 run.py\n", tc=True), "differ"),
    ((act("rm -f old.log\n"), BARE), act("python3 run.py\n", tc=True), "not_deleted"),
    ((act("serve &\n"), BARE), act("python3 run.py\n", tc=True), "procs"),
    ((act("date > reports/stamp\n"), BARE), act("python3 run.py\n", tc=True), "missing"),
], ids=["later_branch", "later_rewrite", "later_delete", "later_process", "later_volatile_file"])
def test_an_early_claim_that_leaves_later_work_undone_stays_at_zero(rest, cand, blocker):
    s = score(pivot(*rest), cand)
    assert s.binary == 0.0 and s.penalties.get("premature_complete") == 0.0
    assert s.detail["claim_rest"]["ok"] is False and s.detail["claim_rest"][blocker]


def test_a_volatile_file_needs_only_to_exist():
    rest = (act("date > reports/stamp\n"), BARE)
    s = score(pivot(*rest), act("python3 run.py\n", "date > reports/stamp\n", tc=True))
    assert s.detail["claim_rest"]["ok"] is True and not s.detail["claim_rest"]["differ"]


def test_the_step_itself_must_still_match():
    s = score(pivot(act("cat reports/out.csv\n"), BARE), act("python3 run.py --bad\n", tc=True))
    assert s.binary == 0.0


def test_without_the_rest_of_the_episode_the_penalty_stays():
    s = score(pivot(), act("python3 run.py\n", tc=True))
    assert s.binary == 0.0 and s.penalties.get("premature_complete") == 0.0
    assert s.detail["claim_rest"] == {"ok": False, "why": "no_rest"}


def test_switched_off_the_penalty_stays():
    s = score(pivot(act("cat reports/out.csv\n"), BARE), act("python3 run.py\n", tc=True), claim_rest=False)
    assert s.binary == 0.0 and s.penalties.get("premature_complete") == 0.0 and "claim_rest" not in s.detail


def test_a_step_that_does_not_claim_is_untouched():
    s = score(pivot(act("git branch release\n"), BARE), RUN)
    assert s.binary == 1.0 and "claim_rest" not in s.detail


def test_gym_core_reads_the_rest_of_the_episode_from_the_anchor():
    from pivots.gym.core import ExecutedPivotCore
    rest = [act("cat reports/out.csv\n"), BARE]
    anchors = {"u": {"uuid": "u", "anchor": "a", "cwd": "/app", "next_answer": rest[0], "rest_answers": rest},
               "v": {"uuid": "v", "anchor": "a", "cwd": "/app", "next_answer": rest[0]}}
    core = ExecutedPivotCore(None, anchors)
    early = act("python3 run.py\n", tc=True)
    assert core.verify({"uuid": "u", "expected_answer": RUN}, early)["reward"] == 1.0
    assert core.verify({"uuid": "v", "expected_answer": RUN}, early)["reward"] == 0.0
