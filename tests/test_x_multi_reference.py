"""H44: X compares a candidate with a per-pivot reference set, the expert's step plus earlier policy actions verified
at that pivot (E=1 and P=1). A candidate is credited when it matches any reference under the unchanged X rules.

Pure unit test: probe.execute is replaced by a small model of a project, no Linux needed. The expert fixes the bug
by editing calc.py in place; a verified earlier action fixed it by writing a patch file and applying it (another file
set, so the expert comparison misses it). A candidate that does what the verified action did is credited only with
the reference set on; a wrong candidate stays at 0 either way.
"""
import hashlib
import json

import pytest

from pivots.effect import EffectJudge, Pivot
from pivots.effect import xreward
from pivots.effect.probe import Effects
from pivots.effect.xreward import load_references
from pivots.gym.core import Config, ExecutedPivotCore

CALC = "/app/calc.py"
PATCH = "/app/fix.patch"
PRE = {"/app", CALC}


def h(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def fake_execute(world, anchor, script, *, cwd, cmd_timeout=30, network=False, sweep=True, extra_roots=()):
    out, changed, listing = "", {}, set(PRE)
    for line in script.splitlines():
        line = line.strip()
        if line.startswith("sed -i"):
            changed[CALC] = h("fixed")
        elif line.startswith("write-patch"):
            changed[PATCH] = h("patch")
            listing.add(PATCH)
        elif line.startswith("apply-patch"):
            changed[CALC] = h("fixed")
            out += "patching file calc.py\n"
        elif line.startswith("break"):
            changed[CALC] = h("broken")
    return Effects(out, 0, cwd, changed, listing, roots=["/app"])


@pytest.fixture(autouse=True)
def fake_sandbox(monkeypatch):
    monkeypatch.setattr(xreward, "execute", fake_execute)
    monkeypatch.setattr(xreward, "listing_of", lambda world, anchor, cwd, roots: set(PRE))


def act(*cmds, tc=False):
    return json.dumps({"analysis": "a", "plan": "p", "task_complete": tc,
                       "commands": [{"keystrokes": c, "duration": 0.5} for c in cmds]})


EXPERT = act("sed -i s/-/+/ calc.py\n")
VERIFIED = act("write-patch\n", "apply-patch\n")         # an earlier policy action with E=1 and P=1
SAME_AS_VERIFIED = act("write-patch && true\n", "apply-patch\n")
WRONG = act("write-patch\n", "break\n")
PIVOT = Pivot("calc:3", "anchor", "/app", EXPERT)


def test_expert_only_misses_the_other_working_route():
    s = EffectJudge(world=None).score(PIVOT, SAME_AS_VERIFIED)
    assert s.binary == 0.0


def test_a_verified_reference_credits_a_candidate_that_matches_it():
    j = EffectJudge(world=None, references={"calc:3": [VERIFIED]})
    s = j.score(PIVOT, SAME_AS_VERIFIED)
    assert s.binary == 1.0 and s.detail["reference"] == 0
    assert s.detail["x_expert_reward"] < 0.75
    assert j.score(PIVOT, EXPERT).binary == 1.0          # the expert's comparison still credits first


def test_references_do_not_credit_a_wrong_step_or_another_pivot():
    j = EffectJudge(world=None, references={"calc:3": [VERIFIED]})
    assert j.score(PIVOT, WRONG).binary == 0.0
    other = Pivot("calc:4", "anchor", "/app", EXPERT)
    assert j.score(other, SAME_AS_VERIFIED).binary == 0.0


def test_a_reference_that_does_not_reproduce_is_not_used(monkeypatch):
    runs = {"n": 0}

    def flaky(world, anchor, script, **kw):
        eff = fake_execute(world, anchor, script, **kw)
        if "write-patch" in script and "true" not in script:
            runs["n"] += 1
            eff.changed[PATCH] = h(f"patch-{runs['n']}")   # different bytes on each run of the reference
        return eff

    monkeypatch.setattr(xreward, "execute", flaky)
    j = EffectJudge(world=None, references={"calc:3": [VERIFIED]})
    assert j.score(PIVOT, SAME_AS_VERIFIED).binary == 0.0


def test_reference_file_accepts_only_e1_p1_rows(tmp_path):
    good = tmp_path / "refs.jsonl"
    good.write_text("".join(json.dumps(r) + "\n" for r in [
        {"uuid": "calc:3", "text": VERIFIED, "E": 1, "P": 1, "source": "h31#1"},
        {"uuid": "calc:3", "text": VERIFIED, "E": 1, "P": 1, "source": "h40#7"}]), encoding="utf-8")
    assert load_references(good) == {"calc:3": [VERIFIED]}
    for bad in ({"E": 0, "P": 1}, {"E": 1, "P": 0}, {"E": 1}, {"P": 1}):
        p = tmp_path / "bad.jsonl"
        p.write_text(json.dumps({"uuid": "calc:3", "text": VERIFIED, **bad}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_references(p)


def _server_files(tmp_path, with_refs: bool):
    anchor = {"uuid": "calc:3", "anchor": "anchor", "cwd": "/app"}
    (tmp_path / "anchors.jsonl").write_text(json.dumps(anchor) + "\n", encoding="utf-8")
    if with_refs:
        ref = {"uuid": "calc:3", "text": VERIFIED, "E": 1, "P": 1, "source": "h31#1"}
        (tmp_path / "references.jsonl").write_text(json.dumps(ref) + "\n", encoding="utf-8")
    return str(tmp_path / "anchors.jsonl")


def test_gym_core_uses_a_reference_file_next_to_the_anchors_by_default(tmp_path):
    core = ExecutedPivotCore.from_files(None, _server_files(tmp_path, True))
    assert core.verify({"uuid": "calc:3", "expected_answer": EXPERT}, SAME_AS_VERIFIED)["reward"] == 1.0
    assert core.verify({"uuid": "calc:3", "expected_answer": EXPERT}, WRONG)["reward"] == 0.0


def test_gym_core_without_a_reference_file_is_expert_only(tmp_path):
    core = ExecutedPivotCore.from_files(None, _server_files(tmp_path, False))
    assert core.verify({"uuid": "calc:3", "expected_answer": EXPERT}, SAME_AS_VERIFIED)["reward"] == 0.0
    with pytest.raises(FileNotFoundError):
        ExecutedPivotCore.from_files(None, _server_files(tmp_path, False), Config(references_path=str(tmp_path / "x")))
