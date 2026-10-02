"""H20: X weights files by origin. A source file the expert's step edited must match, however many generated
artifacts (build/, *.egg-info, site-packages, ...) the candidate reproduces.

The specimen is H17's one X false credit (textstats-pyproject-install turn 4, student hand-wrong:4): the student
installs with --ignore-requires-python and leaves the pyproject.toml pin, so 8 of the expert's 9 changed files
(build outputs) match and X scored 0.867. Pure unit test: effects are built by hand, no sandbox.
"""
import hashlib
import json

from pivots.effect import EffectJudge
from pivots.effect.probe import Effects

ROOT = "/app/textstats"
SOURCE = f"{ROOT}/pyproject.toml"
ARTIFACTS = [f"{ROOT}/build/lib/textstats/{n}" for n in ("__init__.py", "cli.py", "core.py")] + \
    [f"{ROOT}/src/textstats.egg-info/{n}" for n in ("PKG-INFO", "SOURCES.txt", "top_level.txt")] + \
    [f"{ROOT}/build/bdist.linux-x86_64/wheel/textstats/core.cpython-311.pyc", f"{ROOT}/dist/textstats-1.2.0.whl"]
PRE = {ROOT, SOURCE, f"{ROOT}/src", f"{ROOT}/README.md"} | set(ARTIFACTS[:3])
OUT = "Installing collected packages: textstats\nSuccessfully installed textstats-1.2.0\n"


def h(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def eff(changed: dict[str, str]) -> Effects:
    return Effects(OUT, 0, "/tmp", changed, PRE | set(changed), roots=["/app", "/tmp"])


def score(ref: Effects, cand: Effects) -> float:
    exp = {"task_complete": False}
    return EffectJudge(world=None)._compare(ref, cand, PRE, exp, exp)["reward"]


REF = eff({SOURCE: h("pin-fixed"), **{p: h(p) for p in ARTIFACTS}})


def test_matching_artifacts_do_not_cover_a_missed_source_edit():
    cand = eff({p: h(p) for p in ARTIFACTS})  # the pin left as it was
    assert score(REF, cand) == 0.0


def test_a_different_source_edit_is_not_credited():
    cand = eff({SOURCE: h("pin-removed"), **{p: h(p) for p in ARTIFACTS}})
    assert score(REF, cand) == 0.0


def test_matching_source_edit_keeps_full_credit():
    assert score(REF, eff(dict(REF.changed))) == 1.0


def test_artifacts_may_differ_when_the_source_edit_matches():
    cand = eff({SOURCE: h("pin-fixed"), **{p: h(p + "-rebuilt") for p in ARTIFACTS}})
    assert 0 < score(REF, cand) < 1  # the old partial credit, not the gate


def test_source_files_are_preexisting_non_artifact_files():
    from pivots.effect.xreward import source_files
    extra = {f"{ROOT}/new_script.sh": h("new"), f"{ROOT}/src": "D", "/app/venv/lib/python3.11/site-packages/x.py": h("x"),
             f"{ROOT}/lib/fast.o": h("o"), f"{ROOT}/src/textstats/__pycache__/cli.cpython-311.pyc": h("pyc")}
    got = source_files(eff({**REF.changed, **extra}), PRE)
    assert got == {SOURCE: h("pin-fixed")}


def test_gate_is_reported_as_a_penalty():
    cand = eff({p: h(p) for p in ARTIFACTS})
    exp = {"task_complete": False}
    c = EffectJudge(world=None)._compare(REF, cand, PRE, exp, exp)
    assert c["penalties"].get("source_missed") == 0.0
    assert c["detail"]["source_missed"] == [SOURCE]
    assert json.dumps(c)  # serializable, like every other detail field
