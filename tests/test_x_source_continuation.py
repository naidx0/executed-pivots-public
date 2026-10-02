"""H21: a missed source file passes when both worlds converge after the expert's next step.

H20 zeroed X for any byte difference in a source file the expert edited, which also zeroed correct alternative
edits (H17 textstats t4 `hand-close:0`: requires-python >=3.11 where the expert wrote >=3.9). H21 continues both
worlds with the expert's next recorded step: the equivalent pin installs and runs the same way, while H17's
false credit `hand-wrong:4` (pip --ignore-requires-python, pin left at >=3.12) fails the next step's reinstall.

Pure unit test: the sandbox is a small model of the textstats specimen (probe.execute replaced), no Linux needed.
"""
import base64
import hashlib
import json

import pytest

from pivots.effect import EffectJudge, Pivot
from pivots.effect import xreward
from pivots.effect.probe import Effects

ROOT = "/app/textstats"
SOURCE = f"{ROOT}/pyproject.toml"
ARTIFACTS = [f"{ROOT}/build/lib/textstats/{n}" for n in ("__init__.py", "cli.py", "core.py")] + \
    [f"{ROOT}/src/textstats.egg-info/{n}" for n in ("SOURCES.txt", "top_level.txt")] + \
    [f"{ROOT}/build/bdist.linux-x86_64/wheel/textstats/core.cpython-311.pyc", f"{ROOT}/dist/textstats-1.2.0.whl"]
PKG_INFO = f"{ROOT}/src/textstats.egg-info/PKG-INFO"  # carries Requires-Python
STOPWORDS = f"{ROOT}/build/lib/textstats/stopwords.txt"
PRE = {ROOT, SOURCE, f"{ROOT}/src", f"{ROOT}/README.md"}


def h(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def pyproject(pin: str, pd: bool, drop_pin: bool = False) -> str:
    t = '[project]\nname = "textstats"\n' + ("" if drop_pin else f'requires-python = ">={pin}"\n')
    return t + ('\n[tool.setuptools.package-data]\ntextstats = ["stopwords.txt"]\n' if pd else "")


def fake_execute(world, anchor, script, *, cwd, cmd_timeout=30, network=False, sweep=True, extra_roots=()):
    """The textstats world at turn 4: pin >=3.12 in a Python 3.11 venv, stopwords.txt not packaged yet."""
    pin, pd, dropped, installed, out = "3.12", False, False, None, ""
    for line in script.splitlines():
        if line.startswith(f"echo {xreward.NEXT_MARK}"):
            out += xreward.NEXT_MARK + "\n"
        elif line.startswith("sed") and "requires-python/d" in line:
            dropped = True
        elif line.startswith("sed"):
            pin = "3.11" if "3.11" in line else "3.9" if "3.9" in line else pin
        elif line.startswith("printf") and "package-data" in line:
            pd = True
        elif line.startswith("printf") and xreward.NEXT_MARK in line:  # the key-set fallback's file dump
            txt = pyproject(pin, pd, dropped).encode()
            out += f"\n{xreward.NEXT_MARK}F {SOURCE}\n{base64.b64encode(txt).decode()}"
        if "pip install" in line:
            if "--ignore-requires-python" in line or pin != "3.12" or dropped:
                installed = (pin, pd)
                out += "Successfully installed textstats-1.2.0\n"
            else:
                out += "ERROR: Package 'textstats' requires a different Python: 3.11.15 not in '>=3.12'\n"
        if "textstats /data" in line:
            if installed and installed[1]:
                out += "file=sample.txt lines=5 words=77\ntop: every 3 keeper 3 lighthouse 3\n"
            elif installed:
                out += "Traceback (most recent call last):\nFileNotFoundError: stopwords.txt\n"
            else:
                out += "bash: /app/venv/bin/textstats: No such file or directory\n"
    changed = {}
    if pin != "3.12" or pd or dropped:
        changed[SOURCE] = h(pyproject(pin, pd, dropped))
    if installed:
        changed.update({p: h(p) for p in ARTIFACTS})
        changed[PKG_INFO] = h(f"pkg-info {installed[0]}")
        if installed[1]:
            changed[STOPWORDS] = h("stopwords")
    return Effects(out, 0, "/tmp", changed, PRE | set(changed), roots=["/app", "/tmp"])


def act(*cmds, tc=False):
    return json.dumps({"analysis": "a", "plan": "p", "task_complete": tc,
                       "commands": [{"keystrokes": c, "duration": 0.5} for c in cmds]})


INSTALL = "/app/venv/bin/pip install --no-index --no-build-isolation . 2>&1 | tail -n 2\n"
RUN = "cd /tmp && /app/venv/bin/textstats /data/sample.txt 2>&1 | tail -n 4\n"
EXPERT_T4 = act("sed -i 's/^requires-python = \">=3.12\"/requires-python = \">=3.9\"/' pyproject.toml\n", INSTALL, RUN)
EXPERT_T5 = act("printf '\\n[tool.setuptools.package-data]\\ntextstats = [\"stopwords.txt\"]\\n' >> "
                "/app/textstats/pyproject.toml\n",
                "/app/venv/bin/pip install --no-index --no-build-isolation /app/textstats 2>&1 | tail -n 2\n",
                "/app/venv/bin/textstats /data/sample.txt\n")
HAND_CLOSE_0 = act("sed -i 's/>=3.12/>=3.11/' pyproject.toml\n", INSTALL, RUN)  # equivalent pin
HAND_WRONG_4 = act("/app/venv/bin/pip install --no-index --no-build-isolation --ignore-requires-python . 2>&1 "
                   "| tail -n 2\n", RUN)  # H17's false credit: the pin is left at >=3.12
DROP_PIN = act("sed -i '/requires-python/d' pyproject.toml\n", INSTALL, RUN)


@pytest.fixture(autouse=True)
def fake_sandbox(monkeypatch):
    monkeypatch.setattr(xreward, "execute", fake_execute)
    monkeypatch.setattr(xreward, "listing_of", lambda world, anchor, cwd, roots: set(PRE))


def pivot(**kw) -> Pivot:
    return Pivot(f"textstats-t4-{sorted(kw.items())}", "anchor-t4", "/app/textstats", EXPERT_T4, **kw)


def test_equivalent_source_edit_that_converges_after_the_next_step_gets_credit():
    s = EffectJudge(world=None, continuation=True).score(pivot(next_answer=EXPERT_T5), HAND_CLOSE_0)
    assert s.detail["x_gated_reward"] == 0.0  # H20 zeroed it
    assert s.detail["source_converged"]["ok"] is True
    assert "source_missed" not in s.penalties
    assert s.binary == 1.0 and s.reward >= 0.75


def test_h17_hand_wrong_4_stays_zero_because_the_next_step_diverges():
    s = EffectJudge(world=None, continuation=True).score(pivot(next_answer=EXPERT_T5), HAND_WRONG_4)
    assert s.detail["x_ungated_reward"] >= 0.75  # the pre-H20 false credit (0.867 in H17)
    assert s.detail["source_converged"]["ok"] is False
    assert s.detail["source_converged"]["next_f1"] < s.detail["source_converged"]["expert_next_f1"]
    assert s.reward == 0.0 and s.binary == 0.0


def test_expert_itself_still_scores_one():
    s = EffectJudge(world=None, continuation=True).score(pivot(next_answer=EXPERT_T5), EXPERT_T4)
    assert s.reward == 1.0 and "source_converged" not in s.detail


def test_next_step_unknown_keeps_the_h20_gate():
    s = EffectJudge(world=None, continuation=True).score(pivot(), HAND_CLOSE_0)  # the Gym path: no trajectory
    assert s.reward == 0.0 and s.detail["source_converged"]["why"] == "next_step_unknown"


def test_last_step_toml_passes_on_the_same_key_set_only():
    j = EffectJudge(world=None, continuation=True)
    same = j.score(pivot(final=True), HAND_CLOSE_0)
    assert same.detail["source_converged"] == {"ok": True, "mode": "keyset"} and same.binary == 1.0
    dropped = j.score(pivot(final=True), DROP_PIN)
    assert dropped.detail["x_ungated_reward"] >= 0.75
    assert dropped.detail["source_converged"]["ok"] is False and dropped.reward == 0.0


def test_key_set_parses_toml_json_ini():
    assert xreward.key_set("a.toml", '[p]\nx = 1\n') == {"p", "p.x"}
    assert xreward.key_set("a.json", '{"a": {"b": [1]}}') == {"a", "a.b"}
    assert xreward.key_set("a.ini", "[s]\nk = v\n") == {"s", "s.k"}
    assert xreward.key_set("a.toml", "not = = toml") is None
    assert xreward.key_set("a.py", "x = 1") is None


def test_h23_default_is_the_h20_gate():
    """H23 reverted H21: by default a missed source file keeps X=0 even when the next step converges."""
    assert EffectJudge(world=None).continuation is False
    s = EffectJudge(world=None).score(pivot(next_answer=EXPERT_T5), HAND_CLOSE_0)
    assert s.detail["x_ungated_reward"] >= 0.75 and "source_converged" not in s.detail
    assert "source_missed" in s.penalties and s.reward == 0.0 and s.binary == 0.0
