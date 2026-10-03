"""H52: X compares permission bits.

A chmod leaves a file's bytes alone, so before H52 a candidate that chmod-ed the expert's files to the wrong mode
matched the expert's effect byte for byte (H50 tb-processing-pipeline:3, 2 X false credits at 0.8). The probe now
lists the mode of every watched file or directory whose ctime moved, and a path both runs touched must end with
the same mode: a mismatch counts as a half match, and a directory the structural view left out comes back as one.

Pure unit test: probe.execute is replaced by a small model of the pipeline task, no Linux needed.
"""
import hashlib
import json

import pytest

from pivots.effect import EffectJudge, Pivot
from pivots.effect import xreward
from pivots.effect.probe import MARK, Effects, parse_modes

SCRIPTS = ("run.sh", "collect.sh", "process.sh")
PRE = {"/app", "/data", "/data/output", *(f"/app/{s}" for s in SCRIPTS)}
START = {"/app/run.sh": "644", "/app/collect.sh": "311", "/app/process.sh": "644", "/data/output": "0"}


def h(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def chmod(modes: dict, spec: str, path: str) -> None:
    old = int(modes[path], 8)
    if spec.isdigit():
        new = int(spec, 8)
    else:  # the few symbolic forms the model uses
        who, op, what = spec[0] if spec[0] in "ugoa" else "a", "+" if "+" in spec else "-", spec.split("+")[-1]
        bits = sum({"r": 0o444, "w": 0o222, "x": 0o111}[c] for c in what)
        new = old | bits if op == "+" else old & ~bits
    modes[path] = oct(new)[2:]


def fake_execute(world, anchor, script, *, cwd, cmd_timeout=30, network=False, sweep=True, extra_roots=()):
    changed, modes, now = {}, {}, dict(START)
    for line in script.splitlines():
        part = line.split()
        if part[:1] == ["chmod"]:
            for f in part[2:]:
                p = f if f.startswith("/") else f"/app/{f}"
                chmod(now, part[1], p)
                changed[p] = "D" if p == "/data/output" else h(p)
                modes[p] = now[p]
        elif part[:2] == ["sed", "-i"]:
            changed["/app/process.sh"] = h("process.sh fixed")
            modes["/app/process.sh"] = now["/app/process.sh"]
    return Effects("", 0, cwd, changed, set(PRE), roots=["/app", "/data"], modes=modes)


@pytest.fixture(autouse=True)
def fake_sandbox(monkeypatch):
    monkeypatch.setattr(xreward, "execute", fake_execute)
    monkeypatch.setattr(xreward, "listing_of", lambda world, anchor, cwd, roots: set(PRE))


def act(*cmds):
    return json.dumps({"analysis": "a", "plan": "p", "task_complete": False,
                       "commands": [{"keystrokes": c + "\n", "duration": 0.1} for c in cmds]})


EXPERT = act("chmod 755 run.sh collect.sh process.sh", "sed -i s/x/y/ process.sh", "chmod 755 /data/output")


def score(text):
    pv = Pivot("p", "anchor", "/app", EXPERT, next_answer=act("./run.sh"))
    return EffectJudge(world=None, check_determinism=False).score(pv, text)


def test_the_expert_and_an_equal_mode_alternative_are_credited():
    assert score(EXPERT).binary == 1.0
    s = score(act("chmod a+rx run.sh collect.sh process.sh", "sed -i s/x/y/ process.sh", "chmod 755 /data/output"))
    assert s.binary == 1.0 and s.detail["mode_diff"] == []


@pytest.mark.parametrize("text,bad", [
    # H50 rollout 285: collect.sh keeps 311 (not readable), /data/output ends 555
    (act("chmod +x run.sh collect.sh process.sh", "sed -i s/x/y/ process.sh", "chmod a+rx /data/output"),
     {"/app/collect.sh", "/data/output"}),
    # H50 rollout 287: /data/output ends 222 (writable, not listable)
    (act("chmod +x run.sh process.sh collect.sh", "sed -i s/x/y/ process.sh", "chmod a+w /data/output"),
     {"/app/collect.sh", "/data/output"}),
    # only the directory is wrong, and its entries did not change
    (act("chmod 755 run.sh collect.sh process.sh", "sed -i s/x/y/ process.sh", "chmod 700 /data/output"),
     {"/data/output"}),
], ids=["h50_285", "h50_287", "dir_only"])
def test_a_wrong_mode_on_a_path_the_expert_touched_is_not_credited(text, bad):
    s = score(text)
    assert s.binary == 0.0
    assert set(s.detail["mode_diff"]) == bad


def test_parse_modes_reads_only_its_section():
    out = (f"{MARK} RC 0\n{MARK} CHANGED\n{'a' * 64}  /app/x\n{MARK} MODES\n755  /app/x\n0  /data/output\n"
           f"junk line\n{MARK} LISTING\n/app\n{MARK} DONE\n")
    assert parse_modes(out) == {"/app/x": "755", "/data/output": "0"}
    assert parse_modes(f"{MARK} RC 0\n{MARK} CHANGED\n{MARK} LISTING\n") == {}


def test_mode_gate_off_is_the_old_judge():
    pv = Pivot("p", "anchor", "/app", EXPERT, next_answer=act("./run.sh"))
    wrong = act("chmod +x run.sh collect.sh process.sh", "sed -i s/x/y/ process.sh", "chmod a+rx /data/output")
    s = EffectJudge(world=None, check_determinism=False, mode_gate=False).score(pv, wrong)
    assert s.binary == 1.0 and s.detail["mode_diff"] == [] and "mode_mismatch" not in s.penalties
