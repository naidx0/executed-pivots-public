"""H55: X compares each git repo's index (`git ls-files -s`).

The index file is probe noise (its stat data changes on every run), so before H55 a step that wrote a merge's
result by hand (MERGE_HEAD, MERGE_MSG and the conflicted files, via printf) matched the expert's `git merge` file for
file: H50 red team #306 at tb-fix-git:3 scored 0.8 with E=0, because the unmerged index entries the expert's later
steps resolve were never made. The probe now prints a hash of `git ls-files -s` for every repo under the roots, and a
repo whose index ends elsewhere than the expert's sets X to 0 (penalty git_index_mismatch, a gate like H52's).

The first tests drive probe.execute with a scripted world (no sandbox). The last one runs real git in an overlay
world (Linux only).
"""
import json
import os
from types import SimpleNamespace

import pytest

from pivots.effect import EffectJudge
from pivots.effect.probe import MARK, build_probe_script, execute, parse_gitstage

REPO = "/app/site"
PRE = {"/app", REPO, f"{REPO}/.git", f"{REPO}/about.md"}
ABOUT_MERGED = "c" * 64
IDX_PIVOT, IDX_MERGE = "1" * 64, "2" * 64


class ScriptedWorld:
    def __init__(self, stdout: str):
        self.stdout = stdout

    def fork(self, anchor):
        return self

    def close(self):
        pass

    def run(self, cmd, cwd=None, timeout_s=0, keep=True, network=None):
        return SimpleNamespace(stdout=self.stdout, stderr="", exit_code=0, duration_s=0.1)


def effects(index: str | None, changed=(f"{ABOUT_MERGED}  {REPO}/about.md", f"{'d' * 64}  {REPO}/.git/MERGE_HEAD")):
    stage = f"{index}  {REPO}\n" if index else ""
    out = (f"merging\n{MARK} END rc=0 pwd={REPO}\n{MARK} RC 0\n{MARK} CHANGED\n" + "".join(c + "\n" for c in changed)
           + f"{MARK} MODES\n{MARK} GITSTAGE\n{stage}{MARK} LISTING\n"
           + "".join(p + "\n" for p in sorted(PRE | {f"{REPO}/.git/MERGE_HEAD"})) + f"{MARK} DONE\n")
    return execute(ScriptedWorld(out), "anchor", "true\n", cwd=REPO, sweep=False)


def x(ref, cand, **kw) -> dict:
    exp = {"task_complete": False}
    return EffectJudge(world=None, **kw)._compare(ref, cand, PRE, exp, exp)


def test_the_section_is_parsed_and_in_the_probe():
    assert parse_gitstage(f"{MARK} GITSTAGE\n{IDX_MERGE}  {REPO}\nnot a line\n{MARK} LISTING\n") == {REPO: IDX_MERGE}
    s = build_probe_script("true\n", ["/app"], 30)
    assert f"{MARK} GITSTAGE" in s and "ls-files -s" in s and "safe.directory" in s


def test_the_same_files_without_the_merged_index_are_not_credited():
    ref, cand = effects(IDX_MERGE), effects(IDX_PIVOT)
    assert ref.gitstage == {REPO: IDX_MERGE}
    r = x(ref, cand)
    assert r["reward"] < 0.75 and r["penalties"].get("git_index_mismatch") == 0.0


def test_the_gate_off_is_the_old_x():
    ref, cand = effects(IDX_MERGE), effects(IDX_PIVOT)
    assert x(ref, cand, git_gate=False)["reward"] == 1.0


def test_the_same_index_is_credited_and_a_repo_missing_from_the_expert_is_not_a_gate():
    assert x(effects(IDX_MERGE), effects(IDX_MERGE))["reward"] == 1.0
    assert "git_index_mismatch" not in x(effects(None), effects(IDX_MERGE))["penalties"]


overlay = pytest.importorskip("cleave.world.overlay")


def act(*cmds):
    return json.dumps({"analysis": "a", "plan": "p", "task_complete": False,
                       "commands": [{"keystrokes": c, "duration": 0.1} for c in cmds]})


@pytest.mark.skipif(not overlay.available(), reason="needs user namespaces + overlayfs")
def test_overlay_hand_written_merge_is_not_credited_but_a_real_merge_and_git_status_are():
    import shutil
    if not shutil.which("git"):
        pytest.skip("needs git")
    from pivots.effect import Pivot
    w = overlay.OverlayWorld(store=f"/tmp/cleave-h55-test-{os.getpid()}", workdir="/app")
    g = "git -c user.name=t -c user.email=t@t"
    assert w.run(["bash", "-c", f"mkdir -p {REPO} && cd {REPO} && git init -q -b master && echo a > about.md"
                  f" && git add . && {g} commit -qm one && git checkout -qb side && echo side > about.md"
                  f" && {g} commit -qam side && git checkout -q master && echo main > about.md"
                  f" && {g} commit -qam main"]).ok
    anchor = w.checkpoint()
    merge = f"cd {REPO} && {g} merge -m m side\n"
    pv = Pivot("h55:1", anchor, REPO, act(merge))
    j = EffectJudge(w, check_determinism=False)
    assert j.score(pv, pv.expected_answer).reward == 1.0
    # the same conflict state written by hand: about.md with markers, MERGE_HEAD, MERGE_MSG, MERGE_MODE
    probe = w.fork(anchor)
    try:
        assert probe.run(["bash", "-c", merge]).exit_code != 0  # a conflict
        files = {p: probe.run(["bash", "-c", f"base64 -w0 {REPO}/{p}"]).stdout.strip()
                 for p in ("about.md", ".git/MERGE_HEAD", ".git/MERGE_MSG", ".git/MERGE_MODE")}
    finally:
        probe.close()
    copy = [f"printf %s {b} | base64 -d > {REPO}/{p}\n" for p, b in files.items()]
    s = j.score(pv, act(*copy))
    assert s.binary == 0.0 and "git_index_mismatch" in s.penalties
    # an expert step that only refreshes the index (`git status`) does not gate a candidate that skips it
    pv2 = Pivot("h55:2", anchor, REPO, act(f"cd {REPO} && git status --short && cat about.md\n"))
    assert j.score(pv2, act(f"cat {REPO}/about.md\n")).binary == 1.0
