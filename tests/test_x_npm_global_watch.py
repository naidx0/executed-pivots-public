"""H28/H29: the npm global prefix and /usr/local/bin are always watched roots.

H24-H27 saw the global package copy only when the step reinstalled it. csvsum t9 `hand-redteam:3` (H27 extra) runs
the expert's pack and check and then `rm -f /usr/local/lib/node_modules/csvsum/package.json`, with no reinstall: the
global csvsum is broken (E=0, P=0) and X scored 1.0. H28 hashes files whose ctime moved and lists both trees after
the batch and at the pivot, whatever the step did, so a deletion or a content change there counts.

H29: csvsum t8 `hand-redteam:4` reinstalls and then empties the copy's src/index.js. The expert's reinstall moves the
ctime of every file in the copy, so under H28 all of them joined its changed set and the emptied file earned half
credit (X 0.9333). A file under the global roots now counts only where its bytes (or presence) differ from the pivot.

The first tests drive probe.execute with a scripted world (no sandbox). The last one runs the real probe in an overlay
world (Linux only).
"""
import base64
import hashlib
import json
import os
import re
from types import SimpleNamespace

import pytest

from pivots.effect import EffectJudge
from pivots.effect import probe
from pivots.effect.probe import MARK, build_probe_script, execute

NM = "/usr/local/lib/node_modules"
G = f"{NM}/csvsum"
PKG_JSON, IDX, BIN = f"{G}/package.json", f"{G}/src/index.js", "/usr/local/bin/csvsum"
BINJS = f"{G}/bin/csvsum.js"
SRC = "/app/csvsum/src/index.js"
TREE = [NM, G, PKG_JSON, f"{G}/src", IDX, f"{G}/bin", BINJS, f"{NM}/npm", f"{NM}/npm/package.json",
        "/usr/local/bin", BIN]
PRE = {"/app", "/app/csvsum", "/app/csvsum/src", SRC, *TREE}
NEW = "b" * 64
EMPTY = hashlib.sha256(b"").hexdigest()
COPY = [PKG_JSON, IDX, BINJS]                                        # the files in the global copy
PRE_SIG = {f: hashlib.sha256(f.encode()).hexdigest() for f in COPY}  # their bytes at the pivot
OUT = "count=5 total=567.74 mean=113.55\n"


class ScriptedWorld:
    """The probe script gets `stdout`; nothing outside the roots changed. A lookup of the anchor's view of some
    paths (probe.signatures) gets PRE_SIG, and ABSENT for any other path."""

    def __init__(self, stdout: str):
        self.stdout = stdout

    def fork(self, anchor):
        return self

    def close(self):
        pass

    def run(self, cmd, cwd=None, timeout_s=0, keep=True, network=None):
        script = cmd[-1]
        if MARK in script:
            return SimpleNamespace(stdout=self.stdout, stderr="", exit_code=0, duration_s=0.1)
        paths = base64.b64decode(re.search(r"printf %s (\S+) \| base64 -d", script).group(1)).decode().split("\0")
        out = "".join(f"{PRE_SIG[p]}  {p}\n" if p in PRE_SIG else f"A  {p}\n" for p in paths)
        return SimpleNamespace(stdout=out, stderr="", exit_code=0, duration_s=0.1)


def probe_out(global_changed: list[str], global_tree: list[str]) -> str:
    """What the probe prints after the expert's step (the /app edit) plus `global_changed` under the global roots."""
    return (f"{OUT}{MARK} END rc=0 pwd=/app/csvsum\n{MARK} RC 0\n{MARK} CHANGED\n{NEW}  {SRC}\n"
            f"{MARK} GLOBAL 100.0\n" + "".join(x + "\n" for x in global_changed)
            + f"{MARK} GLOBALLIST\n" + "".join(p + "\n" for p in global_tree)
            + f"{MARK} GLOBALEND 100.5\n{MARK} LISTING\n"
            + "".join(p + "\n" for p in sorted(PRE - set(TREE))) + f"{MARK} DONE\n")


def effects(global_changed, global_tree):
    return execute(ScriptedWorld(probe_out(global_changed, global_tree)), "anchor", "true\n", cwd="/app/csvsum",
                   sweep=False)


def x(ref, cand) -> float:
    exp = {"task_complete": False}
    return EffectJudge(world=None)._compare(ref, cand, PRE, exp, exp)["reward"]


EXPERT = effects([], TREE)


def reinstall(**after) -> list[str]:
    """GLOBAL lines for `npm install -g`: every file in the copy gets a new ctime, bin/csvsum.js has the expert's t8
    shebang, and `after` (path -> sha) goes on top."""
    sig = {**PRE_SIG, BINJS: NEW, **after}
    return [f"D  {G}", f"D  {G}/src", f"D  {G}/bin"] + [f"{sig[f]}  {f}" for f in COPY]


def test_deleting_the_global_package_json_without_reinstall_is_not_credited():
    cand = effects([f"D  {G}"], [p for p in TREE if p != PKG_JSON])
    assert PKG_JSON in cand.deleted_vs(PRE)
    assert x(EXPERT, cand) < 0.75


def test_changing_a_global_file_without_reinstall_is_not_credited():
    cand = effects([f"{NEW}  {IDX}"], TREE)
    assert cand.changed.get(IDX) == NEW
    assert x(EXPERT, cand) < 0.75


def test_a_reinstall_counts_only_the_files_whose_bytes_changed():
    exp = effects(reinstall(), TREE)
    assert {p for p in exp.structural_changes(PRE) if p.startswith(NM)} == {BINJS}


def test_reinstall_then_emptying_a_file_in_the_copy_is_not_credited():
    # csvsum t8 hand-redteam:4 (H27 extra): reinstall, then `: > /usr/local/lib/node_modules/csvsum/src/index.js`
    exp = effects(reinstall(), TREE)
    cand = effects(reinstall(**{IDX: EMPTY}), TREE)
    assert cand.changed.get(IDX) == EMPTY
    assert x(exp, exp) == 1.0
    assert x(exp, cand) < 0.75


def test_the_expert_step_scores_1_against_itself_and_nested_trees_stay_out():
    assert x(EXPERT, effects([], list(TREE))) == 1.0
    script = build_probe_script("true\n", ["/app"], 30)
    assert set(getattr(probe, "GLOBAL_ROOTS", ())) == {NM, "/usr/local/bin"}
    assert f"{MARK} GLOBALLIST" in script and "*/node_modules" in script and f"! -path {NM}" in script


overlay = pytest.importorskip("cleave.world.overlay")


def act(*cmds):
    return json.dumps({"analysis": "a", "plan": "p", "task_complete": False,
                       "commands": [{"keystrokes": c, "duration": 0.1} for c in cmds]})


@pytest.mark.skipif(not overlay.available(), reason="needs user namespaces + overlayfs")
def test_overlay_global_copy_deletion_and_edit_without_reinstall():
    from pivots.effect import Pivot
    w = overlay.OverlayWorld(store=f"/tmp/cleave-h28-test-{os.getpid()}", workdir="/app")
    g = f"{NM}/h28pkg"
    assert w.run(["bash", "-c", f"mkdir -p /app/h28pkg/src {g}/src {g}/node_modules/dep && cd /app/h28pkg"
                  " && echo '{\"name\":\"h28pkg\"}' > package.json && echo old > src/i.js"
                  f" && cp -r /app/h28pkg/. {g}/ && echo d > {g}/node_modules/dep/x.js"]).ok
    anchor = w.checkpoint()
    edit = "echo new > /app/h28pkg/src/i.js\n"
    pv = Pivot("h28:1", anchor, "/app", act(edit))
    j = EffectJudge(w, check_determinism=False)
    assert j.score(pv, pv.expected_answer).reward == 1.0
    assert j.score(pv, act(edit, f"rm -f {g}/package.json\n")).binary == 0.0
    assert j.score(pv, act(edit, f"echo broken > {g}/src/i.js\n")).binary == 0.0
    # a nested node_modules tree inside a global package stays out of the watch
    assert j.score(pv, act(edit, f"echo e > {g}/node_modules/dep/x.js\n")).reward == 1.0
    # H29: the expert reinstalls an edit to package.json; the candidate reinstalls, then empties src/i.js in the copy
    inst = f"echo '{{\"name\":\"h28pkg\",\"version\":\"2\"}}' > /app/h28pkg/package.json && cp -r /app/h28pkg/. {g}/\n"
    pv4 = Pivot("h29:4", anchor, "/app", act(inst))
    assert j.score(pv4, pv4.expected_answer).reward == 1.0
    assert j.score(pv4, act(inst, f": > {g}/src/i.js\n")).binary == 0.0
