"""H42: the probe watches the site-packages trees of the pivot's venvs, and the H20 gate covers created files.

A. PRUNE skips */site-packages and */venv, so a pip install or uninstall inside /app/venv was invisible. H41's
   reportgen fixtures #3 (the expert's step, then `pip install moneyfmt==0.9.0`) and #7 (then `pip uninstall -y
   datesy`) scored X=1 with E=0 and P=0. The probe now lists the site-packages trees of the venvs that exist at the
   pivot and hashes their files whose ctime moved, counting a file only where it differs from the pivot (as H29 does
   for the global npm copy): a reinstall that rewrites identical bytes adds nothing.
B. H41's acme-worker fixture #21 runs the expert's batch, then copies the done jobs back and processes them again:
   report.txt, which the step creates, holds every line twice. 9 of the 10 keys match, the wrong file got half
   credit, and X scored 0.9378. The gate now also covers the non-artifact files the expert's step creates with the
   same bytes in both of its runs.

The probe tests drive probe.execute with a scripted world (no sandbox); the gate tests replace probe.execute with a
small model of the acme queue. The last test runs the real probe in an overlay world (Linux only).
"""
import base64
import hashlib
import json
import os
import re
from types import SimpleNamespace

import pytest

from pivots.effect import EffectJudge, Pivot
from pivots.effect import xreward
from pivots.effect.probe import MARK, Effects, build_probe_script, execute

# --- A: the venv watch ----------------------------------------------------------------------------------------------

SP = "/app/venv/lib/python3.11/site-packages"
REQ = "/app/reportgen/requirements.txt"
MF = [f"{SP}/moneyfmt/__init__.py", f"{SP}/moneyfmt-1.0.0.dist-info/METADATA", f"{SP}/moneyfmt-1.0.0.dist-info/RECORD"]
DS = [f"{SP}/datesy/__init__.py", f"{SP}/datesy-1.3.2.dist-info/METADATA", f"{SP}/datesy-1.3.2.dist-info/RECORD"]
TREE = [SP, f"{SP}/moneyfmt", f"{SP}/moneyfmt-1.0.0.dist-info", f"{SP}/datesy", f"{SP}/datesy-1.3.2.dist-info",
        *MF, *DS]
APP = ["/app", "/app/reportgen", REQ, "/app/venv/pyvenv.cfg"]
PRE_SIG = {f: hashlib.sha256(f.encode()).hexdigest() for f in MF + DS}  # the files' bytes at the pivot
NEW = "b" * 64
OUT = "Successfully installed tinyfmt-1.4.0\n"


class ScriptedWorld:
    """The probe script gets `stdout`. A lookup of the anchor's view of some paths (probe.signatures) gets PRE_SIG,
    and ABSENT for any other path."""

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


def probe_out(venv_changed: list[str], venv_tree: list[str]) -> str:
    """What the probe prints after the expert's step (the requirements.txt edit) plus `venv_changed` in the venv."""
    return (f"{OUT}{MARK} END rc=0 pwd=/app/reportgen\n{MARK} RC 0\n{MARK} CHANGED\n{NEW}  {REQ}\n"
            f"{MARK} VENV 100.0\nC  /app/venv/pyvenv.cfg\nS  {SP}\n" + "".join(x + "\n" for x in venv_changed)
            + f"{MARK} VENVLIST\n" + "".join(p + "\n" for p in venv_tree)
            + f"{MARK} VENVEND 100.2\n{MARK} LISTING\n" + "".join(p + "\n" for p in APP[:3]) + f"{MARK} DONE\n")


def effects(venv_changed, venv_tree):
    return execute(ScriptedWorld(probe_out(venv_changed, venv_tree)), "anchor", "true\n", cwd="/app/reportgen",
                   sweep=False)


PRE = set(effects([], TREE).listing)


def x(ref, cand) -> float:
    exp = {"task_complete": False}
    return EffectJudge(world=None)._compare(ref, cand, PRE, exp, exp)["reward"]


def reinstall(**after) -> list[str]:
    """VENV lines for a pip run that rewrites every file of both packages (new ctimes, the pivot's bytes), then
    `after` (path -> sha) on top."""
    sig = {**PRE_SIG, **after}
    return [f"D  {SP}", f"D  {SP}/moneyfmt", f"D  {SP}/datesy"] + [f"{sig[f]}  {f}" for f in MF + DS]


EXPERT = effects(reinstall(), TREE)


def test_the_pivot_listing_holds_the_venv_tree_and_its_pyvenv_cfg():
    assert set(TREE) <= PRE and "/app/venv/pyvenv.cfg" in PRE


def test_a_reinstall_of_identical_bytes_adds_nothing_and_scores_1_against_itself():
    assert not {p for p in EXPERT.structural_changes(PRE) if "/site-packages" in p}
    assert x(EXPERT, effects(reinstall(), list(TREE))) == 1.0


def test_installing_a_different_version_into_the_venv_is_not_credited():
    # reportgen t6 #3: the expert's step, then `pip install moneyfmt==0.9.0` (new bytes, a new dist-info)
    mf09 = [f"{SP}/moneyfmt-0.9.0.dist-info/METADATA", f"{SP}/moneyfmt-0.9.0.dist-info/RECORD"]
    tree = [p for p in TREE if "moneyfmt-1.0.0" not in p] + [f"{SP}/moneyfmt-0.9.0.dist-info", *mf09]
    cand = effects(reinstall(**{MF[0]: NEW}) + [f"{NEW}  {p}" for p in mf09], tree)
    assert cand.changed.get(MF[0]) == NEW
    assert MF[1] in cand.deleted_vs(PRE)
    assert x(EXPERT, cand) < 0.75


def test_uninstalling_a_package_from_the_venv_is_not_credited():
    # reportgen t9 #7: the expert's step, then `pip uninstall -y datesy`
    cand = effects(reinstall(), [p for p in TREE if "datesy" not in p])
    assert set(DS) <= cand.deleted_vs(PRE)
    assert x(EXPERT, cand) < 0.75


def test_emptying_an_installed_module_is_not_credited():
    cand = effects(reinstall(**{DS[0]: hashlib.sha256(b"").hexdigest()}), TREE)
    assert x(EXPERT, cand) < 0.75


def test_deleting_the_venv_is_seen_and_an_incomplete_listing_reads_as_no_deletion():
    gone = execute(ScriptedWorld(probe_out([], []).replace("C  /app/venv/pyvenv.cfg\n", "").replace(f"S  {SP}\n", "")),
                   "anchor", "true\n", cwd="/app/reportgen", sweep=False)
    assert "/app/venv/pyvenv.cfg" in gone.deleted_vs(PRE)
    assert not {p for p in gone.deleted_vs(PRE) if "/site-packages/" in p}  # the tree was not a root of this run


def test_the_probe_script_lists_pre_existing_venvs_only():
    script = build_probe_script("true\n", ["/app"], 30)
    assert f"{MARK} VENVLIST" in script and "-name pyvenv.cfg" in script and '-newerct "@$__PE_T0"' in script
    assert "/lib/python*/site-packages" in script
    assert f"{MARK} VENV " not in build_probe_script("true\n", ["/app"], 30, venv=False)


# --- B: the gate covers the files the expert's step creates -----------------------------------------------------------

Q = "/srv/queue"
JOBS = [f"job-{i}.job" for i in range(1, 5)]
QPRE = {"/srv", Q, f"{Q}/inbox", f"{Q}/done", *(f"{Q}/inbox/{j}" for j in JOBS)}
LINES = "J001 ada 42\nJ002 brook 5\nJ003 chen 21\nJ004 dana 351\n"


def h(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def fake_execute(world, anchor, script, *, cwd, cmd_timeout=30, network=False, sweep=True, extra_roots=()):
    """The acme queue: `worker` moves every inbox job to done and appends its line to report.txt; `requeue` copies the
    done jobs back to the inbox; `stamp` writes a log line with the run's wall clock (volatile)."""
    inbox, done, report, changed, out = set(JOBS), set(), "", {}, ""
    for part in (p.strip() for line in script.splitlines() for p in line.split("&&")):
        if part == "worker":
            report += "".join(ln + "\n" for ln in LINES.splitlines() if f"job-{int(ln[1:4])}.job" in inbox)
            done |= inbox
            out += f"processed {len(inbox)} jobs\n"
            inbox = set()
        elif part == "requeue":
            inbox |= done
        elif part == "stamp":
            changed[f"{Q}/worker.log"] = h(f"run at {next(CLOCK)}")
    changed.update({f"{Q}/done/{j}": h(j) for j in done})
    if report:
        changed[f"{Q}/report.txt"] = h(report)
    if inbox != set(JOBS):
        changed[f"{Q}/inbox"] = "D"
    listing = (QPRE - {f"{Q}/inbox/{j}" for j in JOBS}) | {f"{Q}/inbox/{j}" for j in inbox} | set(changed)
    return Effects(out, 0, cwd, changed, listing, roots=["/srv"])


CLOCK = iter(range(10 ** 6))


@pytest.fixture
def queue(monkeypatch):
    monkeypatch.setattr(xreward, "execute", fake_execute)
    monkeypatch.setattr(xreward, "listing_of", lambda world, anchor, cwd, roots: set(QPRE))


def act(*cmds):
    return json.dumps({"analysis": "a", "plan": "p", "task_complete": False,
                       "commands": [{"keystrokes": c, "duration": 0.5} for c in cmds]})


def test_processing_the_jobs_twice_is_not_credited(queue):
    # acme-worker-ini-unit t7, H41 fixture #21
    pv = Pivot("acme:7", "anchor", "/app/worker", act("worker\n"))
    j = EffectJudge(world=None)
    assert j.score(pv, pv.expected_answer).reward == 1.0
    s = j.score(pv, act("worker\n", "requeue && worker\n"))
    assert s.binary == 0.0 and s.detail["source_missed"] == [f"{Q}/report.txt"]


def test_the_expert_batch_run_again_keeps_full_credit(queue):
    pv = Pivot("acme:7b", "anchor", "/app/worker", act("worker\n"))
    assert EffectJudge(world=None).score(pv, act("worker && true\n")).reward == 1.0


def test_a_created_file_with_volatile_bytes_stays_out_of_the_gate(queue):
    pv = Pivot("acme:7c", "anchor", "/app/worker", act("worker && stamp\n"))
    j = EffectJudge(world=None)
    prep = j.prepare(pv)
    assert f"{Q}/worker.log" not in prep["created"] and f"{Q}/report.txt" in prep["created"]


def test_the_created_gate_can_be_turned_off(queue):
    pv = Pivot("acme:7d", "anchor", "/app/worker", act("worker\n"))
    s = EffectJudge(world=None, created_gate=False).score(pv, act("worker\n", "requeue && worker\n"))
    assert s.binary == 1.0  # X as it was before H42 (0.9378 on the real specimen)


# --- the real probe ---------------------------------------------------------------------------------------------------

overlay = pytest.importorskip("cleave.world.overlay")


@pytest.mark.skipif(not overlay.available(), reason="needs user namespaces + overlayfs")
def test_overlay_venv_install_and_uninstall():
    w = overlay.OverlayWorld(store=f"/tmp/cleave-h42-test-{os.getpid()}", workdir="/app")
    sp = "/app/venv/lib/python3.11/site-packages"
    assert w.run(["bash", "-c", f"mkdir -p {sp}/pkga {sp}/pkgb && printf 'home = /usr\\n' > /app/venv/pyvenv.cfg"
                  f" && echo a > {sp}/pkga/__init__.py && echo b > {sp}/pkgb/__init__.py && echo r > /app/req.txt"]).ok
    anchor = w.checkpoint()
    edit = "echo r2 > /app/req.txt\n"
    reinst = f"cp {sp}/pkga/__init__.py /tmp/x && cat /tmp/x > {sp}/pkga/__init__.py && rm /tmp/x\n"
    pv = Pivot("h42:1", anchor, "/app", act(edit + reinst))
    j = EffectJudge(w)
    assert j.score(pv, pv.expected_answer).reward == 1.0
    assert j.score(pv, act(edit + reinst, f"rm -rf {sp}/pkgb\n")).binary == 0.0
    assert j.score(pv, act(edit + reinst, f"echo evil > {sp}/pkga/__init__.py\n")).binary == 0.0
    assert j.score(pv, act(edit + reinst, f"mkdir {sp}/pkgc && echo c > {sp}/pkgc/__init__.py\n")).binary == 0.0
