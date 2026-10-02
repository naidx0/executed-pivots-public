"""Deterministic E0 rebuild, measured against known ground truth (specimens).

Stage A recovers E0 (initial file contents) from what the expert read. This
module asks how far E0 alone gets before any model fills gaps (Stage B):

  1. turn a specimen's expert replay into a released-row-shaped record
     (same message layout as Terminal-Pivot rows),
  2. recover E0 with Stage A's rules (v1) or with v2 additions:
       - relative reads resolved against the cwd printed in the prompt
         (`root@modal:/app/pkg# cat calc.py` -> /app/pkg/calc.py)
       - paths seen in `find`/`ls` output recreated as empty placeholders
         so directory structure and file presence survive
  3. build a world from E0 only, replay the expert, and score it with G1
     and with the task's own verifier on the expert's final state.
"""
from __future__ import annotations

import json
import posixpath
import re
from dataclasses import dataclass

from pivots.materialize import (DERIVED, GLOB, READ_FULL, TRUNC, build_trajectory, is_read_only,
                                recover_files)
from pivots.specimen_prompt import TERMINUS2_PREAMBLE

PROMPT_CWD = re.compile(r"root@[^\s#]+:(?P<cwd>[^\n#]*)# (?P<cmd>[^\n]*)\n?")
READ_REL = re.compile(r"^\s*cat\s+(?:-A\s+|-v\s+)?(?P<path>[^\s;|&><-][^\s;|&><]*)\s*$")
LS_LONG = re.compile(r"^[-dl][-rwxsStT]{9}[.+@]?\s+\d+\s+\S+\s+\S+\s+\d+\s+\w{3}\s+\d+\s+[\d:]+\s+(?P<name>.+?)(?: -> .*)?$")


def specimen_row(sp, rendered: list[str], initial_screen: str) -> dict:
    """A record shaped like the last released row of a trajectory (whole history + gold turn)."""
    first = (f"{TERMINUS2_PREAMBLE}\n\nTask Description:\n{sp.instruction}\n\n"
             f"Current terminal state:\nCurrent Terminal Screen:\n{initial_screen}\n")
    msgs = [{"role": "user", "content": first}]
    for a, obs in zip(sp.actions[:-1], rendered[:-1]):
        msgs += [{"role": "assistant", "content": a}, {"role": "user", "content": "New Terminal Output:\n" + obs}]
    return {"uuid": f"specimen-{sp.name}", "task_name": sp.name, "expected_answer": sp.actions[-1],
            "responses_create_params": {"input": msgs},
            "metadata": {"source_trajectory_uid": sp.name, "teacher_model": "specimen", "harness": "terminus_2",
                         "pivot_agent_turn_index": len(sp.actions) - 1, "total_source_agent_turns": len(sp.actions)}}


def segments_with_cwd(obs: str):
    """(cwd, cmd, out) for each prompt in an observation."""
    ms = list(PROMPT_CWD.finditer(obs))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(obs)
        yield m.group("cwd").strip(), m.group("cmd").strip(), obs[m.end():end]


def recover_v2(row: dict) -> tuple[dict[str, tuple[str, bool]], set[str], dict]:
    traj = build_trajectory(row)
    e0, written, _, flags = recover_files(traj)
    e0 = dict(e0)
    placeholders: set[str] = set()
    extra = {"relative_reads": 0, "placeholders": 0}
    mutated = False
    for t in traj.turns:
        for cwd, cmd, out in segments_with_cwd(t.observation):
            if not cmd:
                continue
            if not is_read_only(cmd):
                mutated = True
            m = READ_REL.match(cmd)
            if m and not READ_FULL.match(cmd) and not GLOB.search(m.group("path")) and cwd.startswith("/"):
                p = posixpath.normpath(posixpath.join(cwd, m.group("path")))
                if not mutated and p not in written and p not in e0 and not DERIVED.match(p):
                    e0[p] = (out, bool(TRUNC.search(out)))
                    extra["relative_reads"] += 1
            if cmd.startswith("find "):
                for line in out.splitlines():
                    line = line.strip()
                    if line.startswith("/") and not GLOB.search(line) and not DERIVED.match(line):
                        placeholders.add(line)
            elif cmd.startswith("ls") and cwd.startswith("/"):
                target = cmd.split()[-1] if len(cmd.split()) > 1 and not cmd.split()[-1].startswith("-") else cwd
                base = target if target.startswith("/") else posixpath.join(cwd, target)
                for line in out.splitlines():
                    lm = LS_LONG.match(line)
                    if lm and lm.group("name") not in (".", "..") and line[0] == "-":
                        placeholders.add(posixpath.normpath(posixpath.join(base, lm.group("name"))))
    placeholders -= set(e0)
    placeholders = {p for p in placeholders if p not in written}
    extra["placeholders"] = len(placeholders)
    return e0, placeholders, {**flags, **extra}


@dataclass
class RebuildScore:
    e0_files: int
    placeholders: int
    true_files: int
    exact_files: int
    g1_share: float
    g1_admitted: bool
    gold_passes_in_rebuild: bool


def build_from_e0(world, e0: dict[str, tuple[str, bool]], placeholders: set[str]):
    """Write E0 contents (and empty placeholders) into a fresh world; return its checkpoint."""
    payload = {"files": {p: c for p, (c, _) in e0.items()}, "empty": sorted(placeholders)}
    code = ("import json,os,sys\nd=json.load(sys.stdin)\n"
            "for p in d['empty']:\n os.makedirs(os.path.dirname(p),exist_ok=True)\n open(p,'a').close()\n"
            "for p,c in d['files'].items():\n os.makedirs(os.path.dirname(p),exist_ok=True)\n open(p,'w').write(c)\n")
    import base64
    b64 = base64.b64encode(json.dumps(payload).encode()).decode()
    code64 = base64.b64encode(code.encode()).decode()
    op = world.run(["bash", "-c", f'printf %s {b64} | base64 -d | python3 -c "$(printf %s {code64} | base64 -d)"'],
                   cwd="/")
    if not op.ok:
        raise RuntimeError(op.stderr[-800:])
    return world.checkpoint()


def true_files(world, checkpoint: str, roots=("/app", "/data", "/srv", "/opt/app", "/workspace", "/etc/app")) -> dict[str, str]:
    """path -> content of every regular file under the task roots of a world (ground truth)."""
    f = world.fork(checkpoint)
    code = ("import os,json\nout={}\nfor r in %r:\n for d,_,fs in os.walk(r):\n  if '/.git' in d: continue\n"
            "  for n in fs:\n   p=os.path.join(d,n)\n   try: out[p]=open(p,errors='replace').read()\n   except Exception: pass\n"
            "print(json.dumps(out))" % (list(roots),))
    op = f.run(["python3", "-c", code], cwd="/", keep=False)
    return json.loads(op.stdout or "{}")


def evaluate(sp, world_factory, versions=("v1", "v2")) -> dict:
    from pivots import specimen as S
    from pivots.replay.fidelity import gate_g1
    w = S.build(world_factory, sp)
    base = w.checkpoint()
    truth = true_files(w, base)
    rr = S.run_expert(w, sp)
    rendered = [s.rendered for s in rr.steps]
    row = specimen_row(sp, rendered, f"root@{sp.host}:{sp.workdir}# ")
    out = {"task": sp.name, "true_files": len(truth), "gold_passes_true": S.verify(w, rr.steps[-1].checkpoint, sp)["reward"] == 1}
    for v in versions:
        if v == "v1":
            e0, _, _, flags = recover_files(build_trajectory(row))
            ph: set[str] = set()
        else:
            e0, ph, flags = recover_v2(row)
        rw = world_factory(workdir=sp.workdir)
        build_from_e0(rw, e0, ph)
        rr2 = S.run_expert(rw, sp)
        g1 = gate_g1(rendered, [s.rendered for s in rr2.steps])
        exact = sum(1 for p, (c, _) in e0.items() if truth.get(p) == c)
        out[v] = {"e0_files": len(e0), "placeholders": len(ph), "exact_files": exact,
                  "recall_exact": round(exact / max(1, len(truth)), 3),
                  "g1_share": round(g1.share_passing, 3), "g1_admitted": g1.admitted,
                  "gold_passes_in_rebuild": S.verify(rw, rr2.steps[-1].checkpoint, sp)["reward"] == 1}
    return out
