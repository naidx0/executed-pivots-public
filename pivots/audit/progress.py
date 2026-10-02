"""Verified progress at horizon 0: a per-step truth that does not depend on a continuation model.

Task success after a continuation mostly measures the continuation (see closed_loop.py:
a strong one recovers from anything, a weak one fails alike from any start). This asks
the task's own verifier a finer question right after the step: how many of its checks
pass now? A step is labelled

  P = 1   it leaves at least as many verifier checks passing as the expert's step does,
          and, if it claims task_complete, the task really is complete
  P = 0   otherwise
  harm    it leaves fewer checks passing than before the step (a regression)

Read-only expert steps change no checks, so any harmless read scores P = 1 there, which
is the intended reading: at that turn, looking around is as good as the expert's look.

    python -m pivots.audit.progress specimens/* --audit out/audit --out out/audit/progress.jsonl
"""
from __future__ import annotations

import argparse
import functools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cleave.world.overlay import OverlayWorld
from pivots import specimen as S
from pivots.audit.run import compare
from pivots.effect.terminus import parse_action
from pivots.replay.step import step_script


def after_action(world, sp, anchor: str, text: str) -> dict:
    pa = parse_action(text)
    if pa.action is None:
        v = S.verify(world, anchor, sp)
        return {"checks": v["checks_passed"], "reward": v["reward"], "complete": False}
    f = world.fork(anchor)
    prog, _ = step_script(pa.action.commands, host=sp.host, default_cwd=sp.workdir)
    f.run(["bash", "-c", prog], cwd="/", timeout_s=60 * max(1, len(pa.action.commands)) + 60)
    v = S.verify(world, f.checkpoint(), sp)
    return {"checks": v["checks_passed"], "reward": v["reward"], "complete": bool(pa.action.task_complete)}


def run(specimens: list[str], audit: str, store: str, workers: int = 8) -> list[dict]:
    rows = [json.loads(line) for line in open(Path(audit) / "rows.jsonl")]
    out = []
    for p in specimens:
        if not os.path.exists(os.path.join(p, "task.json")):
            continue
        sp = S.load(p)
        mine = [r for r in rows if r["task"] == sp.name]
        if not mine:
            continue
        w = S.build(functools.partial(OverlayWorld, store=store), sp)
        rr = S.run_expert(w, sp)
        before = [S.verify(w, rr.anchor(t), sp)["checks_passed"] for t in range(len(sp.actions))]
        expert = [S.verify(w, rr.steps[t].checkpoint, sp)["checks_passed"] for t in range(len(sp.actions))]

        def label(r):
            t = r["turn"]
            a = after_action(w, sp, rr.anchor(t), r["text"])
            ok = a["checks"] is not None and a["checks"] >= expert[t]
            if a["complete"] and a["reward"] != 1:
                ok = False
            return {"task": r["task"], "turn": t, "action": r["action"], "J": r["J"], "X": r["X"], "E": r["E"],
                    "checks_before": before[t], "checks_expert": expert[t], "checks_after": a["checks"],
                    "P": int(ok), "harm": int(a["checks"] is not None and a["checks"] < before[t])}

        with ThreadPoolExecutor(workers) as ex:
            out += list(ex.map(label, mine))
        print(sp.name, len(mine), flush=True)
    return out


def summarize(labels: list[dict]) -> dict:
    s = {}
    groups = {"all": labels, "ctrl": [r for r in labels if r["action"].startswith("ctrl:")]}
    for pool in sorted({r["action"].split(":")[1] for r in labels if r["action"].startswith("student:")}):
        groups[f"student:{pool}"] = [r for r in labels if r["action"].startswith(f"student:{pool}:")]
    for g, sub in groups.items():
        pr = [{"J": r["J"], "X": r["X"], "E": r["P"]} for r in sub]
        s[g] = {"n": len(sub), "P_rate": round(sum(r["P"] for r in sub) / len(sub), 4),
                "harm": sum(r["harm"] for r in sub), "J": compare(pr, "J"), "X": compare(pr, "X"),
                "E_open_vs_P_agree": round(sum(r["E"] == r["P"] for r in sub) / len(sub), 4)}
    ctrl = {}
    for r in groups["ctrl"]:
        c = ctrl.setdefault(r["action"][5:], {"n": 0, "P1": 0, "harm": 0})
        c["n"] += 1
        c["P1"] += r["P"]
        c["harm"] += r["harm"]
    s["controls"] = ctrl
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("specimens", nargs="+")
    ap.add_argument("--audit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default=f"/tmp/cleave-progress-{os.getpid()}")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    labels = run(a.specimens, a.audit, a.store, a.workers)
    with open(a.out, "w") as fh:
        for r in labels:
            fh.write(json.dumps(r) + "\n")
    summ = summarize(labels)
    Path(a.out).with_suffix(".summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: {kk: v[kk] for kk in ("n", "P_rate", "harm")} for k, v in summ.items() if k != "controls"}))


if __name__ == "__main__":
    main()
