"""Mini audit on specimens: label every action at every pivot with J, X and E, then compare.

  J        shipped string reward (exact port)
  X        effect equivalence with the expert's batch, horizon 0 (one forked run)
  E        executed outcome: fork the anchor, run the action, splice the expert's
           remaining turns (open loop), run the task's verifier. An action that
           sets task_complete ends the episode, so the verifier runs right after it.

Open-loop splicing stands in for GLM-5.1 closed-loop continuation (plan §7.5
uses it only as a comparator): it can over-credit an action whose damage a
later expert turn happens to overwrite, and under-credit one that makes a later
expert command fail. That is reported, not hidden: every row carries both.

    python -m pivots.audit.run specimens/* --students students.jsonl --out out/

Students can also be sampled live from a served model, with the same labels (--progress adds P):

    python -m pivots.audit.run specimens/* --student tokenfactory:nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B \
        --n-students 5 --pivots billing-invoice-bugfix:3,ledger-git-revert:3 --progress --out out/tf
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cleave.world.overlay import OverlayWorld
from pivots import specimen as S
from pivots.audit.controls import controls
from pivots.effect import EffectJudge, Pivot, string_reward
from pivots.effect.terminus import parse_action
from pivots.replay.step import render, step_script


def executed_outcome(world, sp, rr, turn: int, action_text: str) -> dict:
    pa = parse_action(action_text)
    if pa.action is None:
        return {"E": 0, "why": "unparseable"}
    f = world.fork(rr.anchor(turn))
    prog, _ = step_script(pa.action.commands, host=sp.host, default_cwd=sp.workdir)
    op = f.run(["bash", "-c", prog], cwd="/", timeout_s=180)
    ended = bool(pa.action.task_complete)
    if not ended:
        for b in sp.batches()[turn + 1:]:
            p2, _ = step_script(b, host=sp.host, default_cwd=sp.workdir)
            f.run(["bash", "-c", p2], cwd="/", timeout_s=180)
    v = S.verify(world, f.checkpoint(), sp)
    return {"E": int(v["reward"] == 1), "ended_early": ended and turn < len(sp.actions) - 1,
            "action_exit": op.exit_code, "action_output": render(op.stdout)[-600:]}


def audit_specimen(sp_path: str, students: dict, store: str, workers: int = 4, turns=None,
                   sampler=None) -> list[dict]:
    """Label every action at each pivot of one specimen (only `turns`, when given).

    sampler(sp, [(turn, messages)]) -> {turn: [(pool, text)]} adds live student samples at the labelled
    pivots; `messages` is the Terminus-2 prompt at that pivot (pivots.audit.prompts writes the same one).
    """
    sp = S.load(sp_path)
    wf = functools.partial(OverlayWorld, store=store)
    w = S.build(wf, sp)
    rr = S.run_expert(w, sp)
    gold_ok = S.verify(w, rr.steps[-1].checkpoint, sp)["reward"] == 1
    judge = EffectJudge(w, check_determinism=True)
    todo = [t for t in range(len(sp.actions)) if turns is None or t in turns]
    if sampler is not None:
        obs = [s.rendered for s in rr.steps]
        screen = f"root@{sp.host}:{sp.workdir}# "
        students = dict(students)
        for t, got in sampler(sp, [(t, S.prompt_for(sp, t, obs, screen)) for t in todo]).items():
            students[(sp.name, t)] = [*students.get((sp.name, t), []), *got]
    jobs = []
    for t, gold in enumerate(sp.actions):
        if t not in todo:
            continue
        acts = {f"ctrl:{k}": v for k, v in controls(gold, workdir=sp.workdir).items()}
        for i, (pool, text) in enumerate(students.get((sp.name, t), [])):
            acts[f"student:{pool}:{i}"] = text
        cwd = sp.workdir if t == 0 else _cwd_at(w, rr.anchor(t), sp.workdir)
        pv = Pivot(f"{sp.name}:{t}", rr.anchor(t), cwd, gold,
                   next_answer=sp.actions[t + 1] if t + 1 < len(sp.actions) else None,
                   final=t + 1 == len(sp.actions))
        for name, text in acts.items():
            jobs.append((t, name, text, pv, gold))

    def label(job):
        t, name, text, pv, gold = job
        j = string_reward(text, gold)
        x = judge.score(pv, text)
        e = executed_outcome(w, sp, rr, t, text)
        return {"task": sp.name, "turn": t, "turns": len(sp.actions), "action": name,
                "J": int(j["reward"]), "J_sim": j["similarity"], "X": int(x.binary), "X_score": round(x.reward, 4),
                "X_fail": x.failure, "X_pen": x.penalties,
                "X_detail": {k: x.detail.get(k) for k in ("missing", "extra", "destructive_extra", "cand_output_head",
                                                          "ref_changed", "cand_changed", "cand_exit")}, "E": e["E"], "E_detail": {k: v for k, v in e.items() if k != "E"},
                "self_agreement": judge.prepare(pv)["self_agreement"], "text": text}

    for t in todo:  # warm the per-pivot cache serially (teacher runs + listings)
        judge.prepare(next(j[3] for j in jobs if j[0] == t))
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(label, jobs))
    for r in rows:
        r["gold_passes"] = gold_ok
    pivots = [{"task": sp.name, "turn": t, "instruction": sp.instruction, "gold": sp.actions[t],
               "before": rr.steps[t - 1].rendered[-4000:] if t else f"root@{sp.host}:{sp.workdir}# ",
               "after_gold": rr.steps[t].rendered[-2000:]} for t in todo]
    return rows, pivots


def _cwd_at(world, anchor: str, default: str) -> str:
    f = world.fork(anchor)
    op = f.run(["bash", "-c", "cat /var/lib/xp/cwd 2>/dev/null"], cwd="/", keep=False)
    f.close()
    return op.stdout.strip() or default


def kappa(pairs: list[tuple[int, int]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if math.isclose(pe, 1.0) else (po - pe) / (1 - pe)


def compare(rows: list[dict], proxy: str) -> dict:
    pairs = [(r[proxy], r["E"]) for r in rows]
    n = len(pairs)
    cells = {"P1E1": 0, "P1E0": 0, "P0E1": 0, "P0E0": 0}
    for p, e in pairs:
        cells[f"P{p}E{e}"] += 1
    p1 = cells["P1E1"] + cells["P1E0"]
    e1 = cells["P1E1"] + cells["P0E1"]
    return {"n": n, "cells": cells, "D": round((cells["P1E0"] + cells["P0E1"]) / n, 4) if n else None,
            "kappa": None if (k := kappa(pairs)) is None else round(k, 4),
            "P(E=0|rewarded)": round(cells["P1E0"] / p1, 4) if p1 else None,
            "P(unrewarded|E=1)": round(cells["P0E1"] / e1, 4) if e1 else None}


def summarize(rows: list[dict]) -> dict:
    out = {"all": {"J": compare(rows, "J"), "X": compare(rows, "X")}}
    for group in ("ctrl", "student"):
        sub = [r for r in rows if r["action"].startswith(group)]
        if sub:
            out[group] = {"J": compare(sub, "J"), "X": compare(sub, "X")}
    pools = sorted({r["action"].split(":")[1] for r in rows if r["action"].startswith("student:")})
    for pool in pools:
        sub = [r for r in rows if r["action"].startswith(f"student:{pool}:")]
        out[f"student:{pool}"] = {"J": compare(sub, "J"), "X": compare(sub, "X"), "E_rate": round(sum(r["E"] for r in sub) / len(sub), 4)}
    out["pools"] = pools
    # decisive pivots: skipping the expert's batch makes the task fail, so E can tell actions apart there
    # (elsewhere the expert's remaining turns redo most of the work and E passes almost anything harmless)
    decisive = {(r["task"], r["turn"]) for r in rows if r["action"] == "ctrl:no_op" and r["E"] == 0}
    out["decisive_pivots"] = len(decisive)
    for pool in pools:
        sub = [r for r in rows if r["action"].startswith(f"student:{pool}:") and (r["task"], r["turn"]) in decisive]
        if sub:
            out[f"decisive:{pool}"] = {"J": compare(sub, "J"), "X": compare(sub, "X"),
                                       "E_rate": round(sum(r["E"] for r in sub) / len(sub), 4)}
    ctrl = {}
    for r in rows:
        if r["action"].startswith("ctrl:"):
            c = ctrl.setdefault(r["action"][5:], {"n": 0, "J1": 0, "X1": 0, "E1": 0})
            c["n"] += 1
            c["J1"] += r["J"]
            c["X1"] += r["X"]
            c["E1"] += r["E"]
    out["controls"] = ctrl
    out["tasks"] = sorted({r["task"] for r in rows})
    out["gold_passes"] = {t: next(r["gold_passes"] for r in rows if r["task"] == t) for t in out["tasks"]}
    return out


CONTROL_ORDER = ("expert", "paraphrase", "flipped", "no_op", "destructive", "early_complete")
DERIVED = ("progress.jsonl", "progress.summary.json")  # pivots.audit.progress output, keyed to these rows


def clear_stale(out: str) -> list[str]:
    """Remove labels derived from an earlier run's rows in `out`, so nothing joins them to the new rows."""
    gone = []
    for name in DERIVED:
        p = Path(out, name)
        if p.exists():
            p.unlink()
            gone.append(name)
    return gone


def report_lines(summ: dict, out: str, stale: list[str] = ()) -> list[str]:
    """A short terminal summary; summary.json has everything."""
    lines = ["", f"{'control':<16}{'n':>5}{'J=1':>6}{'X=1':>6}{'E=1':>6}"]
    ctrl = summ.get("controls", {})
    for k in [*CONTROL_ORDER, *sorted(set(ctrl) - set(CONTROL_ORDER))]:
        if k in ctrl:
            c = ctrl[k]
            lines.append(f"{k:<16}{c['n']:>5}{c['J1']:>6}{c['X1']:>6}{c['E1']:>6}")
    for pool in summ.get("pools", []):
        j, x = summ[f"student:{pool}"]["J"]["cells"], summ[f"student:{pool}"]["X"]["cells"]
        lines.append(f"students {pool}: {j['P1E1'] + j['P0E1']} of {sum(j.values())} work (E=1); "
                     f"credited J {j['P1E1']}, X {x['P1E1']}; failing ones credited J {j['P1E0']}, X {x['P1E0']}")
    gold = summ.get("gold_passes", {})
    tasks = summ.get("tasks", [])
    lines.append(f"decisive pivots {summ.get('decisive_pivots', 0)}; gold passes on "
                 f"{sum(bool(gold.get(t)) for t in tasks)} of {len(tasks)} tasks")
    lines.append(f"wrote {Path(out, 'rows.jsonl').as_posix()}, pivots.json, summary.json ({summ.get('seconds', '?')} s)")
    if stale:
        lines.append(f"removed {', '.join(stale)} from the previous run; rerun pivots.audit.progress for P")
    return lines


def parse_pivots(spec: str | None) -> dict[str, set[int]] | None:
    """'task:turn,task:turn' -> {task: {turns}}; None means every pivot."""
    if not spec:
        return None
    out: dict[str, set[int]] = {}
    for item in filter(None, (x.strip() for x in spec.split(","))):
        task, _, turn = item.rpartition(":")
        if not task or not turn.isdigit():
            raise SystemExit(f"--pivots: {item!r} is not task:turn")
        out.setdefault(task, set()).add(int(turn))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("specimens", nargs="+")
    ap.add_argument("--students", help="JSONL rows {task, turn, text}")
    ap.add_argument("--out", default="out/audit")
    ap.add_argument("--store", default=f"/tmp/cleave-audit-{os.getpid()}")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pivots", help="label only these pivots: task:turn,task:turn (default: all)")
    ap.add_argument("--progress", action="store_true", help="also label P (pivots.audit.progress) into --out")
    live = ap.add_argument_group("live students (sampled from a served model at each labelled pivot)")
    live.add_argument("--student", help="tokenfactory:<model id>, e.g. tokenfactory:nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
    live.add_argument("--n-students", type=int, default=5, help="samples per pivot")
    live.add_argument("--max-requests", type=int, default=400, help="hard cap on model HTTP attempts, retries included")
    live.add_argument("--student-concurrency", type=int, default=4)
    live.add_argument("--student-max-tokens", type=int, default=8192)
    live.add_argument("--student-pool", help="pool name in action ids (default tf-<model>)")
    live.add_argument("--tf-base-url", default=os.environ.get("TOKENFACTORY_BASE_URL"),
                      help="OpenAI-compatible base URL (default Token Factory; a mock for dry runs)")
    live.add_argument("--tf-cache", help="response cache dir (default ~/.cache/cleave/tokenfactory)")
    a = ap.parse_args(argv)
    students: dict = {}
    if a.students:
        for line in open(a.students):
            r = json.loads(line)
            students.setdefault((r["task"], int(r["turn"])), []).append((r.get("pool", "s"), r["text"]))
    only = parse_pivots(a.pivots)
    specs = [p for p in a.specimens if os.path.exists(os.path.join(p, "task.json"))
             and (only is None or json.load(open(os.path.join(p, "task.json")))["name"] in only)]
    if only is not None:
        missing = set(only) - {json.load(open(os.path.join(p, "task.json")))["name"] for p in specs}
        if missing:
            raise SystemExit(f"--pivots names tasks not among the specimens: {sorted(missing)}")
    os.makedirs(a.out, exist_ok=True)
    stale = clear_stale(a.out)
    sampler = None
    if a.student:
        from pivots.students.sampler import LiveSampler
        from pivots.students.tokenfactory import BASE_URL, DEFAULT_CACHE

        sampler = LiveSampler(a.student, a.n_students, os.path.join(a.out, "students.jsonl"),
                              concurrency=a.student_concurrency, base_url=a.tf_base_url or BASE_URL,
                              max_requests=a.max_requests, cache_dir=a.tf_cache or DEFAULT_CACHE, pool=a.student_pool,
                              max_tokens=a.student_max_tokens)
        n_piv = sum(len(v) for v in only.values()) if only else None
        print(f"live students: {sampler.client.model} as pool {sampler.pool}, {a.n_students} per pivot; "
              f"expected requests {sampler.expected_requests(n_piv) if n_piv else f'{a.n_students} x pivots'} "
              f"(fewer with cache hits, more with retries), cap {a.max_requests}", flush=True)
    rows, pivots = [], []
    t0 = time.time()
    try:
        for p in specs:
            name = json.load(open(os.path.join(p, "task.json")))["name"]
            rs, pv = audit_specimen(p, students, a.store, a.workers, turns=only.get(name) if only else None,
                                    sampler=sampler)
            rows += rs
            pivots += pv
            print(f"{os.path.basename(p.rstrip('/'))}: {len(rs)} actions labelled ({time.time() - t0:.0f}s)", flush=True)
    except Exception as e:
        from pivots.students.tokenfactory import TokenFactoryStop

        if not isinstance(e, TokenFactoryStop):
            raise
        Path(a.out, "student_stats.json").write_text(json.dumps(sampler.stats(), indent=2))
        print(f"STOP: {e}\nnothing labelled; samples received so far are in {a.out}/students.jsonl and the cache "
              f"(a rerun reuses them for free). Stats: {a.out}/student_stats.json", file=sys.stderr)
        raise SystemExit(3)
    with open(os.path.join(a.out, "rows.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    Path(a.out, "pivots.json").write_text(json.dumps(pivots))
    summ = summarize(rows)
    summ["seconds"] = round(time.time() - t0, 1)
    if sampler:
        summ["live_students"] = sampler.stats()
        Path(a.out, "student_stats.json").write_text(json.dumps(sampler.stats(), indent=2))
    Path(a.out, "summary.json").write_text(json.dumps(summ, indent=2))
    lines = report_lines(summ, a.out, stale)
    if sampler:
        st = sampler.stats()
        lines.append(f"live students {st['pool']}: {st['samples']} samples, {st['requests']} requests "
                     f"({st['cache_hits']} cache hits, {len(st['errors'])} failed samples)")
    if a.progress:
        from pivots.audit import progress

        progress.main([*specs, "--audit", a.out, "--out", os.path.join(a.out, "progress.jsonl"),
                       "--store", a.store, "--workers", str(a.workers)])
        lines = [l for l in lines if not l.startswith("removed ")]
        lines.append(f"P labels: {os.path.join(a.out, 'progress.jsonl')}, progress.summary.json")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
