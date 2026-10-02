"""H36: re-run X with full detail on the H31 misses (E=1, X=0, source "content") on Nebius Sandboxes.

A measurement only: X is pivots.effect.EffectJudge unchanged, with the Gym path's settings (threshold 0.75,
determinism check, cmd_timeout 30, no next step: ExecutedPivotCore). The pivot is rebuilt as prepare.py builds
it (setup.sh, expert replay, cwd from /var/lib/xp/cwd) on a NebiusWorld, and the candidate text is the rollout's.

    python scripts/h36_miss_detail.py prepare-image [--base python:3.11-slim-bookworm] [--tag xp-h36-h24]
    python scripts/h36_miss_detail.py run [--image xp-h36-h24] [--max-ops 1500] [--max-cost 1.8]
                                          [--out out/h36/detail.jsonl] [--log out/h36/ops.jsonl]

Per row: X computed as EffectJudge.score computes it, from one candidate run whose effects are kept whole
(every changed, extra, missing and deleted path, the output), the ungated score, and the expert's effects at
the pivot and at the next turn (to see a step done early). Spend guard as in H32 (cleave.world.nebius), plus a
cost cap: the run stops (exit 3) once the platform-reported cost passes --max-cost.

Credentials: NEBIUS_API_KEY and NEBIUS_PROJECT_ID from the environment only; everything written goes through
redact. Exit codes: 0 done, 2 credentials missing, 3 stopped by the guard, 4 preflight refused, 5 prepare failed.
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cleave.world.nebius import (MissingCredentials, NebiusWorld, SpendAbort, SpendGuard,  # noqa: E402
                                 make_client, redact, require_env)

LABELS = REPO / "out" / "gym" / "h31" / "labels.jsonl"
IMAGE = "xp-h36-h24"
# h24-runner:local, rebuilt on the sandbox: python:3.11-slim-bookworm (h17) + jq gawk gzip, git make gcc (h20),
# nodejs npm golang-go (h24), /etc/cron.d (h20); procps for a shell's ps
APT = ("set -e; export DEBIAN_FRONTEND=noninteractive; apt-get update; apt-get install -y --no-install-recommends "
       "jq gawk gzip git make gcc libc6-dev nodejs npm golang-go ca-certificates procps util-linux tar; "
       "rm -rf /var/lib/apt/lists/*; mkdir -p /etc/cron.d")
TOOLS = "bash python3 pip jq awk gzip git make gcc node npm go timeout tar find sha256sum base64"
PREFLIGHT = (f'for t in {TOOLS}; do command -v "$t" >/dev/null 2>&1 || echo "MISSING $t"; done; '
             'python3 -c "import setuptools,sys; print(\'PY\', sys.version.split()[0], \'ST\', setuptools.__version__)"; '
             'awk --version 2>/dev/null | head -1; echo DONE')


def say(msg: str, *, err: bool = False) -> None:
    print(redact(msg), file=sys.stderr if err else sys.stdout, flush=True)


def select_rows(path: Path = LABELS) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r for r in rows if r.get("E") == 1 and r.get("X") == 0 and r.get("source") == "content"]


class CostGuard(SpendGuard):
    """SpendGuard plus a cap on the platform-reported cost."""

    def __init__(self, *a, max_cost: float | None = None, **kw):
        super().__init__(*a, **kw)
        self.max_cost = max_cost

    def before_op(self) -> None:
        if self.max_cost is not None and self.cost >= self.max_cost and not self.tripped:
            raise self._trip(f"max_cost={self.max_cost} reached (cost {self.cost:.4f}): run stopped")
        super().before_op()


def prepare_image(a, client, guard) -> int:
    guard.before_op()
    try:
        base = client.images.oci(a.base)
    except Exception as e:  # noqa: BLE001
        guard.on_error(e)
        say(f"h36: could not import {a.base}: {type(e).__name__}: {e}", err=True)
        return 5
    guard.after_op()
    ref = str(base.uuid) if getattr(base, "uuid", None) is not None else a.base
    w = NebiusWorld(ref, workdir="", client=client, guard=guard, run_cap_s=None, _fresh=False)
    t0 = time.perf_counter()
    op = w.run(["bash", "-c", APT], cwd="/", timeout_s=1800)
    guard.write({"prepare_image": {"base": a.base, "exit": op.exit_code, "wall_s": round(time.perf_counter() - t0, 3)}})
    if not op.ok:
        say(f"h36: apt-get in {a.base} failed (exit {op.exit_code})\n{op.stdout[-800:]}\n{op.stderr[-800:]}", err=True)
        return 5
    guard.before_op()
    w._img.tag_as(a.tag)
    guard.after_op()
    say(f"h36: image {w.checkpoint()} tagged {a.tag}; cost {guard.cost:.4f}")
    return 0


def preflight(client, guard, image: str) -> str | None:
    w = NebiusWorld(image, workdir="", client=client, guard=guard, run_cap_s=60, _fresh=False)
    op = w.run(["bash", "-c", PREFLIGHT], cwd="/", keep=False, timeout_s=30)
    missing = re.findall(r"^MISSING (\S+)", op.stdout, re.M)
    guard.write({"preflight": {"image": image, "exit": op.exit_code, "missing": missing,
                               "stdout": op.stdout[-400:]}})
    say(f"preflight: {op.stdout.strip()}")
    if "DONE" not in op.stdout:
        return f"preflight did not report (exit {op.exit_code}): {op.stderr[-300:]}"
    if missing:
        return f"image {image} lacks {', '.join(missing)}"
    return None


def _kind(p: str, h: str, pre: set[str]) -> str:
    return ("dir" if h == "D" else "link" if h.startswith("L:") else "file") + ("" if p in pre else "+new")


def effects_view(eff, pre: set[str]) -> dict:
    mc = eff.meaningful_changes()
    return {"changed": {p: _kind(p, h, pre) for p, h in sorted(mc.items())},
            "structural": sorted(eff.structural_changes(pre)),
            "deleted": sorted(eff.deleted_vs(pre)),
            "raw_changed_n": len(eff.changed),
            "output_head": eff.output[:1500], "output_len": len(eff.output),
            "exit": eff.exit_code, "timed_out": eff.timed_out, "final_cwd": eff.final_cwd,
            "procs": eff.procs[:10], "backend_error": eff.backend_error}


def compare_dict(c: dict) -> dict:
    return {"reward": round(c["reward"], 4), "state": None if c["state"] is None else round(c["state"], 4),
            "output": round(c["output"], 4), "penalties": c["penalties"],
            "detail": {k: v for k, v in c["detail"].items() if k != "cand_output_head"}}


def run_task(task: str, rows: list[dict], factory, workers: int, next_step: bool = True,
             determinism: bool = True) -> list[dict]:
    from pivots import specimen as S
    from pivots.audit.run import _cwd_at
    from pivots.effect import EffectJudge, Pivot
    from pivots.effect.jreward import check_task_complete
    from pivots.effect.terminus import parse_action
    from pivots.effect.xreward import source_files

    sp = S.load(REPO / "specimens" / task)
    turns = sorted({r["turn"] for r in rows})
    upto = min(len(sp.actions), max(turns) + (2 if next_step else 1))  # anchor(t + 1) for the expert's next step
    w = S.build(factory, sp)
    rr = S.run_expert(w, sp, upto=upto)
    # the Gym path (ExecutedPivotCore); without the determinism check (to save runs) self_agreement is None here,
    # and H31 already masked every pivot whose expert run did not repeat (none of these rows was masked)
    judge = EffectJudge(w, threshold=0.75, check_determinism=determinism)
    pivots, nxt = {}, {}
    for t in turns:
        cwd = sp.workdir if t == 0 else _cwd_at(w, rr.anchor(t), sp.workdir)
        pivots[t] = Pivot(f"{sp.name}:{t}", rr.anchor(t), cwd, sp.actions[t], cmd_timeout=30)
        judge.prepare(pivots[t])
        if next_step and t + 1 < len(sp.actions):
            ncwd = _cwd_at(w, rr.anchor(t + 1), sp.workdir)
            npv = Pivot(f"{sp.name}:{t + 1}", rr.anchor(t + 1), ncwd, sp.actions[t + 1], cmd_timeout=30)
            eff, _ = judge._run(npv, parse_action(sp.actions[t + 1]).action.commands)
            nxt[t] = eff

    def one(r: dict) -> dict:
        t, text = r["turn"], r["text"]
        pv = pivots[t]
        prep = judge.prepare(pv)
        ref, pre = prep["ref"], prep["pre"]
        out = {"i": r["i"], "uuid": r["uuid"], "task": task, "turn": t, "h31_X": r["X"], "E": r["E"], "P": r["P"],
               "h31_score": r["score"], "h31_failure": r["failure_kind"],
               "checks": [r["checks_before"], r["checks_after"], r["checks_expert"]],
               "self_agreement": prep["self_agreement"], "cwd": pv.cwd,
               "expert_text": sp.actions[t], "text": text,
               "ref": effects_view(ref, pre), "ref_source_files": sorted(source_files(ref, pre)),
               "ref_has_state": bool(ref.meaningful_changes() or ref.deleted_vs(pre))}
        if t in nxt:
            out["next_ref"] = effects_view(nxt[t], pre)
        # EffectJudge.score, step by step, so the one candidate run also yields its whole effects
        pa = parse_action(text)
        failure = None
        if pa.action is None:
            failure = "model_output_invalid"
        elif not check_task_complete(pa.action.raw, prep["expected"]):
            failure = "task_complete_check_failed"  # X = 0 before any run; the batch still runs for the detail
        out["run1"] = {"X": 0, "reward": 0.0, "failure": failure}
        if pa.action is not None:
            cand, warns = judge._run(pv, pa.action.commands)
            if cand.backend_error:
                out["run1"] = {"X": 0, "reward": 0.0, "failure": "backend_error", "error": cand.backend_error}
                return out
            c = judge._compare(ref, cand, pre, prep["expected"], pa.action.raw)
            u = judge._compare(ref, cand, pre, prep["expected"], pa.action.raw, gate=False)
            x = int(c["reward"] >= judge.threshold) if failure is None else 0
            out["run1"] = {**compare_dict(c), "X": x, "failure": failure,
                           "effect_X": int(c["reward"] >= judge.threshold)}
            out["ungated"] = compare_dict(u)
            out["cand"] = effects_view(cand, pre)
            out["cand_warnings"] = warns
            out["cand_elapsed"] = round(cand.elapsed, 3)
        return out

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, rows))


def run(a, client, guard) -> int:
    if a.preflight:
        why = preflight(client, guard, a.image)
        if why:
            say(f"h36: preflight refused: {why}", err=True)
            return 4
    rows = select_rows()
    if a.only:
        rows = [r for r in rows if r["task"] in a.only.split(",")]
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    factory = functools.partial(NebiusWorld, a.image, client=client, guard=guard, run_cap_s=a.run_cap_s)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and a.resume:
        done = {json.loads(x)["i"] for x in out.read_text(encoding="utf-8").splitlines() if x.strip()}
    aborted = None
    t0 = time.perf_counter()
    order = a.order.split(",") if a.order else []
    tasks = sorted(by_task.items(), key=lambda kv: (order.index(kv[0]) if kv[0] in order else len(order), kv[0]))
    with open(out, "a" if a.resume else "w", encoding="utf-8", newline="\n") as fh:
        for task, items in tasks:
            items = [r for r in items if r["i"] not in done]
            if not items:
                continue
            ops0, c0, ts = guard.ops, guard.cost, time.perf_counter()
            try:
                res = run_task(task, items, factory, a.workers, next_step=a.next_step, determinism=a.determinism)
            except SpendAbort as e:
                aborted = str(e)
                break
            except Exception as e:  # noqa: BLE001 - one specimen failing does not stop the others
                if guard.tripped:
                    aborted = guard.tripped
                    break
                say(f"h36: {task} failed: {type(e).__name__}: {e}", err=True)
                guard.write({"task": task, "status": "error", "error": f"{type(e).__name__}: {e}"[:1000]})
                continue
            if guard.tripped:
                aborted = guard.tripped
                break
            for r in res:
                fh.write(redact(json.dumps(r)) + "\n")
            fh.flush()
            guard.write({"task": task, "rows": len(res), "ops": guard.ops - ops0, "cost": round(guard.cost - c0, 6),
                         "wall_s": round(time.perf_counter() - ts, 1), "status": "ok"})
            flips = [r["i"] for r in res if r["run1"]["X"] != r["h31_X"]]
            say(f"{task}: {len(res)} rows, flips {flips}, ops {guard.ops - ops0}, cost {guard.cost - c0:.4f}, "
                f"total cost {guard.cost:.4f}, {time.perf_counter() - ts:.0f}s")
    guard.write({"summary": {"ops": guard.ops, "cost": round(guard.cost, 6),
                             "wall_s": round(time.perf_counter() - t0, 1), "aborted": aborted}})
    say(f"h36: ops {guard.ops}, cost {guard.cost:.4f}, wall {time.perf_counter() - t0:.0f}s -> {out}")
    if aborted:
        say(f"h36: STOPPED by the spend guard: {aborted}", err=True)
        return 3
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("prepare-image")
    pi.add_argument("--base", default="python:3.11-slim-bookworm")
    pi.add_argument("--tag", default=IMAGE)
    r = sub.add_parser("run")
    r.add_argument("--image", default=IMAGE)
    r.add_argument("--out", default=str(REPO / "out" / "h36" / "detail.jsonl"))
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--run-cap-s", type=float, default=600.0)
    r.add_argument("--only", help="comma-separated tasks")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--next-step", action=argparse.BooleanOptionalAction, default=True,
                   help="also run the expert's next step (2 more runs per pivot)")
    r.add_argument("--determinism", action=argparse.BooleanOptionalAction, default=True,
                   help="run the expert twice per pivot (self_agreement)")
    r.add_argument("--order", help="comma-separated tasks: run these first, in this order")
    r.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True)
    for p in (pi, r):
        p.add_argument("--log", default=str(REPO / "out" / "h36" / "ops.jsonl"))
        p.add_argument("--max-ops", type=int, default=1500)
        p.add_argument("--max-cost", type=float, default=1.8)
    a = ap.parse_args(argv)
    try:
        require_env()
    except MissingCredentials as e:
        say(f"h36: {e}", err=True)
        return 2
    guard = CostGuard(max_steps=1 << 30, max_ops=a.max_ops, log_path=a.log, max_cost=a.max_cost)
    client = make_client()
    try:
        return prepare_image(a, client, guard) if a.cmd == "prepare-image" else run(a, client, guard)
    except SpendAbort as e:
        say(f"h36: STOPPED by the spend guard: {e}", err=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
