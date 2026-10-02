"""Validate every specimen end to end on the overlay backend.

    python3 specimens/validate.py [--only NAME ...] [--store DIR] [--keep-store]

Per specimen:
  lint     trajectory shape: 6-12 turns, 1-5 commands per working turn, keystrokes end in "\\n",
           no interactive-program warnings, only the last turn has task_complete true
  gold     verifier on the checkpoint after the last expert turn, 3 times: all must be 1
  nop      verifier on the freshly built environment: must be 0
  partial  verifier on every checkpoint before the first one that passes: all must be 0
           (and there must be at least 2 of them)
  replay   rebuild from scratch twice more, replay, and verify base + every step:
           the verdict vectors of all three builds must be identical

Prints a table and exits 1 if any check fails.
"""
from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from cleave.world.overlay import OverlayWorld, available  # noqa: E402
from pivots import specimen as S  # noqa: E402
from pivots.effect.terminus import keystrokes_to_script, parse_action  # noqa: E402


def lint(sp: S.Specimen) -> list[str]:
    problems = []
    if not 6 <= len(sp.actions) <= 12:
        problems.append(f"{len(sp.actions)} turns (want 6-12)")
    for i, raw in enumerate(sp.actions):
        pr = parse_action(raw)
        if pr.action is None or not pr.strict_ok:
            problems.append(f"turn {i}: does not parse strictly ({pr.error})")
            continue
        a = pr.action
        last = i == len(sp.actions) - 1
        if a.task_complete != last:
            problems.append(f"turn {i}: task_complete={a.task_complete}")
        if not last and not 1 <= len(a.commands) <= 5:
            problems.append(f"turn {i}: {len(a.commands)} commands (want 1-5)")
        for c in a.commands:
            if not c.keystrokes.endswith("\n"):
                problems.append(f"turn {i}: keystrokes without trailing newline: {c.keystrokes[:40]!r}")
    for i, b in enumerate(sp.batches()):
        _script, warns = keystrokes_to_script(b)
        problems += [f"turn {i}: {w}" for w in warns]
    return problems


def verdicts(wf, sp: S.Specimen) -> tuple[list[int | None], object, object]:
    """Build, replay, verify base and every step. Returns ([base, step0, step1, ...], world, replay)."""
    w = S.build(wf, sp)
    base = w.checkpoint()
    rr = S.run_expert(w, sp)
    v = [S.verify(w, base, sp)["reward"]]
    v += [S.verify(w, s.checkpoint, sp)["reward"] for s in rr.steps]
    return v, w, rr


def validate(name: str, wf) -> dict:
    t0 = time.time()
    sp = S.load(HERE / name)
    row = {"name": name, "turns": len(sp.actions), "lint": lint(sp)}
    v1, w, rr = verdicts(wf, sp)
    final = rr.steps[-1].checkpoint
    gold = [v1[-1]] + [S.verify(w, final, sp)["reward"] for _ in range(2)]
    row["gold"] = gold
    row["nop"] = v1[0]
    steps = v1[1:]
    first_pass = next((i for i, r in enumerate(steps) if r == 1), None)
    partial = steps[:first_pass] if first_pass is not None else steps
    row["partial"] = partial
    row["first_pass_step"] = first_pass
    row["step_exits"] = [s.exit_code for s in rr.steps]
    row["verdicts"] = v1
    reps = [verdicts(wf, sp)[0] for _ in range(2)]
    row["replay_verdicts"] = reps
    row["replay_same"] = all(r == v1 for r in reps)
    row["ok"] = (not row["lint"] and gold == [1, 1, 1] and row["nop"] == 0 and len(partial) >= 2
                 and all(r == 0 for r in partial) and row["replay_same"])
    row["secs"] = round(time.time() - t0, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--store", default="/tmp/cleave-specimens-store")
    ap.add_argument("--keep-store", action="store_true", help="do not wipe the overlay store before and after")
    ap.add_argument("--json", help="also write the rows to this JSON file")
    args = ap.parse_args()
    if not available():
        print("overlay backend unavailable (needs unprivileged user namespaces + overlayfs)")
        return 2
    names = args.only or sorted(p.name for p in HERE.iterdir() if (p / "task.json").exists())
    if not args.keep_store:
        shutil.rmtree(args.store, ignore_errors=True)
    wf = functools.partial(OverlayWorld, store=args.store)
    rows = []
    for n in names:
        try:
            rows.append(validate(n, wf))
        except Exception as e:  # noqa: BLE001
            rows.append({"name": n, "ok": False, "error": repr(e)})
        r = rows[-1]
        print(f"... {n}: {'ok' if r['ok'] else 'FAIL'} ({r.get('secs', '?')}s)", flush=True)

    print()
    hdr = f"{'specimen':<24} {'turns':>5}  {'lint':<5} {'gold x3':<8} {'nop':>3}  {'partials (all 0?)':<24} {'replay x3':<9} {'result'}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<24} ERROR {r['error']}")
            continue
        part = "".join(str(x) for x in r["partial"])
        part = f"{part} ({len(r['partial'])} ckpts)"
        print(f"{r['name']:<24} {r['turns']:>5}  {'ok' if not r['lint'] else 'FAIL':<5} "
              f"{''.join(str(x) for x in r['gold']):<8} {r['nop']:>3}  {part:<24} "
              f"{'same' if r['replay_same'] else 'DIFFER':<9} {'PASS' if r['ok'] else 'FAIL'}")
        for p in r["lint"]:
            print(f"    lint: {p}")
    print()
    for r in rows:
        if "verdicts" in r:
            print(f"{r['name']}: verdicts base,step0..stepN = {r['verdicts']}  step exit codes = {r['step_exits']}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
    if not args.keep_store:
        shutil.rmtree(args.store, ignore_errors=True)
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
