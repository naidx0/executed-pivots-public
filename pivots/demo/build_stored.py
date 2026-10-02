"""Write pivots/demo/stored.json from a full audit: the decisive pivots, the student samples stored there,
and the audit's labels for every action at them (shown next to live results; never used to score).

    python -m pivots.demo.build_stored <audit dir> [--out pivots/demo/stored.json]

The audit dir holds rows.jsonl (pivots.audit.run) and, optionally, progress.jsonl (pivots.audit.progress).
Decisive pivots are the audit's own definition: the no_op control fails the task there (E = 0).
Controls are not stored: the demo rebuilds them from the teacher's batch with pivots.audit.controls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(audit_dir: str) -> dict:
    d = Path(audit_dir)
    rows = [json.loads(line) for line in open(d / "rows.jsonl", encoding="utf-8")]
    prog = {}
    if (d / "progress.jsonl").exists():
        prog = {(p["task"], p["turn"], p["action"]): p for p in map(json.loads, open(d / "progress.jsonl"))}
    decisive = sorted({(r["task"], r["turn"]) for r in rows if r["action"] == "ctrl:no_op" and r["E"] == 0})
    out = {"source": f"audit {d.name}: {len(rows)} labelled actions", "decisive": [f"{t}:{u}" for t, u in decisive],
           "students": {}, "audit": {}}
    for r in rows:
        if (r["task"], r["turn"]) not in decisive:
            continue
        key = f"{r['task']}:{r['turn']}"
        p = prog.get((r["task"], r["turn"], r["action"]))
        out["audit"].setdefault(key, {})[r["action"]] = {
            "J": r["J"], "J_sim": None if r["J_sim"] is None else round(r["J_sim"], 4), "X": r["X"],
            "X_score": r["X_score"], "E": r["E"], "P": None if p is None else p["P"],
            "checks": None if p is None else [p["checks_before"], p["checks_expert"], p["checks_after"]]}
        if r["action"].startswith("student:"):
            out["students"].setdefault(key, []).append({"id": r["action"], "text": r["text"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("audit_dir")
    ap.add_argument("--out", default=str(HERE / "stored.json"))
    a = ap.parse_args(argv)
    data = build(a.audit_dir)
    Path(a.out).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"{a.out}: {len(data['decisive'])} decisive pivots, "
          f"{sum(len(v) for v in data['students'].values())} student samples")


if __name__ == "__main__":
    main()
