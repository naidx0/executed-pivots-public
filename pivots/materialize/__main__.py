from __future__ import annotations

import argparse
from pathlib import Path

from . import materialize, write_report


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m pivots.materialize")
    ap.add_argument("--src", default="data/atcb_terminal_pivot_release_final_v2.jsonl")
    ap.add_argument("--out", default="data/tasks")
    ap.add_argument("--report", default="docs/stage1-materialize.md")
    a = ap.parse_args()
    rep = materialize(Path(a.src), Path(a.out))
    print(write_report(rep, Path(a.report)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
