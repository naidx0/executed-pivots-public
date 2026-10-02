"""Sample live students from pre-built pivot prompts, with no terminal world needed. This runs on Windows too.

    python -m pivots.students.sample_prompts --pivots access-log-summary:4,ledger-git-revert:3 --n 5 \
        --out out/tf/students.jsonl [--model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B] [--max-requests 44]

The prompts come from pivots/students/prompts/pivots.jsonl, which holds all 45 pivots. It was written by
`python -m pivots.audit.prompts`, which replays each expert in an overlay world (Linux only). This splits the
live audit in two:

  1. sample (anywhere with the key and network access): this module writes students.jsonl
  2. label  (Linux or WSL):  python -m pivots.audit.run specimens/*/ --students students.jsonl --pivots ... --progress

Budget, stop rules, cache and key handling are those of pivots.students.tokenfactory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from pivots.students.sampler import LiveSampler
from pivots.students.tokenfactory import BASE_URL, DEFAULT_CACHE, DEFAULT_MODEL, TokenFactoryStop

PROMPTS = Path(__file__).with_name("prompts") / "pivots.jsonl"


def load_prompts(path: Path = PROMPTS) -> dict[tuple[str, int], list[dict]]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out[(r["task"], int(r["turn"]))] = r["messages"]
    return out


def parse_pivots(s: str) -> list[tuple[str, int]]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            task, turn = part.rsplit(":", 1)
            out.append((task, int(turn)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pivots", required=True, help="task:turn,task:turn")
    ap.add_argument("--n", type=int, default=5, help="samples per pivot")
    ap.add_argument("--out", required=True, help="students.jsonl to write")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=os.environ.get("TOKENFACTORY_BASE_URL", BASE_URL))
    ap.add_argument("--max-requests", type=int, default=400)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--prompts", default=str(PROMPTS))
    a = ap.parse_args(argv)

    prompts = load_prompts(Path(a.prompts))
    want = parse_pivots(a.pivots)
    missing = [p for p in want if p not in prompts]
    if missing:
        raise SystemExit(f"no prompt for {missing}; known tasks: {sorted({t for t, _ in prompts})}")
    sampler = LiveSampler(f"tokenfactory:{a.model}", a.n, a.out, concurrency=a.concurrency, base_url=a.base_url,
                          max_requests=a.max_requests, cache_dir=a.cache_dir, max_tokens=a.max_tokens)
    print(f"sampling {a.n} x {len(want)} pivots = {a.n * len(want)} requests (fewer on cache hits), "
          f"cap {a.max_requests}, model {sampler.client.model} -> {a.out}", flush=True)
    by_task: dict[str, list[tuple[int, list[dict]]]] = {}
    for task, turn in want:
        by_task.setdefault(task, []).append((turn, prompts[(task, turn)]))
    stats_path = Path(a.out).with_name("student_stats.json")
    try:
        for task, pivots in by_task.items():
            got = sampler(SimpleNamespace(name=task), pivots)
            print(f"{task}: {sum(len(v) for v in got.values())} samples", flush=True)
    except TokenFactoryStop as e:
        stats_path.write_text(json.dumps(sampler.stats(), indent=2))
        print(f"STOP: {e}\nsamples received so far are in {a.out} (a rerun reuses the cache)", file=sys.stderr)
        return 3
    stats_path.write_text(json.dumps(sampler.stats(), indent=2))
    st = sampler.stats()
    print(f"done: {st['samples']} samples, {st['requests']} requests, {st['cache_hits']} cache hits, "
          f"{len(st['errors'])} failed samples; label them on Linux/WSL with\n"
          f"  python -m pivots.audit.run specimens/*/ --students {a.out} --pivots {a.pivots} --progress --out <dir>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
