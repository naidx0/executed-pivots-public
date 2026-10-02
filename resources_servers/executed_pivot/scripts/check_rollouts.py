"""Check the rewards Gym reported for stand-in rollouts against the labels already known for the replayed actions.

Joins a `gym eval run` output (rollouts.jsonl) with the audit rows the stand-in replayed from (J = string reward,
X = executed reward) and, optionally, the verified-progress labels (P). Each rollout names its replayed action in
`response.metadata.standin_action` (Responses API path). On the chat-completions path (vllm_model) that metadata
does not survive, so pass the stand-in's --log and run one request per pivot.

    python resources_servers/executed_pivot/scripts/check_rollouts.py rollouts.jsonl \
        --actions out/audit/rows.jsonl \
        --progress out/audit/progress.jsonl [--standin-log standin.jsonl] [--json out.json]

Exit status 1 if Gym's reward differs from X or its j_reward from J on any rollout.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict


def _key(r: dict) -> tuple[str, int, str]:
    return r["task"], int(r["turn"]), r["action"]


def join(rollouts: list[dict], labels: dict, progress: dict, standin_log: list[dict] | None = None) -> list[dict]:
    logged = defaultdict(set)
    for e in standin_log or []:
        logged[e["pivot"]].add(e["action"])
    out = []
    for r in rollouts:
        md = (r.get("response") or {}).get("metadata") or {}
        aid = md.get("standin_action")
        if aid is None:
            acts = logged.get(r["uuid"], set())
            if len(acts) != 1:
                raise SystemExit(f"{r['uuid']}: no standin_action in the response and {len(acts)} logged actions; "
                                 "pass --standin-log from a run with one request per pivot")
            aid = next(iter(acts))
        task, turn = r["uuid"].rsplit(":", 1)
        L = labels[(task, int(turn), aid)]
        P = progress.get((task, int(turn), aid))
        out.append({"uuid": r["uuid"], "action": aid, "reward": int(r["reward"]), "j_reward": int(r["j_reward"]),
                    "score": r.get("score"), "mask_sample": bool(r.get("mask_sample")),
                    "failure_kind": r.get("failure_kind"), "verify_s": r.get("verify_s"),
                    "X": int(L["X"]), "J": int(L["J"]), "X_score": L.get("X_score"),
                    "P": None if P is None else int(P["P"])})
    return out


def report(rows: list[dict]) -> tuple[str, int]:
    lines = [f"rollouts {len(rows)}, distinct pivot-actions {len({(o['uuid'], o['action']) for o in rows})}, "
             f"masked {sum(o['mask_sample'] for o in rows)}"]
    for g, lab in (("reward", "X"), ("j_reward", "J"), ("reward", "P"), ("j_reward", "P")):
        rs = [o for o in rows if o[lab] is not None]
        if not rs:
            continue
        cells = Counter((o[g], o[lab]) for o in rs)
        lines.append(f"{g:8s} vs {lab}: agree {sum(v for (a, b), v in cells.items() if a == b)}/{len(rs)}  "
                     f"(gym, label) cells {dict(sorted(cells.items()))}")
    bad = [o for o in rows if o["reward"] != o["X"] or o["j_reward"] != o["J"]]
    for o in bad:
        lines.append(f"  MISMATCH {o['uuid']} {o['action']}: gym reward {o['reward']} (score {o['score']}, "
                     f"{o['failure_kind']}), j {o['j_reward']} | label X {o['X']} (X_score {o['X_score']}), J {o['J']}")
    v = sorted(o["verify_s"] for o in rows if o["verify_s"] is not None)
    if v:
        lines.append(f"verify_s: mean {statistics.mean(v):.3f}, median {statistics.median(v):.3f}, "
                     f"p90 {v[int(0.9 * (len(v) - 1))]:.3f}, max {v[-1]:.3f}")
    return "\n".join(lines), len(bad)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rollouts")
    ap.add_argument("--actions", required=True, help="audit rows.jsonl the stand-in replayed from")
    ap.add_argument("--progress", help="audit progress.jsonl (verified-progress label P)")
    ap.add_argument("--standin-log", help="the stand-in's --log file (needed on the chat-completions path)")
    ap.add_argument("--json", help="write the joined rows here")
    a = ap.parse_args(argv)
    load = lambda p: [json.loads(line) for line in open(p) if line.strip()]  # noqa: E731
    labels = {_key(r): r for r in load(a.actions)}
    progress = {_key(r): r for r in load(a.progress)} if a.progress else {}
    rows = join(load(a.rollouts), labels, progress, load(a.standin_log) if a.standin_log else None)
    text, n_bad = report(rows)
    print(text)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
