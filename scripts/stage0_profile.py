"""Stage 0: profile NVIDIA's Terminal-Pivot dataset before anything is built.

Reads the JSONL once, streaming. Reports distinct counts to fixed boundaries
(the whole file), never a window. Writes docs/stage0-profile.md.
"""
from __future__ import annotations

import collections
import json
import re
import statistics
import sys
from pathlib import Path

SRC = Path("data/atcb_terminal_pivot_release_final_v2.jsonl")
OUT = Path("docs/stage0-profile.md")

NET = re.compile(r"\b(curl|wget|pip3? install|pip3? download|apt-get|apt |apk add|npm i|npm install|yarn add|git clone|conda install|go get|cargo (add|install)|gem install)\b")
SVC = re.compile(r"\b(systemctl|service |docker |docker-compose|postgres|pg_ctl|mysqld|redis-server|mongod|nginx|uvicorn|gunicorn|flask run|python -m http\.server|nohup|&\s*$)")
SUDO = re.compile(r"\bsudo\b")
TRUNC = re.compile(r"(truncated|\.\.\. \[|output clipped|\[\.\.\.\]|\(\d+ more lines\))", re.I)


def main() -> int:
    if not SRC.exists():
        print("missing", SRC, file=sys.stderr)
        return 2
    rows = 0
    tasks: collections.Counter[str] = collections.Counter()
    trajs: dict[str, dict] = {}
    tools: collections.Counter[str] = collections.Counter()
    harness: collections.Counter[str] = collections.Counter()
    teacher: collections.Counter[str] = collections.Counter()
    schema: collections.Counter[str] = collections.Counter()
    roles: collections.Counter[str] = collections.Counter()
    hist_chars: list[int] = []
    n_msgs: list[int] = []
    ans_chars: list[int] = []
    ans_shapes: collections.Counter[str] = collections.Counter()
    net_rows = svc_rows = sudo_rows = trunc_rows = 0
    first_examples: dict[str, str] = {}
    sample_answer = None
    sample_msgs = None

    with SRC.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            rows += 1
            tasks[o.get("task_name", "")] += 1
            tools[o.get("tool_name", "")] += 1
            schema[o.get("schema_version", "")] += 1
            md = o.get("metadata", {}) or {}
            harness[md.get("harness", "")] += 1
            teacher[md.get("teacher_model", "")] += 1
            tid = md.get("source_trajectory_uid", "")
            t = trajs.setdefault(tid, {"task": o.get("task_name"), "turns": md.get("total_source_agent_turns"), "pivots": set()})
            t["pivots"].add(md.get("pivot_agent_turn_index"))
            msgs = (o.get("responses_create_params") or {}).get("input") or []
            n_msgs.append(len(msgs))
            text = "".join(str(m.get("content", "")) for m in msgs if isinstance(m, dict))
            for m in msgs:
                if isinstance(m, dict):
                    roles[m.get("role", "?")] += 1
            hist_chars.append(len(text))
            ans = o.get("expected_answer", "")
            ans_chars.append(len(ans))
            a = ans.lstrip()
            ans_shapes["json-ish" if a.startswith("{") or a.startswith("[") else "text"] += 1
            if NET.search(text):
                net_rows += 1
            if SVC.search(text):
                svc_rows += 1
            if SUDO.search(text):
                sudo_rows += 1
            if TRUNC.search(text):
                trunc_rows += 1
            if sample_answer is None:
                sample_answer = ans[:700]
                sample_msgs = [(m.get("role"), str(m.get("content", ""))[:600]) for m in msgs[:3] if isinstance(m, dict)]
            for k in ("task_name", "tool_name"):
                first_examples.setdefault(k, o.get(k, ""))

    n_traj = len(trajs)
    turns = [t["turns"] for t in trajs.values() if isinstance(t["turns"], int)]
    coverage = [len(t["pivots"]) / t["turns"] for t in trajs.values() if isinstance(t["turns"], int) and t["turns"]]
    full_cov = sum(1 for c in coverage if c >= 0.999)
    per_task_traj = collections.Counter(t["task"] for t in trajs.values())

    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None

    lines = []
    w = lines.append
    w("# Stage 0 profile: nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1")
    w("")
    w(f"Source file: `{SRC.name}` ({SRC.stat().st_size:,} bytes). Whole file read once; every count is to the end of the file.")
    w("")
    w("## Counts (distinct)")
    w(f"- rows (pivots): **{rows:,}**")
    w(f"- distinct task_name: **{len(tasks):,}**")
    w(f"- distinct source_trajectory_uid: **{n_traj:,}**")
    w(f"- trajectories per task: min {min(per_task_traj.values())}, median {statistics.median(per_task_traj.values())}, max {max(per_task_traj.values())}")
    w(f"- total_source_agent_turns per trajectory: min {min(turns)}, median {statistics.median(turns)}, p90 {q(turns,0.9)}, max {max(turns)}")
    w(f"- pivot coverage (pivots / turns) per trajectory: median {statistics.median(coverage):.2f}; trajectories with every turn present: {full_cov:,} of {n_traj:,}")
    w("")
    w("## Field values")
    w(f"- schema_version: {dict(schema)}")
    w(f"- harness: {dict(harness)}")
    w(f"- teacher_model: {dict(teacher.most_common(8))}")
    w(f"- tool_name: {dict(tools.most_common(8))}")
    w(f"- message roles in history: {dict(roles)}")
    w(f"- expected_answer shape: {dict(ans_shapes)}; chars median {statistics.median(ans_chars):.0f}, p90 {q(ans_chars,0.9)}")
    w("")
    w("## History size")
    w(f"- messages per row: median {statistics.median(n_msgs):.0f}, p90 {q(n_msgs,0.9)}, max {max(n_msgs)}")
    w(f"- history chars per row: median {statistics.median(hist_chars):,.0f}, p90 {q(hist_chars,0.9):,}, max {max(hist_chars):,}")
    w("")
    w("## Reconstruction risk signals (regex over the history text; a row counts once)")
    w(f"- rows whose history contains a network-fetching command (curl/wget/pip install/apt/git clone/npm i): **{net_rows:,}** ({100*net_rows/rows:.1f}%)")
    w(f"- rows whose history contains a long-running service or daemon pattern: **{svc_rows:,}** ({100*svc_rows/rows:.1f}%)")
    w(f"- rows containing sudo: **{sudo_rows:,}** ({100*sudo_rows/rows:.1f}%)")
    w(f"- rows with a truncation marker in observations: **{trunc_rows:,}** ({100*trunc_rows/rows:.1f}%)")
    w("")
    w("These are upper bounds on affected rows, not on affected tasks; the same command appears in every later pivot of its trajectory.")
    w("")
    w("## One row, abbreviated")
    w(f"- task_name: `{first_examples.get('task_name')}`; tool_name: `{first_examples.get('tool_name')}`")
    w("- first messages:")
    for r, c in (sample_msgs or []):
        w(f"  - **{r}**: `{c.replace(chr(10), ' ')[:400]}`")
    w("- expected_answer (first 700 chars):")
    w("```")
    w((sample_answer or "").replace("```", "'''"))
    w("```")
    w("")
    w("## Largest tasks by pivot count")
    for k, v in tasks.most_common(10):
        w(f"- `{k}`: {v} pivots, {per_task_traj[k]} trajectories")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
