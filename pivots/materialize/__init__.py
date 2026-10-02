"""Stage A: materialise NVIDIA's Terminal-Pivot rows into per-task trajectories and E0.

Deterministic, CPU only, no model. For each task:
  - group rows by trajectory, take the longest history, append that row's gold turn
  - split into turns (assistant command batch, observation), and observations into
    (echoed command, output) segments on the shell prompt
  - recover E0: the earliest content of every file the expert read before writing it
  - record files the expert wrote (gold_delta), listings (path inventory), flags
Output: data/tasks/<task>/{manifest.json, trajectories.json, e0/...} and a corpus report.
"""
from __future__ import annotations

import collections
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# not anchored to line start: a file without a trailing newline puts the next prompt on the same line
PROMPT = re.compile(r"(root@[^\s#]+:[^\n#]*# )")
DERIVED = re.compile(r"^/(tmp|logs|var/log|var/tmp|proc|sys|dev|run|root/\.cache)(/|$)")
TRAILING_PROMPT = re.compile(r"\n?root@[^\s#]+:[^\n#]*#\s*$")
TRUNC = re.compile(r"\[\.\.\. output limited to \d+ bytes; \d+ interior bytes omitted \.\.\.\]")
OBS_HEADS = ("New Terminal Output:", "Current Terminal Screen:", "Current terminal state:",
             "Previous response had warnings:", "Previous response had parsing errors:")
NET = re.compile(r"(^|[\s;&|(])(curl|wget|pip3?\s+(install|download)|apt-get|apt\s|apk\s+add|npm\s+(i|install)\b|yarn\s+add|git\s+clone|conda\s+install|go\s+get|cargo\s+(add|install)|gem\s+install)\b")
SVC = re.compile(r"(^|[\s;&|(])(systemctl|service\s|docker\s|docker-compose|pg_ctl|mysqld|redis-server|mongod|nginx|uvicorn|gunicorn|flask\s+run|http\.server|nohup)\b|&\s*$", re.M)
SUDO = re.compile(r"(^|[\s;&|(])sudo\b")
# read commands whose whole stdout is one file's content
READ_FULL = re.compile(r"^\s*cat\s+(?:-A\s+|-v\s+)?(?P<path>/[^\s;|&><]+)\s*$")
READ_PART = re.compile(r"^\s*(?:head|tail)\s+(?:-n\s*\d+|-\d+|-c\s*\d+)?\s*(?P<path>/[^\s;|&><]+)\s*$|^\s*sed\s+-n\s+'?[\d,]+p'?\s+(?P<path2>/[^\s;|&><]+)\s*$")
WRITE_REDIR = re.compile(r">{1,2}\s*(?P<path>/[^\s;|&><]+)")
WRITE_INPLACE = re.compile(r"(^|[\s;&|(])(sed\s+-i|tee\s+(-a\s+)?(?P<path>/[^\s;|&><]+)|mv\s+\S+\s+(?P<dst>/[^\s;|&><]+)|cp\s+\S+\s+(?P<dst2>/[^\s;|&><]+)|rm\s+(-\S+\s+)?(?P<rm>/[^\s;|&><]+)|mkdir\s+(-p\s+)?(?P<mk>/[^\s;|&><]+)|touch\s+(?P<touch>/[^\s;|&><]+)|chmod\s+\S+\s+(?P<chmod>/[^\s;|&><]+))")
HEREDOC = re.compile(r"cat\s*(?:>{1,2}\s*(?P<p1>/[^\s;|&><]+)\s*)?<<-?\s*['\"]?(?P<tag>\w+)['\"]?\s*(?:>{1,2}\s*(?P<p2>/[^\s;|&><]+))?[ \t]*\n(?P<body>.*?)\n(?P=tag)\s*$", re.S | re.M)
LISTING = re.compile(r"^\s*(find\s+(?P<froot>/\S+)|ls\s+(-\S+\s+)*(?P<lroot>/\S+)?)")
# a command is read-only only if it starts with one of these and redirects nothing
READ_ONLY = re.compile(r"^\s*(cat|ls|find|head|tail|grep|rg|wc|pwd|stat|file|which|type|echo|printf|tree|du|df|env|sed\s+-n|awk|sort|uniq|cut|diff|less|more|nl|od|xxd|jq|column|git\s+(status|log|diff|show|branch)|python3?\s+-c\s+['\"]print|ps|top|free|uname|id|whoami|date|true|:)\b")
MUTATING_HINT = re.compile(r">|\bsed\s+-i\b|\btee\b|\|\s*(sudo\s+)?(bash|sh|python|tee)\b")


def is_read_only(cmd: str) -> bool:
    return bool(READ_ONLY.match(cmd)) and not MUTATING_HINT.search(cmd)


@dataclass
class Segment:
    cmd: str
    out: str
    truncated: bool


@dataclass
class Turn:
    index: int
    raw_assistant: str
    valid_json: bool
    keystrokes: list[str]
    task_complete: bool | None
    observation: str
    obs_head: str
    segments: list[Segment] = field(default_factory=list)


@dataclass
class Trajectory:
    uid: str
    teacher: str
    total_turns: int
    rows: int
    hostname: str
    instruction: str
    initial_screen: str
    turns: list[Turn]


def parse_batch(text: str) -> tuple[bool, list[str], bool | None]:
    try:
        o = json.loads(text)
        if not isinstance(o, dict):
            return False, [], None
    except Exception:
        return False, [], None
    cmds = [c.get("keystrokes", "") for c in o.get("commands", []) if isinstance(c, dict)]
    tc = o.get("task_complete")
    return True, cmds, (tc if isinstance(tc, bool) else None)


def split_observation(obs: str) -> tuple[str, list[Segment]]:
    head = next((h for h in OBS_HEADS if obs.startswith(h)), "")
    body = obs[len(head):] if head else obs
    parts = PROMPT.split(body)
    segs: list[Segment] = []
    # parts = [pre, prompt1, rest1, prompt2, rest2, ...]
    for i in range(1, len(parts) - 1, 2):
        rest = parts[i + 1]
        nl = rest.find("\n")
        cmd, out = (rest, "") if nl < 0 else (rest[:nl], rest[nl + 1:])
        out = TRAILING_PROMPT.sub("", out)
        segs.append(Segment(cmd=cmd.strip(), out=out, truncated=bool(TRUNC.search(out))))
    return head, segs


def parse_first_message(content: str) -> tuple[str, str, str]:
    """Return (instruction, initial_screen, hostname) from the Terminus-2 opening message."""
    instr = ""
    m = re.search(r"Task Description:\s*\n(.*?)\n\s*(?:Current [Tt]erminal [Ss](?:tate|creen):)", content, re.S)
    if m:
        instr = m.group(1).strip()
    screen = ""
    m2 = re.search(r"Current [Tt]erminal [Ss](?:tate|creen):\s*\n(.*)$", content, re.S)
    if m2:
        screen = m2.group(1).strip()
    hm = re.search(r"root@([^\s:#]+)", screen)
    return instr, screen, (hm.group(1) if hm else "")


def build_trajectory(row: dict) -> Trajectory:
    msgs = row["responses_create_params"]["input"]
    md = row["metadata"]
    instr, screen, host = parse_first_message(str(msgs[0].get("content", "")))
    turns: list[Turn] = []
    i = 1
    idx = 0
    while i < len(msgs):
        a = msgs[i]
        if a.get("role") != "assistant":
            i += 1
            continue
        obs = str(msgs[i + 1].get("content", "")) if i + 1 < len(msgs) and msgs[i + 1].get("role") == "user" else ""
        ok, ks, tc = parse_batch(str(a.get("content", "")))
        head, segs = split_observation(obs)
        turns.append(Turn(idx, str(a.get("content", "")), ok, ks, tc, obs, head, segs))
        idx += 1
        i += 2
    # the row's own gold turn is the next turn; no observation recorded for it
    ok, ks, tc = parse_batch(row["expected_answer"])
    turns.append(Turn(idx, row["expected_answer"], ok, ks, tc, "", "", []))
    return Trajectory(md["source_trajectory_uid"], md["teacher_model"], md["total_source_agent_turns"],
                      rows=0, hostname=host, instruction=instr, initial_screen=screen, turns=turns)


def recover_files(traj: Trajectory) -> tuple[dict[str, tuple[str, bool]], dict[str, str | None], set[str], dict[str, int]]:
    """E0 (path -> (content, partial)), gold_delta (path -> content or None), listings roots, flags."""
    e0: dict[str, tuple[str, bool]] = {}
    written: dict[str, str | None] = {}
    listings: set[str] = set()
    flags = collections.Counter()
    mutated = False  # once any non-read-only command has run, later reads are not initial state
    for t in traj.turns:
        joined = "".join(t.keystrokes)
        if NET.search(joined):
            flags["net"] += 1
        if SVC.search(joined):
            flags["svc"] += 1
        if SUDO.search(joined):
            flags["sudo"] += 1
        # heredoc writes are only visible in the keystrokes
        for m in HEREDOC.finditer(joined):
            p = m.group("p1") or m.group("p2")
            if p:
                written[p] = m.group("body")
                flags["heredoc"] += 1
        for k in t.keystrokes:
            for m in WRITE_REDIR.finditer(k):
                written.setdefault(m.group("path"), None)
            for m in WRITE_INPLACE.finditer(k):
                for g in ("path", "dst", "dst2", "rm", "mk", "touch", "chmod"):
                    if m.group(g):
                        written.setdefault(m.group(g), None)
        for s in t.segments:
            if not is_read_only(s.cmd):
                mutated = True
            m = READ_FULL.match(s.cmd)
            if m and not GLOB.search(m.group("path")) and not DERIVED.match(m.group("path")):
                p = m.group("path")
                if mutated:
                    flags["late_read"] += 1
                elif p not in written and p not in e0:
                    e0[p] = (s.out, s.truncated)
                    flags["read_full"] += 1
                continue
            m = READ_PART.match(s.cmd)
            if m and not GLOB.search(m.group("path") or m.group("path2") or "") and not DERIVED.match(m.group("path") or m.group("path2") or ""):
                p = m.group("path") or m.group("path2")
                if mutated:
                    flags["late_read"] += 1
                elif p not in written and p not in e0:
                    e0[p] = (s.out, True)
                    flags["read_part"] += 1
                continue
            m = LISTING.match(s.cmd)
            if m:
                listings.add(m.group("froot") or m.group("lroot") or ".")
    return e0, written, listings, dict(flags)


GLOB = re.compile(r"[*?\[\]{}$`]")
ILLEGAL = re.compile(r'[<>:"|?*\x00-\x1f]')


def safe_rel(path: str) -> Path:
    parts = [ILLEGAL.sub(lambda m: "%%%02X" % ord(m.group(0)), p) for p in path.split("/") if p and p not in (".", "..")]
    return Path(*parts) if parts else Path("_root")


def materialize(src: Path, out_root: Path) -> dict:
    by_task: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    row_counts: collections.Counter = collections.Counter()
    with src.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            md = o["metadata"]
            key = (o["task_name"], md["source_trajectory_uid"])
            row_counts[key] += 1
            cur = by_task[o["task_name"]].get(md["source_trajectory_uid"])
            if cur is None or md["pivot_agent_turn_index"] > cur["metadata"]["pivot_agent_turn_index"]:
                by_task[o["task_name"]][md["source_trajectory_uid"]] = o
    report = {"tasks": 0, "trajectories": 0, "turns": 0, "invalid_assistant_json": 0,
              "files_per_task": [], "partial_files": 0, "total_files": 0, "tasks_zero_files": 0,
              "heredoc_writes": 0, "tasks_net": 0, "tasks_svc": 0, "tasks_sudo": 0,
              "e0_conflicts": 0, "instruction_variants": collections.Counter(), "unrecovered_tail_turns": 0}
    for task, trajs in sorted(by_task.items()):
        tdir = out_root / task
        (tdir / "e0").mkdir(parents=True, exist_ok=True)
        e0_task: dict[str, tuple[str, bool]] = {}
        gold_task: dict[str, dict[str, str | None]] = {}
        listings_task: set[str] = set()
        instrs: set[str] = set()
        hosts: set[str] = set()
        flags_task = collections.Counter()
        tj: list[dict] = []
        for uid, row in trajs.items():
            tr = build_trajectory(row)
            tr.rows = row_counts[(task, uid)]
            e0, written, listings, flags = recover_files(tr)
            for p, (c, part) in e0.items():
                if p in e0_task and not part and not e0_task[p][1] and e0_task[p][0].strip() != c.strip():
                    report["e0_conflicts"] += 1
                    report.setdefault("conflict_paths", []).append(f"{task}:{p}")
                if p not in e0_task or (e0_task[p][1] and not part):
                    e0_task[p] = (c, part)
            gold_task[uid] = written
            listings_task |= listings
            instrs.add(tr.instruction)
            hosts.add(tr.hostname)
            for k, v in flags.items():
                flags_task[k] += v
            report["trajectories"] += 1
            report["turns"] += len(tr.turns)
            report["invalid_assistant_json"] += sum(1 for t in tr.turns if not t.valid_json)
            report["unrecovered_tail_turns"] += max(0, tr.total_turns - len(tr.turns))
            tj.append({
                "uid": uid, "teacher": tr.teacher, "total_turns": tr.total_turns, "rows": tr.rows,
                "hostname": tr.hostname, "instruction": tr.instruction, "initial_screen": tr.initial_screen,
                "turns": [{"i": t.index, "valid_json": t.valid_json, "keystrokes": t.keystrokes,
                           "task_complete": t.task_complete, "obs_head": t.obs_head,
                           "segments": [{"cmd": s.cmd, "out": s.out, "truncated": s.truncated} for s in t.segments]}
                          for t in tr.turns],
                "written": written,
            })
        for p, (c, part) in e0_task.items():
            fp = tdir / "e0" / safe_rel(p)
            try:
                fp.parent.mkdir(parents=True, exist_ok=True)
                fp.write_bytes(c.encode("utf-8", "replace"))
            except OSError:
                report["unwritable_paths"] = report.get("unwritable_paths", 0) + 1
        n_files = len(e0_task)
        report["files_per_task"].append(n_files)
        report["total_files"] += n_files
        report["partial_files"] += sum(1 for _, part in e0_task.values() if part)
        report["tasks_zero_files"] += int(n_files == 0)
        report["heredoc_writes"] += flags_task.get("heredoc", 0)
        report["tasks_net"] += int(flags_task.get("net", 0) > 0)
        report["tasks_svc"] += int(flags_task.get("svc", 0) > 0)
        report["tasks_sudo"] += int(flags_task.get("sudo", 0) > 0)
        report["instruction_variants"][len(instrs)] += 1
        manifest = {
            "task": task, "trajectories": len(trajs), "instructions": sorted(instrs), "hostnames": sorted(hosts),
            "e0_files": {p: {"bytes": len(c.encode("utf-8", "replace")), "partial": part} for p, (c, part) in sorted(e0_task.items())},
            "listing_roots": sorted(listings_task), "flags": dict(flags_task),
            "longest_trajectory_turns": max(len(t["turns"]) for t in tj),
        }
        (tdir / "manifest.json").write_bytes(json.dumps(manifest, indent=1).encode("utf-8"))
        (tdir / "trajectories.json").write_bytes(json.dumps(tj).encode("utf-8"))
        report["tasks"] += 1
    return report


def write_report(rep: dict, path: Path) -> str:
    fpt = sorted(rep["files_per_task"])
    q = lambda p: fpt[min(len(fpt) - 1, int(p * len(fpt)))] if fpt else 0
    lines = [
        "# Stage 1 (S1): materialise — corpus report",
        "",
        f"- tasks: **{rep['tasks']}**; trajectories: **{rep['trajectories']}**; parsed turns (incl. each trajectory's gold turn): **{rep['turns']:,}**",
        f"- assistant turns that are not valid JSON (kept, flagged): {rep['invalid_assistant_json']:,}",
        f"- expert turns after a trajectory's last released pivot (unrecoverable): {rep['unrecovered_tail_turns']:,}",
        f"- E0 files recovered: **{rep['total_files']:,}** total; per task median {statistics.median(fpt):.0f}, p10 {q(0.1)}, p90 {q(0.9)}, max {max(fpt) if fpt else 0}",
        f"- partial files (head/tail/sed or truncated): {rep['partial_files']:,} ({100*rep['partial_files']/max(1,rep['total_files']):.1f}%)",
        f"- tasks with zero recovered files: **{rep['tasks_zero_files']}**",
        f"- E0 content conflicts across trajectories of one task (same path, full reads, different bytes): {rep['e0_conflicts']}",
        f"- heredoc writes captured from keystrokes (gold_delta with content): {rep['heredoc_writes']:,}",
        f"- tasks flagged: network-fetching {rep['tasks_net']}, service/background {rep['tasks_svc']}, sudo {rep['tasks_sudo']}",
        f"- instruction variants per task: {dict(sorted(rep['instruction_variants'].items()))}",
        "",
        "Every count is over all 630 tasks. Files are written under `data/tasks/<task>/e0/` (gitignored); manifests are the record.",
    ]
    text = "\n".join(lines) + "\n"
    path.write_bytes(text.encode("utf-8"))
    return text
