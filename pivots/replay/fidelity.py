"""Gate G1: does the rebuilt world reproduce what the expert saw? (plan §7.4)

Per turn: split the recorded and the replayed observation into (command, output)
segments on the shell prompt, normalise both, and score line-level similarity.
A task is admitted when >= 95% of turns score >= 0.9 and no turn errors where
the original succeeded. Truncated recorded outputs are compared only on their
kept head and tail.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from pivots.materialize import TRUNC, split_observation

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\r")
RULES = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?"), "<TS>"),
    (re.compile(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun),? +\w{3} +\d{1,2} +\d{2}:\d{2}:\d{2}( +\w+)? +\d{4}\b"), "<DATE>"),
    (re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +\d{1,2} +(\d{2}:\d{2}|\d{4})\b"), "<DATE>"),
    (re.compile(r"\b\d{1,2}:\d{2}:\d{2}(\.\d+)?\b"), "<TIME>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"\b[0-9a-f]{7,64}\b"), "<HEX>"),
    (re.compile(r"/tmp/tmp[\w.-]+"), "/tmp/<TMP>"),
    (re.compile(r"\b(pid|PID)[ =:]*\d+"), "pid <PID>"),
    (re.compile(r"root@[^\s:#]+"), "root@<HOST>"),
    (re.compile(r"\b\d+(\.\d+)? ?(ms|s|sec|seconds|µs|us|KB|MB|GB|kB|K|M|G|B|bytes)\b"), "<N><UNIT>"),
    (re.compile(r"^(total) \d+$", re.M), r"\1 <N>"),
    (re.compile(r"(^[-dlcbps][-rwxsStT]{9}[.+@]?\s+\d+\s+\S+\s+\S+\s+)\d+", re.M), r"\1<SIZE>"),
]
UNORDERED = re.compile(r"^\s*(ls|find|du|tree|pip3? (list|freeze)|env|printenv)\b")
ERROR = re.compile(r"command not found|No such file or directory|Traceback \(most recent call last\)|"
                   r"Permission denied|cannot access|ModuleNotFoundError|Segmentation fault", re.I)


def normalise(text: str, cmd: str = "") -> list[str]:
    t = ANSI.sub("", text)
    for rx, rep in RULES:
        t = rx.sub(rep, t)
    lines = [ln.rstrip() for ln in t.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    if UNORDERED.match(cmd):
        lines = sorted(lines)
    return lines


def _ratio(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def segment_similarity(recorded: str, rebuilt: str, cmd: str = "") -> float:
    parts = TRUNC.split(recorded)
    if len(parts) == 1:
        return _ratio(normalise(recorded, cmd), normalise(rebuilt, cmd))
    head, tail = normalise(parts[0], cmd), normalise(parts[-1], cmd)
    while tail and not tail[0]:
        tail.pop(0)
    rb = normalise(rebuilt, cmd)
    h = _ratio(head, rb[: len(head)]) if head else 1.0
    t = _ratio(tail, rb[-len(tail):] if tail else []) if tail else 1.0
    return (h * len(head) + t * len(tail)) / max(1, len(head) + len(tail))


@dataclass
class TurnFidelity:
    index: int
    similarity: float
    new_errors: bool
    segments: list[tuple[str, float]] = field(default_factory=list)
    note: str = ""


def turn_fidelity(index: int, recorded_obs: str, rebuilt_obs: str) -> TurnFidelity:
    _, rec = split_observation(recorded_obs)
    _, reb = split_observation(rebuilt_obs)
    if not rec:
        sim = _ratio(normalise(recorded_obs), normalise(rebuilt_obs))
        return TurnFidelity(index, sim, False, [], "no prompt segments in recording; whole-text compare")
    segs, total, weight = [], 0.0, 0
    for i, r in enumerate(rec):
        b = reb[i] if i < len(reb) else None
        sim = 0.0 if b is None else segment_similarity(r.out, b.out, r.cmd)
        w = max(1, len(normalise(r.out, r.cmd)))
        segs.append((r.cmd, round(sim, 4)))
        total += sim * w
        weight += w
    rec_err = sum(len(ERROR.findall(s.out)) for s in rec)
    reb_err = sum(len(ERROR.findall(s.out)) for s in reb)
    note = "" if len(rec) == len(reb) else f"segment count recorded={len(rec)} rebuilt={len(reb)}"
    return TurnFidelity(index, total / weight if weight else 1.0, reb_err > rec_err, segs, note)


@dataclass
class G1:
    turns: list[TurnFidelity]
    min_sim: float = 0.9
    min_share: float = 0.95

    @property
    def share_passing(self) -> float:
        return sum(t.similarity >= self.min_sim for t in self.turns) / max(1, len(self.turns))

    @property
    def admitted(self) -> bool:
        return self.share_passing >= self.min_share and not any(t.new_errors for t in self.turns)

    def summary(self) -> dict:
        return {"turns": len(self.turns), "share_passing": round(self.share_passing, 4), "admitted": self.admitted,
                "worst": sorted(((t.index, round(t.similarity, 3)) for t in self.turns), key=lambda x: x[1])[:5],
                "new_errors": [t.index for t in self.turns if t.new_errors]}


def gate_g1(recorded: list[str], rebuilt: list[str], **kw) -> G1:
    """recorded[i], rebuilt[i]: observation after expert turn i."""
    return G1([turn_fidelity(i, r, b) for i, (r, b) in enumerate(zip(recorded, rebuilt))], **kw)
