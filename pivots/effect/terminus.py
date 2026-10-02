"""Terminus-2 action parsing and keystroke handling.

A Terminus-2 action is a JSON object:
    {"analysis": str, "plan": str,
     "commands": [{"keystrokes": str, "duration": float}, ...],
     "task_complete": bool}

The upstream NeMo Gym verifier (resources_servers/terminus_judge) calls
json.loads on the raw model text, so any prose or <think> block around the
JSON fails. We expose both a strict parse (same behavior as upstream) and a
lenient parse (what the Terminus-2 harness itself tolerates: "Extra text
before or after the JSON will generate warnings but be tolerated").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

# tmux-style special keys the Terminus-2 prompt documents
SPECIAL_KEYS = {"C-c", "C-d", "C-z", "C-l", "Escape", "Enter", "Tab", "Up", "Down"}

# programs that take over the terminal; in non-pty execution they hang or misbehave
INTERACTIVE_PROGRAMS = {
    "vim", "vi", "nano", "emacs", "less", "more", "top", "htop", "man",
    "python", "python3", "ipython", "node", "irb", "psql", "mysql", "sqlite3",
    "gdb", "ssh", "ftp", "telnet", "watch", "tmux", "screen",
}


@dataclass
class Command:
    keystrokes: str
    duration: float = 1.0


@dataclass
class Action:
    analysis: str
    plan: str
    commands: list[Command]
    task_complete: bool = False
    raw: dict = field(default_factory=dict)

    def keystrokes(self) -> list[str]:
        return [c.keystrokes for c in self.commands]


@dataclass
class ParseResult:
    action: Action | None
    error: str | None
    strict_ok: bool  # would upstream json.loads + schema accept it?
    extracted_from_noise: bool = False


def _validate(obj) -> tuple[Action | None, str | None]:
    if not isinstance(obj, dict):
        return None, f"top-level JSON is {type(obj).__name__}, not object"
    for key in ("analysis", "plan", "commands"):
        if key not in obj:
            return None, f"missing required field '{key}'"
    if not isinstance(obj["analysis"], str) or not isinstance(obj["plan"], str):
        return None, "analysis/plan must be strings"
    if not isinstance(obj["commands"], list):
        return None, "commands must be a list"
    cmds = []
    for i, c in enumerate(obj["commands"]):
        if not isinstance(c, dict) or "keystrokes" not in c:
            return None, f"commands[{i}] missing keystrokes"
        if not isinstance(c["keystrokes"], str):
            return None, f"commands[{i}].keystrokes must be a string"
        dur = c.get("duration", 1.0)
        if not isinstance(dur, (int, float)) or isinstance(dur, bool):
            return None, f"commands[{i}].duration must be a number"
        cmds.append(Command(c["keystrokes"], float(dur)))
    tc = obj.get("task_complete", False)
    if not isinstance(tc, bool):
        return None, "task_complete must be a boolean"
    return Action(obj["analysis"], obj["plan"], cmds, tc, obj), None


def _balanced_objects(text: str):
    """Yield every top-level balanced {...} substring, respecting JSON strings."""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[start : i + 1]


def parse_action(text: str) -> ParseResult:
    """Parse a model response into an Action.

    strict_ok mirrors the upstream terminus_judge: raw json.loads must succeed
    and the schema must validate. The lenient path strips <think> blocks and
    code fences and takes the last balanced JSON object that validates.
    """
    strict_ok = False
    try:
        obj = json.loads(text)
        action, err = _validate(obj)
        if action is not None:
            return ParseResult(action, None, True)
        strict_err = err
    except (json.JSONDecodeError, TypeError) as e:
        strict_err = f"not valid JSON: {e}"

    cleaned = _THINK_RE.sub("", text or "")
    # an unclosed <think> means the model never finished reasoning
    if "<think>" in cleaned and "</think>" not in cleaned:
        return ParseResult(None, "unterminated <think> block", strict_ok)
    candidates = [m.group(1) for m in _FENCE_RE.finditer(cleaned)]
    candidates += list(_balanced_objects(cleaned))
    last_err = strict_err
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = f"not valid JSON: {e}"
            continue
        action, err = _validate(obj)
        if action is not None:
            return ParseResult(action, None, strict_ok, extracted_from_noise=True)
        last_err = err
    return ParseResult(None, last_err, strict_ok)


def first_word(line: str) -> str:
    line = line.strip()
    # skip env assignments and sudo
    toks = line.split()
    while toks and ("=" in toks[0] and not toks[0].startswith("=") or toks[0] in ("sudo", "env", "time", "nohup")):
        toks = toks[1:]
    return toks[0].rsplit("/", 1)[-1] if toks else ""


def is_interactive(keystrokes: str) -> bool:
    """True if the keystrokes launch a program that takes over the terminal.

    `python3 script.py` or `python3 -c ...` is not interactive; bare `python3` is.
    """
    for line in keystrokes.splitlines():
        w = first_word(line)
        if w not in INTERACTIVE_PROGRAMS:
            continue
        rest = line.strip().split()[1:]
        if w.startswith("python") or w in ("node", "irb", "ipython"):
            if not rest:
                return True
            continue
        if w in ("psql", "mysql", "sqlite3") and any(a in ("-c", "-e") for a in rest):
            continue
        return True
    return False


def keystrokes_to_script(commands: list[Command]) -> tuple[str, list[str]]:
    """Translate a Terminus keystroke batch into a bash script.

    Returns (script, warnings). Keystrokes ending in newline become lines.
    Text without a trailing newline is typed but never submitted, so it has
    no effect in a real terminal; we drop it (and warn). Special keys C-c/C-d
    between commands have no effect in batch mode.
    """
    lines: list[str] = []
    warnings: list[str] = []
    pending = ""
    for c in commands:
        ks = c.keystrokes
        if ks.strip() in SPECIAL_KEYS:
            if pending:
                warnings.append(f"special key {ks.strip()!r} discarded typed text {pending!r}")
                pending = ""
            continue
        pending += ks
        if "\n" in pending:
            *done, pending = pending.split("\n")
            lines.extend(done)
    if pending.strip():
        warnings.append(f"unsubmitted keystrokes (no trailing newline): {pending!r}")
    if any(is_interactive(line) for line in lines):
        warnings.append("interactive program in batch; stdin closed, it may exit early")
    return "\n".join(lines) + ("\n" if lines else ""), warnings
