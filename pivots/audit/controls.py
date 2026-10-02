"""Control actions for a pivot, generated from the expert's own batch (plan §7.5).

Each control is built so that it can land in only one cell, which is what
makes it a control:

  expert           the gold batch itself                 J=1, executed=1
  paraphrase       same effect, different keystrokes     J=0 (usually), executed=1
  flipped          near-identical keystrokes, different  J=1 (usually), executed=0
                   effect (drop sed -i, > to >>, one
                   character changed in written content)
  no_op            empty batch (wait)                    J=0, executed=0 unless the gold is a wait
  destructive      removes the working tree              J=0, executed=0
  early_complete   gold keystrokes + task_complete=true  J=1 (one-way check), executed=0 unless
                   when the gold says false                the task is already done

Generators are rule-based and conservative: when no safe rewrite applies, the
control is omitted for that pivot (and counted), never guessed.
"""
from __future__ import annotations

import json
import re
import shlex

from pivots.effect.jreward import command_similarity

HEREDOC = re.compile(r"^(?P<pre>cat\s*>\s*(?P<path>\S+)\s*<<-?\s*'?(?P<tag>\w+)'?)\n(?P<body>.*?)\n(?P=tag)\n?$", re.S)
SED_I = re.compile(r"\bsed\s+-i(\s+|'')")
SED_SUB = re.compile(r"s/((?:[^/\\]|\\.)*)/((?:[^/\\]|\\.)*)/(\w*)")
LS_FLAGS = re.compile(r"^(\s*ls\s+)-([a-zA-Z]{2,})(\b.*)$", re.S)
GREP_FLAGS = re.compile(r"^(\s*grep\s+)-([a-zA-Z]{2,})(\b.*)$", re.S)
ECHO_REDIR = re.compile(r"^(\s*)echo\s+(?P<q>['\"])(?P<text>[^'\"$`\\]*)(?P=q)\s*(?P<op>>>?)\s*(?P<path>\S+)\s*$")
FLIP_CHARS = [("+", "-"), ("<", ">"), ("==", "!="), ("True", "False"), ("1", "2"), ("0", "1"), ("a", "e")]


def _action(gold: dict, keystrokes: list[str], **over) -> str:
    out = {"analysis": gold.get("analysis", ""), "plan": gold.get("plan", ""),
           "commands": [{"keystrokes": k, "duration": 1.0} for k in keystrokes],
           "task_complete": gold.get("task_complete", False)}
    out.update(over)
    return json.dumps(out)


def _ks(gold: dict) -> list[str]:
    return [c.get("keystrokes", "") for c in gold.get("commands", [])]


def paraphrase_keystroke(k: str) -> str | None:
    m = HEREDOC.match(k)
    if m:
        path = m.group("path").strip("'\"")
        code = f"open({path!r}, 'w').write({m.group('body') + chr(10)!r})"
        return f"python3 -c {shlex.quote(code)}\n"
    m = LS_FLAGS.match(k)
    if m and len(set(m.group(2))) == len(m.group(2)):
        return f"{m.group(1)}-{m.group(2)[::-1]}{m.group(3)}"
    m = GREP_FLAGS.match(k)
    if m and len(set(m.group(2))) == len(m.group(2)) and not set(m.group(2)) & set("eEfABCm"):
        return f"{m.group(1)}-{m.group(2)[::-1]}{m.group(3)}"
    m = ECHO_REDIR.match(k.rstrip("\n"))
    if m:
        return f"{m.group(1)}printf '%s\\n' {shlex.quote(m.group('text'))} {m.group('op')} {m.group('path')}\n"
    if SED_I.search(k) and SED_SUB.search(k) and "|" not in k:
        return SED_SUB.sub(lambda s: f"s|{s.group(1)}|{s.group(2)}|{s.group(3)}", k, count=1)
    return None


def flip_keystroke(k: str) -> str | None:
    if SED_I.search(k):
        return SED_I.sub("sed ", k, count=1)
    m = HEREDOC.match(k)
    if m:
        body = m.group("body")
        for a, b in FLIP_CHARS:
            i = body.rfind(a)
            if i >= 0:
                nb = body[:i] + b + body[i + len(a):]
                return k.replace(body, nb, 1)
    m = ECHO_REDIR.match(k.rstrip("\n"))
    if m and m.group("op") == ">":
        return k.replace(">", ">>", 1)
    return None


def controls(gold_text: str, *, workdir: str = "/app") -> dict[str, str]:
    gold = json.loads(gold_text)
    ks = _ks(gold)
    out = {"expert": gold_text}
    para = [paraphrase_keystroke(k) for k in ks]
    if any(p is not None for p in para):
        out["paraphrase"] = _action(gold, [p if p is not None else k for p, k in zip(para, ks)])
    # flip exactly one keystroke: the last mutating one, so the rest of the batch is identical
    for i in range(len(ks) - 1, -1, -1):
        f = flip_keystroke(ks[i])
        if f is not None and f != ks[i]:
            out["flipped"] = _action(gold, ks[:i] + [f] + ks[i + 1:])
            break
    if ks:
        out["no_op"] = _action(gold, [])
    out["destructive"] = _action(gold, [f"rm -rf {shlex.quote(workdir)}/*\n"])
    if not gold.get("task_complete"):
        out["early_complete"] = _action(gold, ks, task_complete=True)
    return out


def string_distance_report(gold_text: str, ctrls: dict[str, str]) -> dict[str, float]:
    gold = json.loads(gold_text)
    return {name: round(command_similarity(gold, json.loads(t)), 3) for name, t in ctrls.items()}
