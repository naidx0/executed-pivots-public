"""Pure-Python ports of the crucible tool_use and code_exec environments (H16).

Source (read-only kit, not edited): /mnt/project-files/auto-search/evals/crucible/
  envs/tool_use.py  sha256 a544285c7bc8ac13305380b853aa717bdc67dd885bb9e4ee2b342fdaf40220ce
  envs/code_exec.py sha256 3bc737dc605d54e7d48218a89730062d54aec44e148c75874c0ae15701f96f54
  tasks.py          sha256 21dfb4c0dcd881b4eab37e7cc235949ec11a543188f521b3f4525e90c5869374 (_eval_expr)
  envs/tinylang.py  copied verbatim as pivots/xfamily/tinylang.py

The kit modules import torch (for rollouts and GRPO) which this repo does not
need, so only the pure parts are ported: task generation, the tool semantics
(call_tool / step / script), and verify(). The code is the kit's code with the
torch-only parts removed; generate() consumes the rng in the same order, so the
same seed gives the same problems as the kit. code_exec runs programs with the
tinylang interpreter in-process (the kit's sandbox=False path); tinylang never
executes Python, so this is safe for our generated and handwritten programs.
"""
from __future__ import annotations

import ast
import random
import re
import string

from . import tinylang

_INT = re.compile(r"-?[0-9]+")
_CALC = re.compile(r"[0-9+*()-]+")
_ALLOWED = set("0123456789+-*() ")


def _eval_expr(expr: str):
    """crucible.tasks._eval_expr: safe integer + - * evaluation. (value, literals) or None."""
    if not expr or not set(expr) <= _ALLOWED or len(expr) > 40:
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    lits = []

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult)):
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return l + r
            if isinstance(n.op, ast.Sub):
                return l - r
            return l * r
        if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool):
            lits.append(n.value)
            return n.value
        raise ValueError("disallowed node")

    try:
        v = ev(tree)
    except (ValueError, RecursionError):
        return None
    return v, lits


class ToolUse:
    name = "tool_use"

    def __init__(self, n_keys=4, lo=10, hi=99, kinds=("get", "add", "sub", "mul", "max", "add3"), max_turns=5):
        self.n_keys, self.lo, self.hi = n_keys, lo, hi
        self.kinds, self.max_turns = tuple(kinds), max_turns

    def generate(self, rng: random.Random):
        keys = set()
        while len(keys) < self.n_keys:
            keys.add(rng.choice(string.ascii_lowercase) + rng.choice(string.ascii_lowercase))
        keys = sorted(keys)
        rng.shuffle(keys)
        state = {k: rng.randint(self.lo, self.hi) for k in keys}
        kind = rng.choice(self.kinds)
        a, b, c = keys[0], keys[1], keys[2 % len(keys)]
        va, vb, vc = state[a], state[b], state[c]
        if kind == "get":
            prompt, gold, turns = f"{a}?", va, [f"!{a}", str(va)]
        elif kind == "add":
            prompt, gold = f"{a}+{b}?", va + vb
            turns = [f"!{a}", f"!{b}", f"!{va}+{vb}", str(va + vb)]
        elif kind == "sub":
            prompt, gold = f"{a}-{b}?", va - vb
            turns = [f"!{a}", f"!{b}", f"!{va}-{vb}", str(va - vb)]
        elif kind == "max":
            prompt, gold = f"{a}>{b}?", max(va, vb)
            turns = [f"!{a}", f"!{b}", str(max(va, vb))]
        elif kind == "add3":
            prompt, gold = f"{a}+{b}+{c}?", va + vb + vc
            turns = [f"!{a}", f"!{b}", f"!{c}", f"!{va}+{vb}+{vc}", str(gold)]
        else:
            n = rng.randint(2, 9)
            prompt, gold = f"{a}*{n}?", va * n
            turns = [f"!{a}", f"!{va}*{n}", str(va * n)]
        p = {"prompt": prompt, "gold": str(gold), "kind": kind, "state": state, "turns": turns}
        p["trace"] = self.script(p, turns)["transcript"]
        return p

    def involved(self, p):
        return re.findall("[a-z][a-z]", p["prompt"])

    def call_tool(self, p, arg: str) -> str:
        arg = arg.strip()
        if arg in p["state"]:
            return str(p["state"][arg])
        if arg and _CALC.fullmatch(arg):
            r = _eval_expr(arg)
            if r is not None and abs(r[0]) < 10 ** 9:
                return str(r[0])
        return "?"

    def step(self, p, context: str, turn_text: str, last_turn: bool):
        """Apply one model turn. Returns (new_context, transcript_piece, done)."""
        if turn_text.startswith("!") and not last_turn:
            piece = turn_text + "=" + self.call_tool(p, turn_text[1:]) + ";"
            return context + piece, piece, False
        return context + turn_text, turn_text, True

    def script(self, p, turns):
        ctx, transcript, pairs = p["prompt"], "", []
        for t, text in enumerate(turns[: self.max_turns]):
            pairs.append((ctx, text))
            ctx, piece, done = self.step(p, ctx, text, t == self.max_turns - 1)
            transcript += piece
            if done:
                break
        return {"transcript": transcript, "pairs": pairs}

    def final_answer(self, out: str) -> str:
        return out.rsplit(";", 1)[-1].strip()

    def verify(self, p, out: str) -> float:
        fa = self.final_answer(out)
        if not _INT.fullmatch(fa):
            return 0.0
        return float(int(fa) == int(p["gold"]))


def poly_str(c2, c1, c0, order=(2, 1, 0)):
    """A short program for c2*x*x + c1*x + c0 with its terms in the given order."""
    parts = []
    for deg in order:
        c = (c2, c1, c0)[2 - deg]
        if c == 0:
            continue
        mono = {2: "x*x", 1: "x", 0: ""}[deg]
        mag = abs(c)
        if deg == 0:
            body = str(mag)
        else:
            body = mono if mag == 1 else f"{mag}*{mono}"
        sign = "-" if c < 0 else "+"
        parts.append((sign, body))
    if not parts:
        return "0"
    s = ("-" if parts[0][0] == "-" else "") + parts[0][1]
    for sign, body in parts[1:]:
        s += sign + body
    return s


def lookup_program(pairs):
    """Hard-codes the examples: x=a?ya:x=b?yb:yc. Right on those points only."""
    s = str(pairs[-1][1])
    for x, y in reversed(pairs[:-1]):
        s = f"x={x}?{y}:" + s
    return s


def run_program(prog: str, xs):
    """Outputs on xs, or 'error: ...' (the kit's sandbox=False path)."""
    r = tinylang.run_program(prog, xs)
    return r if isinstance(r, list) else "error: " + r[1]


class CodeExec:
    name = "code_exec"

    def __init__(self, visible=(0, 1, 2), test_lo=-3, test_hi=9, c2_choices=(-1, 0, 1, 2), c1_max=9, c0_max=20):
        self.visible = list(visible)
        self.test_lo, self.test_hi = test_lo, test_hi
        self.c2_choices, self.c1_max, self.c0_max = tuple(c2_choices), c1_max, c0_max

    def generate(self, rng: random.Random):
        while True:
            c2 = rng.choice(self.c2_choices)
            c1 = rng.randint(-self.c1_max, self.c1_max)
            c0 = rng.randint(-self.c0_max, self.c0_max)
            if c2 or c1:
                break
        vis = list(self.visible)

        def f(x):
            return c2 * x * x + c1 * x + c0

        hidden = [x for x in range(self.test_lo, self.test_hi + 1) if x not in vis]
        prompt = ";".join(f"{x}>{f(x)}" for x in vis) + "="
        return {"coef": [c2, c1, c0], "visible": [[x, f(x)] for x in vis],
                "hidden": [[x, f(x)] for x in hidden], "prompt": prompt, "gold": poly_str(c2, c1, c0)}

    def _passes(self, p, out: str, pairs) -> float:
        prog = out.strip()
        if not prog:
            return 0.0
        got = run_program(prog, [x for x, _ in pairs])
        return float(isinstance(got, list) and got == [y for _, y in pairs])

    def verify(self, p, out: str) -> float:
        return self._passes(p, out, p["visible"] + p["hidden"])
