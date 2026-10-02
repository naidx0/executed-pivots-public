# Copied verbatim from /mnt/project-files/auto-search/evals/crucible/envs/tinylang.py (read-only kit), 2026-09-24.
# Only this header line pair was added; the interpreter below is unchanged.
"""A tiny expression language over one integer input x, and its sandbox worker.

Model-written programs are NEVER executed as Python. They are parsed and
interpreted by the code in this file, which knows exactly these constructs:

  expr := cmp [ "?" expr ":" expr ]          conditional (right associative)
  cmp  := sum [ (">" | "=") sum ]            comparison, yields 1 or 0
  sum  := term { ("+" | "-") term }
  term := unary { ("*" | "/") unary }        "/" is floor division; x/0 is an error
  unary:= "-" unary | atom
  atom := integer | "x" | "(" expr ")"

Limits enforced by the interpreter itself: program length, literal size, nesting
depth, node count (fuel) and the magnitude of every intermediate value. There are
no loops, names, calls or strings, so a program cannot reach anything outside
its own arithmetic.

Defense in depth: this file is also the sandbox worker. `python -I -S tinylang.py`
(isolated mode, no site-packages) reads JSON jobs on stdin and answers on stdout.
The parent (code_exec.Sandbox) starts it with CPU, memory, file-size and
open-file limits and kills it on a wall-clock timeout. After start-up the worker
deletes the dangerous builtins (open, eval, exec, compile, __import__, ...), so
even a bug in this interpreter would not hand a program the Python runtime.

Standard library only; this module must not import torch or the crucible package,
because the worker runs it as a plain script.
"""
from __future__ import annotations

import json
import sys

MAX_LEN = 40
MAX_LITERAL_DIGITS = 4
MAX_PARENS = 8
MAX_DEPTH = 120  # parser call depth; paren nesting and program length bound it well below this
MAX_NODES = 64
MAX_ABS = 10 ** 9
ALPHABET = set("0123456789x+-*/()?:>= ")
DIGITS = set("0123456789")


class ProgramError(Exception):
    pass


def tokenize(src: str):
    if not isinstance(src, str):
        raise ProgramError("not a string")
    if len(src) > MAX_LEN:
        raise ProgramError("program too long")
    bad = set(src) - ALPHABET
    if bad:
        raise ProgramError("illegal characters")
    toks, i = [], 0
    while i < len(src):
        c = src[i]
        if c == " ":
            i += 1
            continue
        if c in DIGITS:
            j = i
            while j < len(src) and src[j] in DIGITS:
                j += 1
            if j - i > MAX_LITERAL_DIGITS:
                raise ProgramError("literal too large")
            toks.append(("n", int(src[i:j])))
            i = j
            continue
        toks.append(("o", c))
        i += 1
    if not toks:
        raise ProgramError("empty program")
    depth = 0
    for kind, v in toks:
        depth += (v == "(") - (v == ")") if kind == "o" else 0
        if depth > MAX_PARENS:
            raise ProgramError("nesting too deep")
    return toks


class _Parser:
    def __init__(self, toks):
        self.t, self.i, self.nodes = toks, 0, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else ("e", None)

    def eat(self, v):
        if self.peek() != ("o", v):
            raise ProgramError("expected " + v)
        self.i += 1

    def node(self, n, depth):
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise ProgramError("program too large")
        return n

    def expr(self, d):
        if d > MAX_DEPTH:
            raise ProgramError("nesting too deep")
        c = self.cmp(d + 1)
        if self.peek() == ("o", "?"):
            self.i += 1
            a = self.expr(d + 1)
            self.eat(":")
            b = self.expr(d + 1)
            return self.node(("if", c, a, b), d)
        return c

    def cmp(self, d):
        a = self.sum(d + 1)
        k = self.peek()
        if k in (("o", ">"), ("o", "=")):
            self.i += 1
            b = self.sum(d + 1)
            return self.node((k[1], a, b), d)
        return a

    def sum(self, d):
        a = self.term(d + 1)
        while self.peek() in (("o", "+"), ("o", "-")):
            op = self.peek()[1]
            self.i += 1
            a = self.node((op, a, self.term(d + 1)), d)
        return a

    def term(self, d):
        a = self.unary(d + 1)
        while self.peek() in (("o", "*"), ("o", "/")):
            op = self.peek()[1]
            self.i += 1
            a = self.node((op, a, self.unary(d + 1)), d)
        return a

    def unary(self, d):
        if d > MAX_DEPTH:
            raise ProgramError("nesting too deep")
        if self.peek() == ("o", "-"):
            self.i += 1
            return self.node(("neg", self.unary(d + 1)), d)
        return self.atom(d + 1)

    def atom(self, d):
        k = self.peek()
        if k[0] == "n":
            self.i += 1
            return self.node(("lit", k[1]), d)
        if k == ("o", "x"):
            self.i += 1
            return self.node(("x",), d)
        if k == ("o", "("):
            self.i += 1
            e = self.expr(d + 1)
            self.eat(")")
            return e
        raise ProgramError("unexpected token")


def parse(src: str):
    p = _Parser(tokenize(src))
    tree = p.expr(0)
    if p.i != len(p.t):
        raise ProgramError("trailing tokens")
    return tree


def _chk(v):
    if v > MAX_ABS or v < -MAX_ABS:
        raise ProgramError("value out of range")
    return v


def evaluate(tree, x: int) -> int:
    k = tree[0]
    if k == "lit":
        return tree[1]
    if k == "x":
        return x
    if k == "neg":
        return _chk(-evaluate(tree[1], x))
    if k == "if":
        return evaluate(tree[2], x) if evaluate(tree[1], x) != 0 else evaluate(tree[3], x)
    a, b = evaluate(tree[1], x), evaluate(tree[2], x)
    if k == "+":
        return _chk(a + b)
    if k == "-":
        return _chk(a - b)
    if k == "*":
        return _chk(a * b)
    if k == "/":
        if b == 0:
            raise ProgramError("division by zero")
        return _chk(a // b)
    if k == ">":
        return int(a > b)
    if k == "=":
        return int(a == b)
    raise ProgramError("bad node")


def run_program(src: str, inputs):
    """Outputs of the program on each input, or ("error", reason). Never raises."""
    try:
        tree = parse(src)
        return [evaluate(tree, int(x)) for x in inputs]
    except ProgramError as e:
        return ("error", str(e))
    except (RecursionError, ValueError, TypeError, OverflowError):
        return ("error", "interpreter fault")


# --------------------------------------------------------------------------- sandbox worker

_UNSAFE_BUILTINS = ("open", "eval", "exec", "compile", "__import__", "input", "breakpoint",
                    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview", "help")


def _harden():
    """Remove the builtins that reach files, code objects or imports. Called after json is loaded."""
    import builtins
    has, remove = hasattr, delattr
    for name in _UNSAFE_BUILTINS:
        if has(builtins, name):
            remove(builtins, name)


def worker_main():
    """Line protocol: {"jobs": [[program, [inputs...]], ...]} -> {"res": [outputs | "error: ..."]}."""
    rd, wr = sys.stdin.readline, sys.stdout
    loads, dumps = json.loads, json.dumps
    _harden()
    wr.write(dumps({"ready": True}) + chr(10))
    wr.flush()
    while True:
        line = rd()
        if not line:
            return
        try:
            jobs = loads(line)["jobs"]
            res = []
            for prog, inputs in jobs:
                r = run_program(prog, inputs[:64])
                res.append(r if isinstance(r, list) else "error: " + r[1])
            out = {"res": res}
        except Exception as e:  # malformed request: answer, never die
            out = {"error": type(e).__name__}
        wr.write(dumps(out) + chr(10))
        wr.flush()


if __name__ == "__main__" and "--sandbox-worker" in sys.argv:
    sys.setrecursionlimit(200)
    worker_main()
