"""Seeded candidate generation for H16 (no model calls).

Categories, sampled per candidate with fixed weights (pre-registered):
  gold 0.15   the expert step verbatim
  equiv 0.25  equivalent rewrites (commuted operands, reordered independent lookups,
              term reorder / "x*2+7" for "2*x+7", brackets, spacing, leading zero)
  near 0.30   near-miss wrong steps (off-by-one key or literal, wrong operator,
              hard-coded lookup table, wrong-outside-the-examples programs)
  useless 0.15 steps that change nothing useful (re-reading a key, echoing a value,
              leaving the draft as it was)
  early 0.15  early answers (answer before the information is in, or before the
              calculator; for code: a constant, or a line through two examples)
Each pivot gets CANDS_PER_PIVOT candidates from distinct categories; a category
with no applicable rewrite at that pivot is re-drawn. Duplicate texts within a
pivot are dropped.
"""
from __future__ import annotations

import itertools
import random

from . import core
from .envs import lookup_program, poly_str

WEIGHTS = {"gold": 0.15, "equiv": 0.25, "near": 0.30, "useless": 0.15, "early": 0.15}
CANDS_PER_PIVOT = 2
TOOL_SEED, TOOL_PROBLEMS = 16001, 45
CODE_SEED, CODE_PROBLEMS = 16002, 75


# --------------------------------------------------------------------------- tool_use

def _turn_type(turn, p):
    if not turn.startswith("!"):
        return "answer"
    return "read" if turn[1:] in p["state"] else "calc"


def _calc_operands(turn):
    import re
    return re.findall(r"-?[0-9]+|[+*-]", turn[1:])


def tool_gen(cat, p, prefix, expert, rng):
    tt = _turn_type(expert, p)
    inv = list(dict.fromkeys(core.TOOL.involved(p)))
    events, _, _ = core.run_turns(p, list(prefix))
    known, _ = core._knowledge(p, events)
    read = [t[1:] for t, r, d in events if t[1:] in p["state"]]
    unread = [k for k in inv if k not in known]
    others = [k for k in p["state"] if k not in inv]
    opts = []
    if cat == "gold":
        return expert
    if cat == "equiv":
        if tt == "read":
            k = expert[1:]
            opts += ["!" + o for o in unread if o != k]          # reordered independent lookup
            opts += [f"! {k}", f"!{k} "]                          # spacing
        elif tt == "calc":
            body = expert[1:]
            if "*" in body:
                a, b = body.split("*")
                opts.append(f"!{b}*{a}")
            elif body.count("+") >= 1 and "-" not in body:
                parts = body.split("+")
                opts += ["!" + "+".join(q) for q in itertools.permutations(parts) if list(q) != parts]
            opts += [f"!({body})", "!" + body.replace("+", " + ").replace("-", " - ").replace("*", " * ")]
        else:
            g = int(expert)
            opts += [f" {expert}", f"{expert} "]
            if g > 0:
                opts.append("0" + expert)
    elif cat == "near":
        if tt == "read":
            k = expert[1:]
            k2 = k[0] + chr((ord(k[1]) - 97 + 1) % 26 + 97)
            k3 = chr((ord(k[0]) - 97 + 1) % 26 + 97) + k[1]
            opts += ["!" + k2, "!" + k3] + ["!" + o for o in others]
        elif tt == "calc":
            body = expert[1:]
            for op, alt in (("+", "-"), ("-", "+"), ("*", "+")):
                if op in body:
                    opts.append("!" + body.replace(op, alt, 1))
            nums = [x for x in _calc_operands(expert) if x not in "+-*"]
            n0 = nums[-1]
            opts.append("!" + body[: len(body) - len(n0)] + str(int(n0) + 1))
            if "-" in body:
                a, b = body.split("-")
                opts.append(f"!{b}-{a}")
        else:
            g = int(expert)
            opts += [str(g + 1), str(g - 1), "+" + expert if g >= 0 else expert[1:], str(g + 10)]
    elif cat == "useless":
        opts += ["!" + k for k in read]                            # re-read a key already read
        opts += [f"!{v}" for v in known.values()]                  # echo a known value through the calculator
        if not opts:
            opts.append("!" + p["prompt"])                         # call the tool with the question: "?"
    elif cat == "early":
        if tt == "answer":
            return None
        need = None
        if not unread:
            need, _ = core._need(p, known)
        if need is not None:
            opts += [str(need), str(need + 10), str(need - 1)]    # mental arithmetic: right, carry slip, off by one
        else:
            vals = list(known.values())
            opts += [str(sum(vals)) if vals else "0", str((core.TOOL.lo + core.TOOL.hi) // 2)]
            if vals:
                opts.append(str(vals[0]))
    opts = [o for o in dict.fromkeys(opts) if o != expert]
    return rng.choice(opts) if opts else None


def tool_pivots(seed=TOOL_SEED, n=TOOL_PROBLEMS):
    rng = random.Random(seed)
    probs = [core.TOOL.generate(rng) for _ in range(n)]
    out = []
    for i, p in enumerate(probs):
        for t, expert in enumerate(p["turns"]):
            out.append({"family": "tool_use", "problem": p, "pid": f"tu{seed}-{i}", "t": t,
                        "prefix": p["turns"][:t], "expert": expert})
    return out


def _sample(rng, pivot, gen, k=CANDS_PER_PIVOT):
    cats, recs, seen = list(WEIGHTS), [], set()
    used = set()
    tries = 0
    while len(recs) < k and tries < 50:
        tries += 1
        avail = [c for c in cats if c not in used]
        if not avail:
            break
        cat = rng.choices(avail, weights=[WEIGHTS[c] for c in avail])[0]
        cand = gen(cat, pivot, rng)
        used.add(cat)
        if cand is None or cand in seen:
            continue
        seen.add(cand)
        recs.append(dict(pivot, cand=cand, category=cat, source="generated", note=""))
    return recs


def tool_candidates(seed=TOOL_SEED, n=TOOL_PROBLEMS):
    rng = random.Random(seed + 1)
    out = []
    for pv in tool_pivots(seed, n):
        out += _sample(rng, pv, lambda cat, v, r: tool_gen(cat, v["problem"], v["prefix"], v["expert"], r))
    return out


# --------------------------------------------------------------------------- code_exec

def _equiv_programs(c2, c1, c0):
    g = poly_str(c2, c1, c0)
    opts = [poly_str(c2, c1, c0, order=o) for o in itertools.permutations((2, 1, 0))]
    # coefficient after the variable: "x*2+7" for "2*x+7", "x*x*2" for "2*x*x"
    opts.append(poly_str(c2, c1, c0).replace(f"{abs(c2)}*x*x", f"x*x*{abs(c2)}", 1)
                if abs(c2) > 1 else g)
    if abs(c1) > 1:
        s = poly_str(0, c1, c0).replace(f"{abs(c1)}*x", f"x*{abs(c1)}", 1)
        head = poly_str(c2, 0, 0) if c2 else ""
        opts.append(head + (s if not head or s.startswith("-") else "+" + s))
    if c2 != 0 and c1 != 0:                                   # Horner form x*(c2*x+c1)+c0
        inner = poly_str(0, c2, c1)
        h = f"x*({inner})"
        if c0:
            h += ("+" if c0 > 0 else "-") + str(abs(c0))
        opts.append(h)
    opts.append(f"({g})")
    return [o for o in dict.fromkeys(opts) if o != g and len(o) <= 40]


def _near_programs(p):
    c2, c1, c0 = p["coef"]
    g = p["gold"]
    opts = [poly_str(c2, c1, c0 + 1), poly_str(c2, c1, c0 - 1), poly_str(c2, c1 + 1, c0), poly_str(c2, c1 - 1, c0)]
    if c2:
        opts.append(poly_str(-c2, c1, c0))
    if c1:
        opts.append(poly_str(c2, -c1, c0))
    opts += [lookup_program(p["visible"]), f"x>2?0:({g})", f"{g}+x/100"]
    return [o for o in dict.fromkeys(opts) if (c2 or o != "0") and len(o) <= 40 and o != g]


def make_draft(p, rng):
    """A wrong draft program for an edit pivot: one coefficient off, a term missing, or the lookup table."""
    c2, c1, c0 = p["coef"]
    opts = [poly_str(c2, c1, c0 + rng.choice([-2, -1, 1, 2])), lookup_program(p["visible"])]
    if c2:
        opts.append(poly_str(0, c1, c0) if c1 else poly_str(0, 1, c0))
    if c1:
        opts.append(poly_str(c2, c1 + rng.choice([-1, 1]), c0))
    opts = [o for o in opts if o != "0" and core.CODE.verify(p, o) == 0.0]
    return rng.choice(opts)


def code_gen(cat, pv, rng):
    p, draft, g = pv["problem"], pv["draft"], pv["expert"]
    c2, c1, c0 = p["coef"]
    opts = []
    if cat == "gold":
        return g
    if cat == "equiv":
        opts = _equiv_programs(c2, c1, c0)
    elif cat == "near":
        opts = _near_programs(p)
    elif cat == "useless":
        if draft is not None:
            opts = [draft, draft.replace("+", " + ").replace("-", " - "), f"({draft})"]
        else:
            opts = ["x", "0", "x*x"]
    elif cat == "early":
        (x0, y0), (x1, y1) = p["visible"][0], p["visible"][1]
        opts = [str(y0), poly_str(0, y1 - y0, y0) if y1 != y0 else str(y0)]
    opts = [o for o in dict.fromkeys(opts) if o != g and len(o) <= 40]
    return rng.choice(opts) if opts else None


def code_pivots(seed=CODE_SEED, n=CODE_PROBLEMS):
    rng = random.Random(seed)
    probs = [core.CODE.generate(rng) for _ in range(n)]
    drng = random.Random(seed + 7)
    out = []
    for i, p in enumerate(probs):
        out.append({"family": "code_exec", "problem": p, "pid": f"ce{seed}-{i}-final", "pivot": "final",
                    "draft": None, "expert": p["gold"]})
        out.append({"family": "code_exec", "problem": p, "pid": f"ce{seed}-{i}-edit", "pivot": "edit",
                    "draft": make_draft(p, drng), "expert": p["gold"]})
    return out


def code_candidates(seed=CODE_SEED, n=CODE_PROBLEMS):
    rng = random.Random(seed + 1)
    out = []
    for pv in code_pivots(seed, n):
        out += _sample(rng, pv, code_gen)
    return out
