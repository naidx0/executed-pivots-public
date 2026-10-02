#!/usr/bin/env python3
"""arloop: the bookkeeping half of the autoresearch loop. Python 3.8+, standard library only.

  python arloop.py init [--dir research]
  python arloop.py seal --rows data/sealed.jsonl                  # sha256 of the sealed set, to log once
  python arloop.py prereg --id H1 --hypothesis "..." --win "new beats base by > 2 SE on sealed, seeds 0-4"
  python arloop.py compare --base base.json --new new.json [--rule above2se|within2se] [--paired] [--lower-better]
  python arloop.py verdict --id H1 --base base.json --new new.json [--rule ...] [--note "..."]
  python arloop.py screen --cands cands/ --cmd "python score.py {cand} {seed} {steps}" [--rungs 600:0,1200:0-1,2400:0-1]
  python arloop.py novelty --cand cands/x.py --kept kept/
  python arloop.py selftest

Score files (base.json, new.json): {"seed_or_run_id": score, ...} or a JSON list of scores, or
{"runs": [{"seed": 0, "<key>": score}, ...]} with --key naming the score field.
Verdicts: above2se keeps when new - base > 2 SE; within2se keeps when new >= base - 2 SE (use it for
"as good as, but cheaper"). SE is unpaired by default: on noisy hardware the same seed does not
reproduce, so pairing by seed is not valid until a rerun shows it is.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import keyword
import math
import os
import re
import subprocess
import sys
import time
import tokenize

NL = chr(10)
DIR = os.environ.get("ARLOOP_DIR", "research")


def now():
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())


def append(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8", newline=NL) as f:
        f.write(text)


# ------------------------------------------------------------------ scores and verdicts


def load_scores(path, key=None):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict) and "runs" in d:
        k = key or next(x for x in ("sealed", "sealed_exact", "score", "value") if x in d["runs"][0])
        return {str(r.get("seed", i)): float(r[k]) for i, r in enumerate(d["runs"])}
    if isinstance(d, list):
        return {str(i): float(x) for i, x in enumerate(d)}
    return {str(k): float(v) for k, v in d.items()}


def mean(a):
    return sum(a) / len(a)


def var(a):
    m = mean(a)
    return sum((x - m) ** 2 for x in a) / (len(a) - 1) if len(a) > 1 else float("nan")


def compare(base, new, rule="above2se", paired=False, lower_better=False):
    """base, new: dict id -> score. Returns the report row the loop logs."""
    if paired:
        ids = sorted(set(base) & set(new))
        b, n = [base[i] for i in ids], [new[i] for i in ids]
        d = [y - x for x, y in zip(b, n)]
        se = math.sqrt(var(d) / len(d)) if len(d) > 1 else float("nan")
    else:
        b, n = list(base.values()), list(new.values())
        se = math.sqrt(var(b) / len(b) + var(n) / len(n)) if len(b) > 1 and len(n) > 1 else float("nan")
    delta = mean(n) - mean(b)
    gain = -delta if lower_better else delta
    if se != se:
        verdict = "NOT DECIDED"
    elif rule == "within2se":
        verdict = "KEEP" if gain >= -2 * se else "REVERT"
    elif gain > 2 * se:
        verdict = "KEEP"
    else:
        verdict = "REVERT" if gain <= 0 else "NOT DECIDED"
    return {"baseline_mean": round(mean(b), 6), "new_mean": round(mean(n), 6), "delta": round(delta, 6),
            "se": round(se, 6) if se == se else None, "t": round(gain / se, 2) if se and se == se else None,
            "n_baseline": len(b), "n_new": len(n), "paired": paired, "rule": rule,
            "direction": "lower" if lower_better else "higher", "verdict": verdict}


# ------------------------------------------------------------------ screen (successive halving)


def parse_rungs(spec):
    out = []
    for part in spec.split(","):
        steps, seeds = part.split(":")
        if "-" in seeds:
            a, b = seeds.split("-")
            s = list(range(int(a), int(b) + 1))
        else:
            s = [int(x) for x in seeds.split("+")]
        out.append((int(steps), s))
    return out


def last_float(text):
    m = re.findall(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", text)
    if not m:
        raise ValueError("scorer printed no number")
    return float(m[-1])


def screen(cands, score_fn, rungs, keep=0.5, ledger=None, lower_better=False):
    alive = list(cands)
    sign = -1 if lower_better else 1
    for ri, (steps, seeds) in enumerate(rungs):
        res = {}
        for c in alive:
            vals = []
            for s in seeds:
                try:
                    v = score_fn(c, s, steps)
                except Exception as e:
                    v = None
                    if ledger:
                        append(ledger, json.dumps({"rung": ri, "cand": c, "seed": s, "error": str(e)[:300]}) + NL)
                    break
                vals.append(v)
                if ledger:
                    append(ledger, json.dumps({"rung": ri, "steps": steps, "cand": c, "seed": s, "score": v,
                                               "time": now()}) + NL)
            if vals and v is not None:
                res[c] = mean(vals)
        ranked = sorted(res, key=lambda c: sign * res[c], reverse=True)
        if ri == len(rungs) - 1:
            return [(c, res[c]) for c in ranked]
        alive = ranked[:max(1, int(round(len(ranked) * keep)))]
    return []


# ------------------------------------------------------------------ novelty


def shingles(code, n=3):
    t = []
    try:
        for tk in tokenize.generate_tokens(io.StringIO(code).readline):
            if tk.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
                           tokenize.ENDMARKER, tokenize.STRING):
                continue
            t.append("NUM" if tk.type == tokenize.NUMBER else tk.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        t = code.split()
    return {tuple(t[i:i + n]) for i in range(max(0, len(t) - n + 1))}


def novelty(code, kept):
    """kept: name -> code. 1 = shares nothing with any kept file, 0 = a copy (comments and numbers ignored)."""
    s = shingles(code)
    best, near = 0.0, None
    for name, k in kept.items():
        o = shingles(k)
        j = len(s & o) / len(s | o) if s and o else 0.0
        if j >= best:
            best, near = j, name
    return {"novelty": round(1 - best, 4), "nearest": near, "similarity": round(best, 4)}


# ------------------------------------------------------------------ files


LOG_HEAD = "# Research log" + NL + NL + "Every hypothesis, pre-registered before it runs, then its verdict with numbers. A discard is a result." + NL
HANDOFF_HEAD = ("# Hand-off" + NL + NL + "Newest update on top. What is kept, what is discarded, what runs now (with the exact"
                " command), what is next." + NL)


def cmd_init(a):
    d = a.dir
    os.makedirs(d, exist_ok=True)
    for name, head in (("log.md", LOG_HEAD), ("handoff.md", HANDOFF_HEAD)):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            append(p, head)
    open(os.path.join(d, "ledger.jsonl"), "a").close()
    print(json.dumps({"dir": d, "files": sorted(os.listdir(d))}))


def cmd_seal(a):
    h = hashlib.sha256(open(a.rows, "rb").read()).hexdigest()[:16]
    print(json.dumps({"sealed": a.rows, "sha256_16": h}))


def cmd_prereg(a):
    text = (NL + "## %s: PRE-REGISTERED %s" % (now(), a.id) + NL + "- Hypothesis: " + a.hypothesis + NL
            + "- Win: " + a.win + NL + ("- Budget: " + a.budget + NL if a.budget else ""))
    append(os.path.join(a.dir, "log.md"), text)
    append(os.path.join(a.dir, "ledger.jsonl"), json.dumps({"id": a.id, "event": "prereg", "win": a.win, "time": now()}) + NL)
    print(text.strip())


def cmd_compare(a):
    print(json.dumps(compare(load_scores(a.base, a.key), load_scores(a.new, a.key), a.rule, a.paired, a.lower_better)))


def cmd_verdict(a):
    r = compare(load_scores(a.base, a.key), load_scores(a.new, a.key), a.rule, a.paired, a.lower_better)
    word = {"KEEP": "KEPT", "REVERT": "DISCARDED", "NOT DECIDED": "NOT DECIDED"}[r["verdict"]]
    text = (NL + "## %s: %s %s" % (now(), a.id, word) + NL
            + "- baseline %.4f (n %d), new %.4f (n %d), delta %+.4f, SE %s, t %s, rule %s, %s SE." % (
                r["baseline_mean"], r["n_baseline"], r["new_mean"], r["n_new"], r["delta"], r["se"], r["t"], r["rule"],
                "paired" if r["paired"] else "unpaired") + NL
            + ("- " + a.note + NL if a.note else ""))
    append(os.path.join(a.dir, "log.md"), text)
    append(os.path.join(a.dir, "ledger.jsonl"), json.dumps(dict(r, id=a.id, event="verdict", time=now())) + NL)
    print(text.strip())
    sys.exit(0 if r["verdict"] == "KEEP" else 1)


def cmd_screen(a):
    cands = sorted(os.path.join(a.cands, f) for f in os.listdir(a.cands) if not f.startswith((".", "_")) and not f.endswith((".md", ".json")))

    def score(c, s, steps):
        cmd = a.cmd.format(cand=chr(34) + c.replace(chr(92), "/") + chr(34), seed=s, steps=steps)
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=a.timeout)
        if p.returncode != 0:
            raise RuntimeError("exit %d: %s" % (p.returncode, p.stderr[-300:]))
        return last_float(p.stdout)
    ranked = screen(cands, score, parse_rungs(a.rungs), a.keep, os.path.join(a.dir, "ledger.jsonl"), a.lower_better)
    print(json.dumps([{"cand": c, "select": v} for c, v in ranked], indent=1))
    if not ranked:
        sys.exit("screen: no candidate survived; errors are in ledger.jsonl")


def cmd_novelty(a):
    kept = {}
    for f in sorted(os.listdir(a.kept)):
        p = os.path.join(a.kept, f)
        if os.path.isfile(p):
            kept[f] = open(p, encoding="utf-8", errors="replace").read()
    print(json.dumps(novelty(open(a.cand, encoding="utf-8", errors="replace").read(), kept)))


def cmd_selftest(a):
    fails = []

    def check(name, ok):
        if not ok:
            fails.append(name)
    big = compare({str(i): 0.2 + 0.01 * i for i in range(5)}, {str(i): 0.9 + 0.01 * i for i in range(5)})
    check("clear win keeps", big["verdict"] == "KEEP")
    same = compare({str(i): 0.5 + 0.1 * (i % 2) for i in range(5)}, {str(i): 0.5 + 0.1 * ((i + 1) % 2) for i in range(5)})
    check("noise does not keep", same["verdict"] != "KEEP")
    worse = compare({str(i): 0.8 + 0.01 * i for i in range(5)}, {str(i): 0.3 + 0.01 * i for i in range(5)})
    check("loss reverts", worse["verdict"] == "REVERT")
    low = compare({str(i): 2.4 + 0.01 * i for i in range(5)}, {str(i): 2.2 + 0.01 * i for i in range(5)}, lower_better=True)
    check("lower-better win keeps", low["verdict"] == "KEEP")
    tie = compare({str(i): 0.62 + 0.05 * (i % 3) for i in range(5)}, {str(i): 0.60 + 0.05 * (i % 3) for i in range(5)}, "within2se")
    check("within2se keeps a close tie", tie["verdict"] == "KEEP")
    table = {"a": 0.1, "b": 0.9, "c": 0.5, "d": 0.3}
    r = screen(list(table), lambda c, s, n: table[c] * n / 100, [(10, [0]), (20, [0, 1]), (40, [0])])
    check("screen ranks the best first", r and r[0][0] == "b")
    code = "def f(x):" + NL + "    return x * 2 + 1" + NL
    check("copy has novelty 0", novelty(code, {"k": code})["novelty"] == 0.0)
    check("comment edit is not novel", novelty(code.replace("2", "3") + "# note" + NL, {"k": code})["novelty"] == 0.0)
    other = "class M:" + NL + "    def g(self, y):" + NL + "        return [y for _ in range(3)]" + NL
    check("different code is novel", novelty(other, {"k": code})["novelty"] > 0.9)
    print(json.dumps({"selftest": "PASS" if not fails else "FAIL", "failed": fails}))
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser(description="autoresearch loop bookkeeping")
    ap.add_argument("--dir", default=DIR, help="research folder (log.md, handoff.md, ledger.jsonl)")
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("init")
    p = sub.add_parser("seal"); p.add_argument("--rows", required=True)
    p = sub.add_parser("prereg")
    p.add_argument("--id", required=True); p.add_argument("--hypothesis", required=True)
    p.add_argument("--win", required=True); p.add_argument("--budget", default="")
    for name in ("compare", "verdict"):
        p = sub.add_parser(name)
        p.add_argument("--base", required=True); p.add_argument("--new", required=True)
        p.add_argument("--rule", default="above2se", choices=["above2se", "within2se"])
        p.add_argument("--paired", action="store_true"); p.add_argument("--lower-better", action="store_true")
        p.add_argument("--key", default=None)
        if name == "verdict":
            p.add_argument("--id", required=True); p.add_argument("--note", default="")
    p = sub.add_parser("screen")
    p.add_argument("--cands", required=True); p.add_argument("--cmd", required=True)
    p.add_argument("--rungs", default="600:0,1200:0-1,2400:0-1"); p.add_argument("--keep", type=float, default=0.5)
    p.add_argument("--timeout", type=int, default=7200); p.add_argument("--lower-better", action="store_true")
    p = sub.add_parser("novelty"); p.add_argument("--cand", required=True); p.add_argument("--kept", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"init": cmd_init, "seal": cmd_seal, "prereg": cmd_prereg, "compare": cmd_compare, "verdict": cmd_verdict,
     "screen": cmd_screen, "novelty": cmd_novelty, "selftest": cmd_selftest}[a.action](a)


if __name__ == "__main__":
    main()
