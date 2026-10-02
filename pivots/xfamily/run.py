"""H16: does executed effect (X) credit steps better than keystroke match (J) on crucible
tool_use and code_exec?

PRE-REGISTRATION (fixed in this docstring and committed before any label exists)

Hypothesis: executed effect credits steps better than keystroke match in two other
agent task families, crucible tool_use and crucible code_exec.

Definitions: see pivots/xfamily/core.py (J, X, oracle, "working") and
pivots/xfamily/candidates.py (candidate categories and weights).
  Working (tool_use): after the candidate step, the gold policy finishes within the
    kit's 5-turn budget and verify() passes, AND the gold policy's remaining turn
    count went down (a verified final answer counts as progress).
  Working (code_exec): verify(candidate program) == 1 on the visible and hidden tests.
  False credit: credit = 1 and oracle working = 0.

Win condition, per family, all bars on point estimates:
  B0  n >= 150 candidates (generated + handwritten, all labelled by the oracle)
  B1  X working credits >= 2 x J working credits, and X working credits > 0
  B2  X false credits <= J false credits
H16 is KEPT only if both families pass B0, B1 and B2; otherwise DISCARDED.
Reported with 95% pivot-clustered bootstrap CIs (10000 reps, seed 0, whole pivots
resampled): X-J and X-2J working credit, X-J false credit, agreement with the oracle.
Also reported, not used for the verdict: the same table on generated-only candidates.

Order: (1) write candidates, (2) compute J labels and seal them with a sha256,
(3) oracle labels, (4) X labels, (5) analysis. Stage 3-4 refuse to run if the seal
does not match.

Usage: python -m pivots.xfamily.run {candidates,seal,label,analyze,blind} --out DIR

The blind stage is the second-author re-check (pivots/xfamily/blind.py), pre-registered in
the H16 blind pre-registration (RESEARCH.md): it writes blind candidates and J labels, seals them, then
labels oracle, then X, then analyses, in one run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

from . import candidates as C
from . import blind, core, handwritten

FAMILIES = ("tool_use", "code_exec")
N_BOOT, BOOT_SEED = 10000, 0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def build_candidates(family):
    if family == "tool_use":
        rows = C.tool_candidates() + handwritten.tool_records()
    else:
        rows = C.code_candidates() + handwritten.code_records()
    tag = "tu" if family == "tool_use" else "ce"
    for i, r in enumerate(rows):
        r["id"] = f"h16-{tag}-{i:04d}"
    return rows


def cmd_candidates(out):
    for fam in FAMILIES:
        rows = build_candidates(fam)
        write_jsonl(os.path.join(out, f"candidates_{fam}.jsonl"), rows)
        print(fam, len(rows), dict(Counter((r["source"], r["category"]) for r in rows)))


def cmd_seal(out):
    seal = {"hypothesis": "H16", "baseline": "J = SequenceMatcher(expert, cand).ratio() >= 0.9", "files": {}}
    for fam in FAMILIES:
        cpath = os.path.join(out, f"candidates_{fam}.jsonl")
        rows = read_jsonl(cpath)
        jrows = [{"id": r["id"], "j_similarity": round(core.j_similarity(r["expert"], r["cand"]), 6),
                  "j_credit": core.label_j(r)} for r in rows]
        jpath = os.path.join(out, f"baseline_j_{fam}.jsonl")
        write_jsonl(jpath, jrows)
        seal["files"][fam] = {"candidates": os.path.basename(cpath), "candidates_sha256": sha256_file(cpath),
                              "j_labels": os.path.basename(jpath), "j_labels_sha256": sha256_file(jpath),
                              "n": len(rows), "j_credits": sum(r["j_credit"] for r in jrows)}
    with open(os.path.join(out, "baseline_j_sealed.json"), "w") as f:
        json.dump(seal, f, indent=2, sort_keys=True)
    print(json.dumps(seal, indent=2))


def _check_seal(out):
    with open(os.path.join(out, "baseline_j_sealed.json")) as f:
        seal = json.load(f)
    for fam, s in seal["files"].items():
        assert sha256_file(os.path.join(out, s["candidates"])) == s["candidates_sha256"], fam + " candidates changed"
        assert sha256_file(os.path.join(out, s["j_labels"])) == s["j_labels_sha256"], fam + " J labels changed"
    return seal


def cmd_label(out):
    _check_seal(out)
    for fam in FAMILIES:
        rows = read_jsonl(os.path.join(out, f"candidates_{fam}.jsonl"))
        oracle = [core.label_oracle(r) for r in rows]          # oracle first, then X
        xs = [core.label_x(r) for r in rows]
        labels = [{"id": r["id"], "oracle": o, "working": o["working"], "x_credit": x, "x_detail": d}
                  for r, o, (x, d) in zip(rows, oracle, xs)]
        write_jsonl(os.path.join(out, f"labels_{fam}.jsonl"), labels)
        print(fam, "working", sum(l["working"] for l in labels), "of", len(labels),
              "X credits", sum(l["x_credit"] for l in labels))


# --------------------------------------------------------------------------- analysis

def counts(rows):
    n = len(rows)
    w = [r["working"] for r in rows]
    out = {"n": n, "working": sum(w)}
    for arm in ("j", "x"):
        c = [r[arm] for r in rows]
        out[f"{arm}_credit"] = sum(c)
        out[f"{arm}_work"] = sum(a and b for a, b in zip(c, w))
        out[f"{arm}_false"] = sum(a and not b for a, b in zip(c, w))
        out[f"{arm}_agree"] = sum(a == b for a, b in zip(c, w)) / n if n else 0.0
    return out


def stats(c):
    return {"x_minus_j_work": c["x_work"] - c["j_work"], "x_minus_2j_work": c["x_work"] - 2 * c["j_work"],
            "x_minus_j_false": c["x_false"] - c["j_false"], "x_minus_j_agree": c["x_agree"] - c["j_agree"],
            "x_work": c["x_work"], "j_work": c["j_work"], "x_false": c["x_false"], "j_false": c["j_false"],
            "x_agree": c["x_agree"], "j_agree": c["j_agree"]}


def bootstrap(rows, n_boot=N_BOOT, seed=BOOT_SEED):
    by = defaultdict(list)
    for r in rows:
        by[r["pivot_key"]].append(r)
    keys = sorted(by)
    rng = random.Random(seed)
    draws = defaultdict(list)
    for _ in range(n_boot):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in by[k]]
        for k, v in stats(counts(sample)).items():
            draws[k].append(v)
    ci = {}
    for k, v in draws.items():
        v.sort()
        ci[k] = [v[int(0.025 * n_boot)], v[int(0.975 * n_boot) - 1]]
    return ci


def verdict(c):
    b0 = c["n"] >= 150
    b1 = c["x_work"] >= 2 * c["j_work"] and c["x_work"] > 0
    b2 = c["x_false"] <= c["j_false"]
    return {"B0_n>=150": b0, "B1_x_work>=2j": b1, "B2_x_false<=j_false": b2, "pass": b0 and b1 and b2}


def joined(out, fam):
    cands = {r["id"]: r for r in read_jsonl(os.path.join(out, f"candidates_{fam}.jsonl"))}
    js = {r["id"]: r for r in read_jsonl(os.path.join(out, f"baseline_j_{fam}.jsonl"))}
    rows = []
    for l in read_jsonl(os.path.join(out, f"labels_{fam}.jsonl")):
        c = cands[l["id"]]
        pk = c["pid"] + (f"-t{c['t']}" if fam == "tool_use" else "")
        rows.append({"id": l["id"], "pivot_key": pk, "source": c["source"], "category": c["category"],
                     "working": l["working"], "x": l["x_credit"], "j": js[l["id"]]["j_credit"]})
    return rows


def cmd_analyze(out):
    seal = _check_seal(out)
    res = {"seal": seal, "families": {}}
    for fam in FAMILIES:
        rows = joined(out, fam)
        c = counts(rows)
        gen = [r for r in rows if r["source"] == "generated"]
        cats = {}
        for cat in C.WEIGHTS:
            sub = [r for r in rows if r["category"] == cat]
            if sub:
                cats[cat] = counts(sub)
        res["families"][fam] = {"all": c, "verdict": verdict(c), "ci": bootstrap(rows),
                                "pivots": len({r["pivot_key"] for r in rows}),
                                "generated_only": counts(gen), "generated_only_verdict": verdict(counts(gen)),
                                "by_category": cats,
                                "handwritten": counts([r for r in rows if r["source"] == "handwritten"])}
    res["kept"] = all(res["families"][f]["verdict"]["pass"] for f in FAMILIES)
    with open(os.path.join(out, "results-h16.json"), "w") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    print(json.dumps({f: {"all": v["all"], "verdict": v["verdict"]} for f, v in res["families"].items()}, indent=1))
    print("kept", res["kept"])


# --------------------------------------------------------------------------- blind re-check

BLIND_CATEGORIES = ("equiv", "near", "useless", "early", "reorder", "gold")


def blind_verdict(c):
    w1 = c["x_work"] >= 2 * c["j_work"] and c["x_work"] > 0
    w2 = c["x_false"] <= c["j_false"]
    return {"W1_x_work>=2j": w1, "W2_x_false<=j_false": w2, "pass": w1 and w2}


def _pivot_key(r):
    return r["pid"] + (f"-t{r['t']}" if r["family"] == "tool_use" else "")


def _score(rows):
    """Oracle first, then X, on already-sealed rows. Returns analysis rows."""
    oracle = [core.label_oracle(r) for r in rows]
    xs = [core.label_x(r) for r in rows]
    out = []
    for r, o, (x, d) in zip(rows, oracle, xs):
        out.append({"id": r["id"], "pivot_key": _pivot_key(r), "category": r["category"], "note": r.get("note", ""),
                    "expert": r["expert"], "cand": r["cand"], "working": o["working"], "oracle": o,
                    "x": x, "x_detail": d, "j": core.label_j(r)})
    return out


def _summary(rows):
    c = counts(rows)
    return {"all": c, "verdict": blind_verdict(c), "ci": bootstrap(rows),
            "pivots": len({r["pivot_key"] for r in rows}),
            "by_category": {k: counts([r for r in rows if r["category"] == k])
                            for k in BLIND_CATEGORIES if any(r["category"] == k for r in rows)}}


def cmd_blind(out):
    builders = {"tool_use": blind.tool_records, "code_exec": blind.code_records}
    seal = {"hypothesis": "H16-blind", "files": {}}
    recs = {}
    for fam in FAMILIES:
        rows = builders[fam]()
        tag = "tu" if fam == "tool_use" else "ce"
        for i, r in enumerate(rows):
            r["id"] = f"h16b-{tag}-{i:04d}"
        cpath = os.path.join(out, f"blind_candidates_{fam}.jsonl")
        write_jsonl(cpath, rows)
        jpath = os.path.join(out, f"blind_j_{fam}.jsonl")
        write_jsonl(jpath, [{"id": r["id"], "j_similarity": round(core.j_similarity(r["expert"], r["cand"]), 6),
                             "j_credit": core.label_j(r)} for r in rows])
        seal["files"][fam] = {"candidates_sha256": sha256_file(cpath), "j_labels_sha256": sha256_file(jpath),
                              "n": len(rows)}
        recs[fam] = (cpath, jpath)
    with open(os.path.join(out, "blind_sealed.json"), "w") as f:
        json.dump(seal, f, indent=2, sort_keys=True)
    res = {"seal": seal, "families": {}}
    for fam in FAMILIES:
        cpath, jpath = recs[fam]
        assert sha256_file(cpath) == seal["files"][fam]["candidates_sha256"]
        assert sha256_file(jpath) == seal["files"][fam]["j_labels_sha256"]
        rows = read_jsonl(cpath)
        scored = _score(rows)
        write_jsonl(os.path.join(out, f"blind_labels_{fam}.jsonl"), scored)
        # sensitivity (not the verdict): one gold copy of the expert step per pivot
        seen, gold = set(), []
        for r in rows:
            k = _pivot_key(r)
            if k not in seen:
                seen.add(k)
                gold.append(dict(r, cand=r["expert"], category="gold", note="gold copy (sensitivity)",
                                 id=r["id"] + "-gold"))
        res["families"][fam] = dict(_summary(scored), with_gold_copies=_summary(scored + _score(gold)))
    res["supported"] = all(res["families"][f]["verdict"]["pass"] for f in FAMILIES)
    with open(os.path.join(out, "results-h16-blind.json"), "w") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    print(json.dumps({f: {"all": v["all"], "verdict": v["verdict"]} for f, v in res["families"].items()}, indent=1))
    print("supported", res["supported"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["candidates", "seal", "label", "analyze", "blind"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    {"candidates": cmd_candidates, "seal": cmd_seal, "label": cmd_label, "analyze": cmd_analyze,
     "blind": cmd_blind}[a.stage](a.out)


if __name__ == "__main__":
    sys.exit(main())
