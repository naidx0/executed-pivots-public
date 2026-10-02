"""Label policy rollouts with E and P, and score H31: does X agree with E at least as well as J does?

Input is a rollouts file from either driver: `gym eval run` (Gym's rollouts.jsonl: uuid, reward, j_reward,
response.output) or pivots.gym.rollouts (the same verify fields plus the text). Per rollout:

  X  the rollout's own reward, what executed_pivot's /verify paid (effect equivalence, threshold 0.75)
  J  the rollout's j_reward, what terminus_judge_string_only would have paid for the same text
  E  executed outcome (pivots.audit.run.executed_outcome): fork the pivot, run the action, splice the expert's
     remaining turns, run the task's verifier. E = 1 when the task then passes
  P  verified progress at horizon 0 (pivots.audit.progress): at least as many verifier checks pass right after
     the action as after the expert's step, and a task_complete claim is true

Masked rollouts (environment failures and policy_error) are counted but not labelled: nothing was scored.
Identical texts at one pivot are executed once.

    python -m pivots.gym.labels --rollouts out/gym/run/rollouts.jsonl --out out/gym/run --store /tmp/gym-labels

writes labels.jsonl (one line per labelled rollout), summary.json and summary.md next to --out.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MIN_N = 100  # H31 is inconclusive below this many labelled rollouts


def rollout_text(r: dict) -> str | None:
    """The policy's action text: `text` (direct driver) or the assistant message in Gym's response.output."""
    if "text" in r:
        return r["text"]
    texts = []
    for o in (r.get("response") or {}).get("output") or []:
        if o.get("type") == "message" and o.get("role") == "assistant":
            c = o.get("content")
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, list):
                texts += [x.get("text") for x in c if isinstance(x, dict) and isinstance(x.get("text"), str)]
    text = "\n".join(texts).strip()
    return text.split("</think>")[-1].strip() if "</think>" in text else text


def load_rollouts(path: str | Path) -> list[dict]:
    out = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        if not line.strip():
            continue
        r = json.loads(line)
        task, _, turn = str(r["uuid"]).rpartition(":")
        usage = r.get("usage") or ((r.get("response") or {}).get("usage") or {})
        out.append({"i": i, "uuid": r["uuid"], "task": task, "turn": int(turn), "text": rollout_text(r),
                    "X": int(float(r.get("reward") or 0) >= 1), "J": None if r.get("j_reward") is None else int(r["j_reward"]),
                    "score": r.get("score"), "masked": bool(r.get("mask_sample")), "failure_kind": r.get("failure_kind"),
                    "verify_s": r.get("verify_s"), "gen_s": r.get("gen_wall_s"),
                    "source": (r.get("policy") or {}).get("source"),
                    "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
                    "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens"))})
    return out


def specimen_dirs(root: str | Path) -> dict[str, Path]:
    out = {}
    for p in sorted(Path(root).glob("*/task.json")):
        out[json.loads(p.read_text(encoding="utf-8"))["name"]] = p.parent
    return out


def label_specimen(sp_dir: Path, items: list[dict], store: str, workers: int = 4) -> dict[tuple[int, str], dict]:
    """{(turn, text): {E, P, harm, ...}} for the distinct (turn, text) pairs of one specimen."""
    from cleave.world.overlay import OverlayWorld
    from pivots import specimen as S
    from pivots.audit.progress import after_action
    from pivots.audit.run import executed_outcome

    sp = S.load(sp_dir)
    w = S.build(functools.partial(OverlayWorld, store=store), sp)
    rr = S.run_expert(w, sp)
    turns = sorted({t for t, _ in items})
    before = {t: S.verify(w, rr.anchor(t), sp)["checks_passed"] for t in turns}
    expert = {t: S.verify(w, rr.steps[t].checkpoint, sp)["checks_passed"] for t in turns}

    def one(key):
        t, text = key
        e = executed_outcome(w, sp, rr, t, text)
        a = after_action(w, sp, rr.anchor(t), text)
        ok = a["checks"] is not None and a["checks"] >= expert[t] and not (a["complete"] and a["reward"] != 1)
        return key, {"E": int(e["E"]), "P": int(ok), "harm": int(a["checks"] is not None and a["checks"] < before[t]),
                     "checks_after": a["checks"], "checks_expert": expert[t], "checks_before": before[t],
                     "E_why": e.get("why"), "ended_early": e.get("ended_early")}

    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, sorted(set(items))))


def binom_two_sided(b: int, c: int) -> float | None:
    """Exact two-sided sign test on the discordant pairs (McNemar); None with no discordant pair."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(labelled: list[dict], all_rollouts: list[dict], min_n: int = MIN_N) -> dict:
    n = len(labelled)
    rs = [r for r in labelled if r["J"] is not None]

    def vs(lab: str, truth: str, rows: list[dict] | None = None) -> dict:
        rows = rs if rows is None else rows
        cells = Counter((r[lab], r[truth]) for r in rows)
        return {"agree": sum(v for (a, b), v in cells.items() if a == b), "n": len(rows),
                "credit_working": cells[(1, 1)], "false_credit": cells[(1, 0)], "missed_working": cells[(0, 1)],
                "rate": round(sum(v for (a, b), v in cells.items() if a == b) / len(rows), 4) if rows else None}

    xe, je = vs("X", "E"), vs("J", "E")
    parsed = [r for r in rs if r.get("failure_kind") != "executed_pivot:model_output_invalid"]
    b = sum(r["X"] == r["E"] and r["J"] != r["E"] for r in rs)  # only X right
    c = sum(r["J"] == r["E"] and r["X"] != r["E"] for r in rs)  # only J right
    if len(rs) < min_n:
        verdict = f"inconclusive: {len(rs)} labelled rollouts < {min_n}"
    elif xe["agree"] >= je["agree"] and xe["false_credit"] == 0:
        verdict = "win"
    else:
        why = []
        if xe["agree"] < je["agree"]:
            why.append(f"X agrees with E on {xe['agree']} < J's {je['agree']}")
        if xe["false_credit"]:
            why.append(f"X false credits {xe['false_credit']}")
        verdict = "loss: " + "; ".join(why)
    gen = [r["gen_s"] for r in all_rollouts if r.get("gen_s") is not None]
    ver = [r["verify_s"] for r in all_rollouts if r.get("verify_s") is not None]
    tok_in = [r["prompt_tokens"] for r in all_rollouts if r.get("prompt_tokens") is not None]
    tok_out = [r["completion_tokens"] for r in all_rollouts if r.get("completion_tokens") is not None]
    per_pivot = defaultdict(int)
    for r in all_rollouts:
        per_pivot[r["uuid"]] += 1
    return {
        "rollouts": len(all_rollouts), "pivots": len(per_pivot), "labelled": n,
        "masked": dict(Counter(r["failure_kind"] for r in all_rollouts if r["masked"])),
        "action_source": dict(Counter(r["source"] for r in all_rollouts if r.get("source"))),
        "failure_kinds": dict(Counter(r["failure_kind"] for r in labelled if r["failure_kind"])),
        "E1": sum(r["E"] for r in labelled), "P1": sum(r["P"] for r in labelled), "X1": sum(r["X"] for r in labelled),
        "J1": sum(r["J"] or 0 for r in labelled),
        "X_vs_E": xe, "J_vs_E": je, "X_vs_P": vs("X", "P"), "J_vs_P": vs("J", "P"),
        "paired": {"only_X_right": b, "only_J_right": c, "sign_test_p": binom_two_sided(b, c)},
        # Unparseable replies score X = J = E = 0 and agree trivially; the same tables without them.
        "parsed": {"n": len(parsed), "X_vs_E": vs("X", "E", parsed), "J_vs_E": vs("J", "E", parsed)},
        "cost": {"prompt_tokens": sum(tok_in), "completion_tokens": sum(tok_out),
                 "completion_tokens_median": statistics.median(tok_out) if tok_out else None,
                 "gen_s_total": round(sum(gen), 1), "gen_s_median": statistics.median(gen) if gen else None,
                 "verify_s_median": statistics.median(ver) if ver else None},
        "h31": {"min_n": min_n, "verdict": verdict,
                "rule": "win iff labelled >= min_n, X agrees with E on at least as many rollouts as J does "
                        "(paired, same rollouts), and X false credits (X=1, E=0) are 0"},
    }


def proxy_summary(path: str | Path) -> dict:
    """What the local-policy proxy logged (one line per policy request): the Gym driver's only record of where
    each action came from, and of generation time and tokens when Gym's rollouts do not carry usage."""
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    ok = [r for r in rows if r.get("status") == "ok"]
    gen = [r["gen_s"] for r in ok if r.get("gen_s") is not None]
    usage = [r.get("usage") or {} for r in ok]
    return {"requests": len(rows), "upstream_errors": len(rows) - len(ok),
            "action_source": dict(Counter(r.get("source") for r in ok)),
            "retried": sum((r.get("attempts") or 1) > 1 for r in ok),
            "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
            "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
            "gen_s_total": round(sum(gen), 1), "gen_s_median": statistics.median(gen) if gen else None}


def summary_md(s: dict) -> str:
    xe, je = s["X_vs_E"], s["J_vs_E"]
    lines = [
        f"# H31 rollouts: {s['h31']['verdict']}", "",
        f"- rollouts {s['rollouts']} over {s['pivots']} pivots; labelled {s['labelled']}; masked {s['masked'] or 0}",
        f"- action source (proxy): {s['action_source'] or 'n/a (Gym driver: see the proxy log)'}",
        f"- labels: X=1 {s['X1']}, J=1 {s['J1']}, E=1 {s['E1']}, P=1 {s['P1']}", "",
        "| reward | agrees with E | credit on E=1 | false credit (E=0) | missed E=1 |", "|---|---|---|---|---|",
        f"| X | {xe['agree']}/{xe['n']} | {xe['credit_working']} | {xe['false_credit']} | {xe['missed_working']} |",
        f"| J | {je['agree']}/{je['n']} | {je['credit_working']} | {je['false_credit']} | {je['missed_working']} |", "",
        f"- paired: only X right {s['paired']['only_X_right']}, only J right {s['paired']['only_J_right']}, "
        f"sign test p {s['paired']['sign_test_p']}",
        f"- parsed replies only ({s['parsed']['n']}): X agrees with E {s['parsed']['X_vs_E']['agree']}, "
        f"J {s['parsed']['J_vs_E']['agree']}",
        f"- vs P: X agrees {s['X_vs_P']['agree']}/{s['X_vs_P']['n']} (false credit {s['X_vs_P']['false_credit']}), "
        f"J agrees {s['J_vs_P']['agree']}/{s['J_vs_P']['n']} (false credit {s['J_vs_P']['false_credit']})",
        f"- cost: {s['cost']['prompt_tokens']} prompt + {s['cost']['completion_tokens']} completion tokens; "
        f"generation {s['cost']['gen_s_total']} s total (median {s['cost']['gen_s_median']} s); "
        f"median verify {s['cost']['verify_s_median']} s",
        f"- rule: {s['h31']['rule']}",
    ]
    if s.get("proxy"):
        p = s["proxy"]
        lines.insert(5, f"- proxy log: {p['requests']} requests, {p['upstream_errors']} upstream errors, "
                        f"{p['retried']} retried; action source {p['action_source']}; {p['prompt_tokens']} prompt + "
                        f"{p['completion_tokens']} completion tokens; generation {p['gen_s_total']} s "
                        f"(median {p['gen_s_median']} s)")
    return "\n".join(lines) + "\n"


def run(rollouts_path: str, out: str, store: str, specimens: str = str(REPO_ROOT / "specimens"),
        workers: int = 4, min_n: int = MIN_N, labeler=label_specimen, policy_log: str | None = None) -> dict:
    rollouts = load_rollouts(rollouts_path)
    todo = [r for r in rollouts if not r["masked"] and r["text"] is not None]
    dirs = specimen_dirs(specimens)
    missing = sorted({r["task"] for r in todo} - set(dirs))
    if missing:
        raise SystemExit(f"rollouts name tasks with no specimen under {specimens}: {missing}")
    by_task = defaultdict(list)
    for r in todo:
        by_task[r["task"]].append((r["turn"], r["text"]))
    t0 = time.time()
    labels = {}
    for task, items in sorted(by_task.items()):
        for (t, text), lab in labeler(dirs[task], items, f"{store}-{task}", workers).items():
            labels[(task, t, text)] = lab
        print(f"{task}: {len(set(items))} distinct actions labelled ({time.time() - t0:.0f}s)", flush=True)
    labelled = [{**r, **labels[(r["task"], r["turn"], r["text"])]} for r in todo]
    os.makedirs(out, exist_ok=True)
    with open(Path(out, "labels.jsonl"), "w", encoding="utf-8") as fh:
        for r in labelled:
            fh.write(json.dumps(r) + "\n")
    s = summarize(labelled, rollouts, min_n)
    s["label_s"] = round(time.time() - t0, 1)
    if policy_log and os.path.exists(policy_log):
        s["proxy"] = proxy_summary(policy_log)
    Path(out, "summary.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    Path(out, "summary.md").write_text(summary_md(s), encoding="utf-8")
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default=f"/tmp/gym-labels-{os.getpid()}", help="overlay store prefix (per task)")
    ap.add_argument("--specimens", default=str(REPO_ROOT / "specimens"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--policy-log", help="the local-policy proxy's --log (source, tokens, time per request)")
    a = ap.parse_args(argv)
    s = run(a.rollouts, a.out, a.store, a.specimens, a.workers, a.min_n, policy_log=a.policy_log)
    print(summary_md(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
