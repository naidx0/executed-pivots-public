"""Round 3 checkpoint gate (calibration docs only; no test docs, no maps).

usage: SB_SUFFIX=_r3 python3 gate_r3.py ATTEMPT

1. Runs the registered gate, statebridge2.gate(ATTEMPT), unchanged, on the target
   (ckpt_tgt{SB_SUFFIX}.pt). This is the gate of the protocol in the statebridge2.py docstring
   (commit 3afa18d).
2. Runs the identical procedure with the source substituted as the gated model (attempt label
   "src-ATTEMPT"). This is not part of the registered gate; round 3 additionally requires it so that
   the source state has document-specific content for a map to carry.
Both results go to gate{SB_SUFFIX}_attempts.json (tagged with "gated_model"), and a one-line summary
is appended to gate{SB_SUFFIX}_log.jsonl.
"""
import json
import sys
import time

attempt = int(sys.argv[1])
import statebridge2 as sb  # noqa: E402  (loads ckpt_{src,tgt}{SB_SUFFIX}.pt at import)

t0 = time.time()
path = f"gate{sb.SUFFIX}_attempts.json"
print("== registered gate: target", flush=True)
sb.gate(attempt)
print("== same gate, source as the gated model", flush=True)
tgt, tidx = sb.tgt, sb.T_IDX
sb.tgt, sb.T_IDX = sb.src, sb.S_IDX
sb.gate(f"src-{attempt}")
sb.tgt, sb.T_IDX = tgt, tidx

allr = json.load(open(path))
for a in allr:
    a.setdefault("gated_model", "src" if str(a["attempt"]).startswith("src-") else "tgt")
json.dump(allr, open(path, "w"), indent=1)


def summ(a):
    w = a["by_window"][str(sb.PRIMARY_W)]
    return {"gate_pass": a["gate_pass"],
            "full_prefill_value_ce": round(a["full"]["value_ce"], 4),
            "full_prefill_value_exact": round(a["full"]["value_exact"], 4),
            "w32_value_ce": {k: round(w[k]["value_ce"], 4) for k in ["oracle", "blank", "mismatched_own"]},
            "w32_value_exact": {k: round(w[k]["value_exact"], 4) for k in ["oracle", "blank", "mismatched_own"]},
            "w32_diff_oracle_minus_blank": [round(x, 4) for x in w["diff_oracle_minus_blank"]],
            "w32_diff_oracle_minus_mismatched_own": [round(x, 4) for x in w["diff_oracle_minus_mismatched_own"]]}


by = {str(a["attempt"]): a for a in allr}
line = {"attempt": attempt, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tgt": summ(by[str(attempt)]), "src": summ(by[f"src-{attempt}"])}
line["both_pass"] = bool(line["tgt"]["gate_pass"] and line["src"]["gate_pass"])
line["gate_runtime_s"] = round(time.time() - t0, 1)
with open(f"gate{sb.SUFFIX}_log.jsonl", "a") as f:
    f.write(json.dumps(line) + "\n")
print("GATE_SUMMARY " + json.dumps(line), flush=True)
