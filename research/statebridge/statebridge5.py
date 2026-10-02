"""StateBridge round 5 (last in this series): does an ALL-LAYERS bridge, or LONGER training of the
round-4 paired bridge, close the gap to the source model alone?

ROUND 5 PROTOCOL (fixed and committed before any round-5 training step or test-document run)

Background (round 4, w = 32, 96 test docs): the dense trained bridge reached 3.27 nats/byte value
CE / 13.9% per-byte acc; ridge 5.07 / 5.7%; oracle 1.36 / 59.1%; source alone 1.41 / 57.8%.
It beat blank and a different document's state, lost to source alone. The source re-reading from
its own SSM state scores 57.7%, so the bindings are in the source state. Both round-4 bridges hit
their wall-clock cap with val CE still falling. Two untested explanations: (1) each target layer
reads only one paired source layer; (2) training was too short. Round 5 tests both.

Unchanged from round 4: frozen models ckpt_{src,tgt}_r3b.pt; task T = 864, C = 150, 6 bindings;
the 96 test docs (test split, seed 2); windows {1, 8, 32, 128, 256}, primary w = 32; handoff =
SSM state only, conv tail zero, attention KV empty; the target re-reads [T-w, T+C-1) and is scored
on [T, T+C). Calibration data = round 4's prep file states_r4.pt, reused as-is (calibration split
only: ridge fit on 432 calib docs; 2048 train docs seed 41; 256 val docs seed 42; precomputed
source states). Same standardisation (fixed), same loss through the frozen target:
    loss = mean CE on the 24 recalled value bytes + mean CE on all bytes after the handoff,
window cycling through {1, 8, 32, 128, 256}, batch 32.

Bridge variants
(a) all_layers (PRIMARY for question 1): each target SSM layer j is predicted from the
    concatenation of ALL 4 source SSM layers' standardised states z_all (16384 dims):
        y_j = ( z_pair(j) W_j + b_j + (z_all V) U_j ) * sy_j + my_j
    = round 4's paired dense map plus a shared rank-256 bottleneck over every source layer.
    Init exactly as round 4 (W_j = ridge solution in standardised target units, b_j = 0) plus
    U_j = 0 (so step 0 IS the ridge map) and V ~ N(0, 1/16384) (torch seed 0).
    Params: 201.4M (W, b) + 4.2M (V) + 12.6M (U) = 218.2M (1.08x round 4's 201.4M).
    Adam: W, b at lr 5e-5 (round 4's); V, U at lr 1e-3 (round 4's MLP-variant lr for zero-init
    residual parts). Wall-clock cap 18 min (round 4 dense: 8.5 min).
(b) dense_long (question 2): round 4's paired dense bridge trained further. Warm start from round
    4's selected checkpoint (ckpt_bridge_dense_r4.pt = step 400 of 417, 8.5 min, chosen on val),
    fresh Adam state, same lr 5e-5, same data, data-order seed 1 (round 4 used 0). Cap 19 more
    minutes -> about 27.5 min cumulative = 3.2x round 4's training time.

Early stopping (calibration val slice only, as round 4): every 40 steps, and at step 0, mean
recalled-value CE at w = 32 on the 256 val docs; keep the best checkpoint (step 0 included);
stop after 6 evaluations without improvement or at the cap. Mismatched (different-doc) val CE is
logged as a diagnostic. Same pre-declared fallback: if the first eval after step 0 is worse than
step 0 by > 0.5 nats/byte, restart once from the init with all lrs / 4.
Operational only: a Bash call may not exceed 10 min, so training runs in foreground segments of
at most 9 min that save and resume the full training state (weights, Adam state, data order,
step, evals, best checkpoint, elapsed training time). The cap counts training+eval time only.

Test arms (per w; 96 test docs, read once, in one final pass after both trainings finish)
- all_layers + w, all_layers_mismatched + w (the SAME trained bridge fed a different test doc's
  source state: next doc in the batch, as rounds 3-4);
- dense_long + w, dense_long_mismatched + w;
- trained_r4 + w, trained_r4_mismatched + w (round 4's checkpoint, reference);
- blank + w; oracle (target's own true SSM state) + w; source alone; full target prefill.

Metrics (as round 4): per doc, mean CE on the 24 recalled value bytes (primary), per-byte argmax
accuracy, exact-value accuracy, CE on all C bytes. Gap closed = (blank - arm) / (blank - full).
Paired bootstrap over test docs (10k resamples, 95% percentile CIs) of arm differences for value
CE, value acc and all-bytes CE.

HEADLINE (w = 32, value CE): for EACH variant (all_layers, dense_long) separately, does it beat all
three controls: (a) blank + w, (b) its own mismatched arm, (c) source alone -- each paired
difference's 95% CI entirely below 0? Also reported per variant:
- oracle share of accuracy = acc(variant) / acc(oracle), and blank-corrected
  (acc - acc_blank) / (acc_oracle - acc_blank);
- document-specific part of the gain: matched - mismatched (value CE and acc, with CIs), and the
  fraction of the CE gain over blank that is document-specific =
  (mismatched - matched) / (blank - matched);
- paired differences all_layers - trained_r4, dense_long - trained_r4, all_layers - dense_long
  (the last compares bridges with different training time; read with that in mind).
Same numbers at every window.

Stages: python statebridge5.py train {all_layers|dense_long}  (repeat until it prints
TRAINING_DONE) | python statebridge5.py test.
SB5_SMOKE=1: tiny caps and segments, val subset, test docs replaced by calib-split docs (seed 99),
all outputs to the scratch dir; never reads test documents.
"""
import json
import os
import resource
import sys
import time

import numpy as np
import torch

import statebridge4 as sb4
from recall import eval_docs, streams

sb2 = sb4.sb2
torch.set_num_threads(int(os.environ.get("SB_THREADS", "4")))
SMOKE = bool(os.environ.get("SB5_SMOKE"))
SCR = "/tmp/claude-0/-home-claude/85fced09-8eb7-5482-be2c-9bc85e817f60/scratchpad"
OUT = SCR if SMOKE else "/mnt/project-files/nebius/statebridge/round5"
WD = SCR if SMOKE else "."
TAG = "_r5smoke" if SMOKE else "_r5"
T, C, NB, WINDOWS, PW = sb2.T, sb2.C, sb2.NB, sb2.WINDOWS, sb2.PRIMARY_W
S_IDX, T_IDX, PAIR, NT = sb2.S_IDX, sb2.T_IDX, sb2.PAIR, len(sb2.T_IDX)
src, tgt = sb2.src, sb2.tgt
BATCH, VAL_EVERY, PATIENCE, RANK = 32, 40, 6, 256
CFG = {"all_layers": dict(lr=5e-5, lr_lowrank=1e-3, minutes=18.0, seed=0),
       "dense_long": dict(lr=5e-5, minutes=19.0, seed=1, warm="ckpt_bridge_dense_r4.pt")}
SEG_MIN = 9.0
N_TEST = 96
if SMOKE:
    for c in CFG.values():
        c["minutes"] = 0.6
    SEG_MIN, VAL_EVERY, N_TEST = 0.3, 5, 16


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


class AllLayers(torch.nn.Module):
    """Paired dense map (init ridge) + shared rank-RANK bottleneck over all source layers (U = 0)."""

    def __init__(self, ridge):
        super().__init__()
        base = sb4.Bridge(ridge, "dense")
        self.mx, self.sx, self.sy, self.my = base.mx, base.sx, base.sy, base.my
        self.W, self.b = base.W, base.b
        dx = sum(m.numel() for m in self.mx)
        dy = self.W[0].shape[1]
        g = torch.Generator().manual_seed(0)
        self.V = torch.nn.Parameter(torch.randn(dx, RANK, generator=g) / dx ** 0.5)
        self.U = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(RANK, dy)) for _ in range(NT)])

    def z(self, xs):
        return [(x - m) / s for x, m, s in zip(xs, self.mx, self.sx)]

    def forward(self, zs):
        code = torch.cat(zs, 1) @ self.V
        return [(zs[PAIR[j]] @ self.W[j] + self.b[j] + code @ self.U[j]) * self.sy[j] + self.my[j] for j in range(NT)]


def make_bridge(kind, ridge):
    if kind == "all_layers":
        return AllLayers(ridge)
    br = sb4.Bridge(ridge, "dense")
    if kind in ("dense_long", "trained_r4"):
        br.load_state_dict(torch.load("ckpt_bridge_dense_r4.pt", mmap=True))
    return br


def make_opt(kind, br, scale):
    cfg = CFG[kind]
    if kind == "all_layers":
        groups = [{"params": list(br.W) + list(br.b), "lr": cfg["lr"] * scale},
                  {"params": [br.V] + list(br.U), "lr": cfg["lr_lowrank"] * scale}]
    else:
        groups = [{"params": list(br.parameters()), "lr": cfg["lr"] * scale}]
    try:
        return torch.optim.Adam(groups, fused=True)
    except Exception:
        return torch.optim.Adam(groups)


def load_prep():
    D = torch.load(sb4.PREP, weights_only=False, mmap=True)
    if SMOKE:
        D["va_tok"] = D["va_tok"][:32]
        D["Xva"] = {w: [x[:32] for x in v] for w, v in D["Xva"].items()}
        D["Tva"] = {w: [x[:32] for x in v] for w, v in D["Tva"].items()}
    return D


# ---------------------------------------------------------------- train (calibration only)
def train(kind):
    t_call = time.time()
    cfg = CFG[kind]
    D = load_prep()
    vpos, tok_all = D["vpos"], D["tr_tok"]
    resume = f"{WD}/ckpt_resume_{kind}{TAG}.pt"
    br = make_bridge(kind, D["ridge"])
    Ztr = {w: [z.contiguous() for z in br.z(D["Xtr"][w])] for w in WINDOWS}
    if os.path.exists(resume):
        R = torch.load(resume, weights_only=False, mmap=True)
        if R["done"]:
            print("TRAINING_DONE (already)", flush=True)
            return
        S = R["S"]
        if S["scale"] != 1.0:
            br = make_bridge(kind, D["ridge"])
        br.load_state_dict(R["model"])
        opt = make_opt(kind, br, S["scale"])
        opt.load_state_dict(R["opt"])
        best_state = R["best_state"]
        rng = np.random.default_rng()
        rng.bit_generator.state = S["rng"]
        del R
        print(f"resumed at step {S['step']}, t_acc {S['t_acc']:.0f}s; rss {rss_gb():.1f} GB", flush=True)
    else:
        S = {"step": 0, "t_acc": 0.0, "evals": [], "train": [], "since": 0, "best": None, "scale": 1.0,
             "attempts": [], "oi": 0, "order": None, "n_params": sum(p.numel() for p in br.parameters())}
        opt = make_opt(kind, br, 1.0)
        rng = np.random.default_rng(cfg["seed"])
        best_state = None
    br.train()
    t_seg = time.time()
    t_base = S["t_acc"]

    def elapsed():
        return t_base + time.time() - t_seg

    def evaluate():
        br.eval()
        r = sb4.val_diag(D, br, PW)
        br.train()
        e = {"step": S["step"], "t": round(elapsed(), 1), "val_value_ce": r["bridge"]["value_ce"],
             "val_value_acc": r["bridge"]["value_acc"], "val_mismatched_value_ce": r["bridge_mismatched"]["value_ce"],
             "val_mismatched_value_acc": r["bridge_mismatched"]["value_acc"], "rss_gb": round(rss_gb(), 2)}
        S["evals"].append(e)
        print(kind, e, flush=True)
        return e

    def snapshot():
        return {k: v.detach().clone() for k, v in br.state_dict().items()}

    if S["step"] == 0 and not S["evals"]:
        e = evaluate()
        print(kind, "params", S["n_params"], flush=True)
        S["best"], best_state = [e["val_value_ce"], 0], snapshot()
    done = False
    while True:
        if elapsed() >= cfg["minutes"] * 60:
            done = True
            break
        if time.time() - t_seg >= SEG_MIN * 60:
            break
        w = WINDOWS[S["step"] % len(WINDOWS)]
        p = T - w
        if S["order"] is None or S["oi"] + BATCH > len(S["order"]):
            S["order"], S["oi"] = rng.permutation(len(tok_all)), 0
        idx = torch.from_numpy(S["order"][S["oi"]:S["oi"] + BATCH])
        S["oi"] += BATCH
        tok = tok_all[idx]
        ys = br([z[idx] for z in Ztr[w]])
        lv, la = sb4.loss_fn(sb4.window_logits(tgt, tok, p, sb4.tgt_init(ys)), tok, p, vpos)
        loss = lv + la
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        S["step"] += 1
        S["train"].append([S["step"], w, round(float(lv.detach()), 4), round(float(la.detach()), 4)])
        if S["step"] % VAL_EVERY == 0:
            e = evaluate()
            recent = np.array(S["train"][-VAL_EVERY:])[:, 2:].mean(0)
            print(kind, "train(v,all)", recent.round(3).tolist(), flush=True)
            first = [x for x in S["evals"] if x["step"] > 0]
            if len(first) == 1 and S["scale"] == 1.0 and e["val_value_ce"] > S["evals"][0]["val_value_ce"] + 0.5:
                print("fallback: first eval worse by > 0.5; restarting from init with lrs / 4", flush=True)
                S["attempts"].append({"evals": S["evals"], "train": S["train"]})
                br = make_bridge(kind, D["ridge"]).train()
                opt = make_opt(kind, br, 0.25)
                rng = np.random.default_rng(cfg["seed"])
                ev0 = S["evals"][0]
                S.update(step=0, evals=[ev0], train=[], since=0, scale=0.25, order=None, oi=0)
                continue
            if e["val_value_ce"] < S["best"][0]:
                S["best"], best_state, S["since"] = [e["val_value_ce"], S["step"]], snapshot(), 0
            else:
                S["since"] += 1
            if S["since"] >= PATIENCE:
                done = True
                break
    S["t_acc"] = elapsed()
    S["rng"] = rng.bit_generator.state
    if done:
        torch.save(best_state, f"{WD}/ckpt_bridge_{kind}{TAG}.pt")
        log = {k: v for k, v in S.items() if k not in ("order", "rng")}
        log.update(kind=kind, cfg=cfg, batch=BATCH, val_every=VAL_EVERY, patience=PATIENCE, rank=RANK,
                   stop="patience" if S["since"] >= PATIENCE else "cap", best_step=S["best"][1],
                   best_val_value_ce=S["best"][0], peak_rss_gb=rss_gb())
        json.dump(log, open(f"{WD}/train_bridge_{kind}{TAG}.json", "w"), indent=1)
        torch.save({"done": True}, resume)
        print(f"TRAINING_DONE step {S['step']} best {S['best']} t_acc {S['t_acc']:.0f}s rss {rss_gb():.1f} GB", flush=True)
    else:
        torch.save({"done": False, "S": S, "model": br.state_dict(), "opt": opt.state_dict(), "best_state": best_state},
                   resume + ".tmp")
        os.replace(resume + ".tmp", resume)
        print(f"SEGMENT_DONE step {S['step']} t_acc {S['t_acc']:.0f}s call {time.time()-t_call:.0f}s "
              f"rss {rss_gb():.1f} GB", flush=True)


# ---------------------------------------------------------------- test (read once)
BRIDGES = ["all_layers", "dense_long", "trained_r4"]


def test():
    t0 = time.time()
    D = load_prep()
    S = streams()
    if SMOKE:
        docs, vpos = eval_docs(S["calib"], N_TEST, T, seed=99, nbind=NB)
    else:
        docs, vpos = eval_docs(S["test"], N_TEST, T, seed=2, nbind=NB)
    assert (vpos == D["vpos"]).all()
    bridges = {}
    for arm in BRIDGES:
        br = make_bridge(arm if arm != "dense_long" else "dense", D["ridge"])
        ck = "ckpt_bridge_dense_r4.pt" if arm == "trained_r4" else f"{WD}/ckpt_bridge_{arm}{TAG}.pt"
        br.load_state_dict(torch.load(ck, mmap=True))
        bridges[arm] = br.eval()
    del D
    store = {}
    cuts = [T - w for w in WINDOWS] + [T + C - 1]
    for s in range(0, len(docs), sb2.BATCH):
        tok = sb2.tt(docs[s:s + sb2.BATCH])
        B = len(tok)
        tst, tlo = sb2.chain(tgt, tok, cuts)
        sst, slo = sb2.chain(src, tok, cuts)
        sb2.add(store, "full", sb2.metrics(*sb2.score(tlo[:, -C:], tok), vpos))
        sb2.add(store, "source_alone", sb2.metrics(*sb2.score(slo[:, -C:], tok), vpos))
        for w in WINDOWS:
            p = T - w
            with torch.no_grad():
                win = lambda init: sb2.metrics(*sb2.score(sb4.window_logits(tgt, tok, p, init)[:, -C:], tok), vpos)
                true = {j: sb2.ssm_flat(tst[p], i) for j, i in enumerate(T_IDX)}
                sb2.add(store, f"oracle@{w}", win(sb2.init_from(true, B)))
                sb2.add(store, f"blank@{w}", win(None))
                xs = [torch.from_numpy(sb2.ssm_flat(sst[p], i)) for i in S_IDX]
                for arm, br in bridges.items():
                    ys = br(br.z(xs))
                    sb2.add(store, f"{arm}@{w}", win(sb4.tgt_init(ys)))
                    sb2.add(store, f"{arm}_mismatched@{w}", win(sb4.tgt_init([torch.roll(y, -1, 0) for y in ys])))
        print(f"test batch {s // sb2.BATCH}: {time.time() - t0:.0f}s rss {rss_gb():.1f} GB", flush=True)

    R = {a: sb2.stack(v) for a, v in store.items()}
    arms = ["oracle", "blank"] + [x for a in BRIDGES for x in (a, f"{a}_mismatched")]
    comps = [(a, ["blank", f"{a}_mismatched", "source_alone", "oracle", "full"]) for a in BRIDGES]
    comps += [("all_layers", ["trained_r4", "dense_long"]), ("dense_long", ["trained_r4"]), ("oracle", ["blank"])]
    by_w = {}
    for w in WINDOWS:
        key = lambda a: a if a in ("source_alone", "full") else f"{a}@{w}"
        row = {"means": {a: {k: float(v.mean()) for k, v in R[key(a)].items()} for a in arms + ["source_alone", "full"]}}
        for metric in ["value_ce", "value_acc", "all_ce"]:
            g = lambda a: R[key(a)][metric]
            d = {}
            if metric != "value_acc":
                bl, full = g("blank"), g("full")
                gap = bl.mean() - full.mean()
                d["gap_blank_minus_full"] = float(gap)
                for a in arms:
                    d[f"gap_closed_{a}"] = float((bl.mean() - g(a).mean()) / gap)
            for a, ctrls in comps:
                for c in ctrls:
                    d[f"diff_{a}_minus_{c}"] = sb2.boot(g(a) - g(c))
            if metric == "value_ce":
                for a in BRIDGES:
                    d[f"{a}_beats_all_three"] = bool(d[f"diff_{a}_minus_blank"][2] < 0 and
                                                     d[f"diff_{a}_minus_{a}_mismatched"][2] < 0 and
                                                     d[f"diff_{a}_minus_source_alone"][2] < 0)
            row[metric] = d
        m = row["means"]
        row["oracle_acc_share"] = {a: {"ratio": m[a]["value_acc"] / m["oracle"]["value_acc"],
                                       "blank_corrected": (m[a]["value_acc"] - m["blank"]["value_acc"]) /
                                       (m["oracle"]["value_acc"] - m["blank"]["value_acc"])}
                                   for a in arms if a not in ("oracle", "blank")}
        row["doc_specific"] = {}
        for a in BRIDGES:
            ce_gain = m["blank"]["value_ce"] - m[a]["value_ce"]
            ds = m[f"{a}_mismatched"]["value_ce"] - m[a]["value_ce"]
            row["doc_specific"][a] = {"ce_gain_over_blank": ce_gain, "ce_matched_minus_mismatched": -ds,
                                      "frac_of_ce_gain_doc_specific": ds / ce_gain,
                                      "acc_matched_minus_mismatched": m[a]["value_acc"] - m[f"{a}_mismatched"]["value_acc"],
                                      "acc_gain_over_blank": m[a]["value_acc"] - m["blank"]["value_acc"]}
        by_w[str(w)] = row
    trlogs = {k: json.load(open(f"{WD}/train_bridge_{k}{TAG}.json")) for k in ["all_layers", "dense_long"]}
    res = {
        "protocol": {"T": T, "C": C, "n_bindings": NB, "windows": WINDOWS, "primary_window": PW,
                     "pairing": {str(k): v for k, v in PAIR.items()}, "bridge_cfg": CFG, "batch": BATCH,
                     "rank_all_layers": RANK, "val_every": VAL_EVERY, "patience": PATIENCE,
                     "units": "nats/byte; value_ce = mean over the 24 recalled value bytes per doc, then over docs",
                     "code": "statebridge5.py (protocol in docstring)"},
        "data": {"n_test_docs": len(docs), "smoke": SMOKE, "calibration_prep": "states_r4.pt (round 4, reused)"},
        "models": "ckpt_src_r3b.pt, ckpt_tgt_r3b.pt (frozen)",
        "round4_dense_training_log": json.load(open("train_bridge_dense_r4.json"))["attempts"][-1]["evals"],
        "bridge_training": trlogs,
        "full_prefill": {k: float(v.mean()) for k, v in R["full"].items()},
        "source_alone": {k: float(v.mean()) for k, v in R["source_alone"].items()},
        "by_window": by_w,
        "per_doc": {a: {k: v.tolist() for k, v in mm.items()} for a, mm in R.items()},
        "runtime_s": time.time() - t0, "peak_rss_gb": rss_gb(),
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/round5-results.json", "w"), indent=1)
    print("full", res["full_prefill"], "\nsource_alone", res["source_alone"])
    for w in WINDOWS:
        r = by_w[str(w)]
        print(w, {a: round(r["means"][a]["value_ce"], 3) for a in arms})
        print("   acc", {a: round(r["means"][a]["value_acc"], 3) for a in arms})
        print("   ", {k: ([round(x, 3) for x in v] if isinstance(v, list) else v)
                      for k, v in r["value_ce"].items() if k.startswith("diff_") or "beats" in k})
        print("   acc diffs", {k: [round(x, 3) for x in v] for k, v in r["value_acc"].items() if k.startswith("diff_")})
        print("   share", {a: {k: round(v, 3) for k, v in s.items()} for a, s in r["oracle_acc_share"].items()})
        print("   docspec", {a: {k: round(v, 3) for k, v in s.items()} for a, s in r["doc_specific"].items()})
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2])
    elif sys.argv[1] == "test":
        test()
