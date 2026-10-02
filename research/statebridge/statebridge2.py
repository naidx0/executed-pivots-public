"""StateBridge round 2: a toy where long context matters (associative recall in code-like text).

ROUND 2 PROTOCOL (fixed and committed before any test-set number was computed)

Task and data (recall.py)
- Eval doc = 6 bindings `cfg_abc = "q7rt"` (bytes [0,102)), then real stdlib code up to byte
  T = 864, then 6 queries `assert cfg_abc == "q7rt"` in random order: C = 150 continuation bytes
  [T, T+C). Recalled value bytes = the 4 value bytes of each query (24 per doc). They are
  predictable only from the bindings, >= 762 bytes before T, i.e. >= 506 bytes outside the largest
  window tested. Docs are 1014 bytes <= the 1024-byte training length (no length extrapolation).
- Code filler comes from the round-1 split: training rows use the train split, calibration docs
  the calib split, test docs the test split. Test docs: 96 (seed 2). Gate docs: 64 (seed 1).
- Models: round-1 checkpoints fine-tuned on a 50/50 mix of stdlib windows and packed recall
  episodes (train.py, round-2 options recorded in train_{src,tgt}_r2.json).

Gate (calibration docs only, target only, no maps involved)
- At the primary window w = 32, on the 64 gate docs, mean loss on recalled value bytes:
  oracle (target's own true SSM state at T-w) must beat blank + w AND mismatched-own (target's
  own true SSM state from a different gate doc) + w by >= 0.5 nats/byte each, with the 95% paired
  bootstrap CI of each difference entirely below 0.
- If the gate fails: change training only (never T, windows, docs, metrics or arms), retrain,
  re-run the gate; every attempt is logged in gate_r2_attempts.json and in the notes.

Calibration and maps
- Calibration samples: both models read 320 synthetic calib-split docs (seed 1; includes the 64
  gate docs) and the 112 plain calib-split stdlib files (first 1024 bytes); SSM states every 64
  bytes from byte 128. Validation = every 5th doc.
- Layer pairing as round 1: target SSM j <- source SSM round(3j/7).
- Dense map: per target SSM layer, affine ridge from the standardised flattened source state
  (4096) to the flattened target state (6144); lambda from {1e-4..10} by pooled val R^2, then
  refit on all calibration samples.
- Scalable map (primary rank r = 64): PCA bottleneck. Top-r principal directions of the
  standardised source state and of the centred target state (randomised SVD, as one would at
  scale), ridge between the r-dim codes (lambda by val R^2), W = Ux A Vy^T.
  Parameters r*(dx+dy)+r^2 (655k here vs 25M dense). Val R^2 also reported for r = 16, 256.

Test arms (per window w in {1, 8, 32, 128, 256}; target re-reads [T-w, T+C-1) and is scored on
[T, T+C); only SSM state is handed over; conv tail zero; attention KV empty)
- full target prefill; source alone (full prefill); oracle (true target SSM state) + w;
  StateBridge dense + w; StateBridge low-rank-64 + w; blank + w;
  mismatched control + w: the source's state from a DIFFERENT test doc of the same kind (next
  doc in the batch), mapped with the same map (dense and low-rank); mean calibration state + w.

Metrics
- Primary: mean CE (nats/byte) on the 24 recalled value bytes per doc; also byte accuracy
  (argmax) and exact-value accuracy (all 4 bytes right). Secondary: CE on all C bytes.
- Gap closed = (blank - arm) / (blank - full) at each w.
- Per-layer R^2 of both maps (val, and test at all handoff points).
- Paired bootstrap over test docs (10k resamples), 95% percentile CIs.
- HEADLINE: at w = 32, does StateBridge-dense beat all three controls (blank + w, mismatched + w,
  source alone) on recalled-value CE, each paired-difference CI entirely below 0?
  Also reported for w = 1 and 8, and for the low-rank map.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import split
from model import HybridLM
from recall import BIND_LEN, eval_docs, streams

torch.set_num_threads(int(os.environ.get("SB_THREADS", "4")))
torch.manual_seed(0)
OUT = os.environ.get("SB_OUT", "/mnt/project-files/nebius/statebridge")
T, NB = 864, 6
C = NB * 25
WINDOWS = [256, 128, 32, 8, 1]
PRIMARY_W = 32
N_TEST, N_GATE, N_CAL_SYN = 96, 64, 320
SUFFIX = os.environ.get("SB_SUFFIX", "_r2")  # smoke tests only
if os.environ.get("SB_SMOKE"):
    N_TEST, N_GATE, N_CAL_SYN = 16, 16, 20
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
RANKS, PRIMARY_RANK = [16, 64, 256], 64
SNAP_EVERY, SNAP_FROM, CAL_CAP = 64, 128, 1024
BATCH = 16
GATE_MARGIN = 0.5


def load(name):
    ck = torch.load(f"ckpt_{name}{SUFFIX}.pt")
    m = HybridLM(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.eval()


src, tgt = load("src"), load("tgt")
S_IDX, T_IDX = src.ssm_idx, tgt.ssm_idx
PAIR = {j: round(j * (len(S_IDX) - 1) / (len(T_IDX) - 1)) for j in range(len(T_IDX))}
L = tgt.layers[T_IDX[0]].L


def tt(bs):
    return torch.tensor(np.stack([np.frombuffer(b, dtype=np.uint8) for b in bs]).astype(np.int64))


def boot(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return [float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


@torch.no_grad()
def chain(model, tok, cuts):
    states, st, start, lo = {}, None, 0, None
    for c in cuts:
        lo, st, _ = model(tok[:, start:c], st)
        states[c] = st
        start = c
    return states, lo


def ssm_flat(st, idx):
    return st[idx][0].reshape(st[idx][0].shape[0], -1).numpy()


def score(lo, tok):
    """lo: logits for predicting bytes [T, T+C). Returns per-byte CE and argmax-correct (B, C)."""
    tg = tok[:, T:T + C]
    ce = F.cross_entropy(lo.reshape(-1, 256), tg.reshape(-1), reduction="none").view(tg.shape)
    return ce, (lo.argmax(-1) == tg)


@torch.no_grad()
def window(tok, p, init):
    lo, _, _ = tgt(tok[:, p:T + C - 1], init)
    return score(lo[:, -C:], tok)


def init_from(flat_by_layer, B):
    st = [None] * len(tgt.pattern)
    for j, i in enumerate(T_IDX):
        m = tgt.layers[i]
        h = torch.from_numpy(np.asarray(flat_by_layer[j], dtype=np.float32)).view(B, m.H, m.P, m.N)
        st[i] = (h, None)
    return st


def metrics(ce, ok, vpos):
    """Per-doc: value CE, value byte acc, exact value acc, all-bytes CE."""
    v = ce[:, vpos]
    vo = ok[:, vpos]
    exact = vo.view(len(vo), NB, 4).all(-1).float().mean(1)
    return {"value_ce": v.mean(1).numpy(), "value_acc": vo.float().mean(1).numpy(),
            "value_exact": exact.numpy(), "all_ce": ce.mean(1).numpy()}


def stack(acc):
    return {k: np.concatenate(v) for k, v in acc.items()}


def add(store, arm, m):
    for k, v in m.items():
        store.setdefault(arm, {}).setdefault(k, []).append(v)


# ---------------------------------------------------------------- gate
def gate(attempt):
    t0 = time.time()
    docs, vpos = eval_docs(streams()["calib"], N_GATE, T, seed=1, nbind=NB)
    store = {}
    for s in range(0, len(docs), BATCH):
        tok = tt(docs[s:s + BATCH])
        B = len(tok)
        cuts = [T - w for w in WINDOWS] + [T + C - 1]
        tst, tlo = chain(tgt, tok, cuts)
        add(store, "full", metrics(*score(tlo[:, -C:], tok), vpos))
        for w in WINDOWS:
            p = T - w
            true = {j: ssm_flat(tst[p], i) for j, i in enumerate(T_IDX)}
            add(store, f"oracle@{w}", metrics(*window(tok, p, init_from(true, B)), vpos))
            add(store, f"blank@{w}", metrics(*window(tok, p, None), vpos))
            mis = {j: np.roll(v, -1, 0) for j, v in true.items()}
            add(store, f"mismatched_own@{w}", metrics(*window(tok, p, init_from(mis, B)), vpos))
    R = {a: stack(v) for a, v in store.items()}
    res = {"attempt": attempt, "n_docs": len(docs), "full": {k: float(v.mean()) for k, v in R["full"].items()}, "by_window": {}}
    for w in WINDOWS:
        o, b, m = (R[f"{a}@{w}"]["value_ce"] for a in ["oracle", "blank", "mismatched_own"])
        row = {a: {k: float(v.mean()) for k, v in R[f"{a}@{w}"].items()} for a in ["oracle", "blank", "mismatched_own"]}
        row["diff_oracle_minus_blank"] = boot(o - b)
        row["diff_oracle_minus_mismatched_own"] = boot(o - m)
        res["by_window"][str(w)] = row
    r = res["by_window"][str(PRIMARY_W)]
    d1, d2 = r["diff_oracle_minus_blank"], r["diff_oracle_minus_mismatched_own"]
    res["gate_pass"] = bool(d1[0] <= -GATE_MARGIN and d2[0] <= -GATE_MARGIN and d1[2] < 0 and d2[2] < 0)
    res["models"] = {n: {k: v for k, v in json.load(open(f"train_{n}{SUFFIX}.json")).items() if k != "log"} for n in ["src", "tgt"]}
    res["runtime_s"] = time.time() - t0
    path = f"gate{SUFFIX}_attempts.json"
    allr = json.load(open(path)) if os.path.exists(path) else []
    allr = [a for a in allr if a["attempt"] != attempt] + [res]
    json.dump(allr, open(path, "w"), indent=1)
    print(json.dumps({"full": res["full"], "gate_pass": res["gate_pass"]}))
    for w in WINDOWS:
        r = res["by_window"][str(w)]
        print(w, {a: round(r[a]["value_ce"], 3) for a in ["oracle", "blank", "mismatched_own"]},
              {a: round(r[a]["value_exact"], 3) for a in ["oracle", "blank", "mismatched_own"]},
              [round(x, 3) for x in r["diff_oracle_minus_blank"]], [round(x, 3) for x in r["diff_oracle_minus_mismatched_own"]])


# ---------------------------------------------------------------- maps
def r2(Y, Yhat, ymean):
    return float(1 - ((Y - Yhat) ** 2).sum() / ((Y - ymean) ** 2).sum())


@torch.no_grad()
def collect(model, docs):
    per_layer = {i: [] for i in model.ssm_idx}
    doc_ids = []
    for d, b in enumerate(docs):
        n = min(len(b), CAL_CAP) // L * L
        _, _, snaps = model(tt([b[:n]]), snap=True)
        pos = np.arange(1, n // L + 1) * L
        keep = np.where((pos % SNAP_EVERY == 0) & (pos >= SNAP_FROM))[0]
        for i in model.ssm_idx:
            per_layer[i].append(snaps[i][0, keep].reshape(len(keep), -1).float())
        doc_ids += [d] * len(keep)
    return {i: torch.cat(v).numpy() for i, v in per_layer.items()}, np.array(doc_ids)


class Std:
    def __init__(self, X):
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-6

    def __call__(self, X):
        return (X - self.mx) / self.sx


class Dense:
    def __init__(self, std, W, my):
        self.std, self.W, self.my = std, W, my
        self.n_params = W.size + my.size

    def __call__(self, X):
        return self.std(X.astype(np.float64)) @ self.W + self.my


class LowRank:
    def __init__(self, std, Ux, A, Vy, my):
        self.std, self.Ux, self.A, self.Vy, self.my = std, Ux, A, Vy, my
        self.n_params = Ux.size + A.size + Vy.size + my.size

    def __call__(self, X):
        return ((self.std(X.astype(np.float64)) @ self.Ux) @ self.A) @ self.Vy.T + self.my


def pcs(M, q):
    torch.manual_seed(0)
    _, _, V = torch.svd_lowrank(torch.from_numpy(M), q=q + 10, niter=4)
    return V[:, :q].numpy()


def fit_maps(Xs, Ys, did):
    val = did % 5 == 4
    fit = ~val
    maps = {"dense": {}, "lowrank": {}}
    info = [dict() for _ in T_IDX]
    for k, si in enumerate(S_IDX):
        js = [j for j in range(len(T_IDX)) if PAIR[j] == k]
        if not js:
            continue
        t0 = time.time()
        X = Xs[si].astype(np.float64)
        sf, sa = Std(X[fit]), Std(X)
        Zf, Zv, Za = sf(X[fit]), sf(X[val]), sa(X)
        ef, Vf = np.linalg.eigh(Zf.T @ Zf)
        ea, Va = np.linalg.eigh(Za.T @ Za)
        Uxf, Uxa = pcs(Zf, max(RANKS)), pcs(Za, max(RANKS))
        for j in js:
            Y = Ys[T_IDX[j]].astype(np.float64)
            myf, mya = Y[fit].mean(0), Y.mean(0)
            # dense ridge path (val) and refit on all
            VtZY = Vf.T @ (Zf.T @ (Y[fit] - myf))
            ZvV = Zv @ Vf
            path = {lam: r2(Y[val], ZvV @ (VtZY / (ef + lam * len(Zf))[:, None]) + myf, myf) for lam in LAMBDAS}
            lam = max(path, key=path.get)
            W = Va @ ((Va.T @ (Za.T @ (Y - mya))) / (ea + lam * len(Za))[:, None])
            maps["dense"][j] = Dense(sa, W, mya)
            # low-rank PCA bottleneck
            Vyf, Vya = pcs(Y[fit] - myf, max(RANKS)), pcs(Y - mya, max(RANKS))
            lr_path, best = {}, {}
            for r in RANKS:
                Zr, Zvr, Yr = Zf @ Uxf[:, :r], Zv @ Uxf[:, :r], (Y[fit] - myf) @ Vyf[:, :r]
                G, R = Zr.T @ Zr, Zr.T @ Yr
                p = {lam_: r2(Y[val], (Zvr @ np.linalg.solve(G + lam_ * len(Zr) * np.eye(r), R)) @ Vyf[:, :r].T + myf, myf) for lam_ in LAMBDAS}
                lr_path[r] = p
                best[r] = max(p, key=p.get)
            r = PRIMARY_RANK
            Zr = Za @ Uxa[:, :r]
            A = np.linalg.solve(Zr.T @ Zr + best[r] * len(Zr) * np.eye(r), Zr.T @ ((Y - mya) @ Vya[:, :r]))
            maps["lowrank"][j] = LowRank(sa, Uxa[:, :r], A, Vya[:, :r], mya)
            info[j] = {"tgt_layer": T_IDX[j], "tgt_ssm_j": j, "src_layer": si, "src_ssm_k": k,
                       "dense_lambda": lam, "dense_val_r2": path[lam], "dense_val_r2_path": path,
                       "lowrank_val_r2": {str(r_): lr_path[r_][best[r_]] for r_ in RANKS},
                       "lowrank_lambda": {str(r_): best[r_] for r_ in RANKS},
                       "dense_calib_r2_insample": r2(Y, maps["dense"][j](Xs[si]), mya),
                       "dense_params": maps["dense"][j].n_params, "lowrank64_params": maps["lowrank"][j].n_params,
                       "src_dim": X.shape[1], "tgt_dim": Y.shape[1]}
            print(json.dumps({k_: v for k_, v in info[j].items() if "path" not in k_}), flush=True)
        print(f"source layer {si}: {time.time()-t0:.0f}s", flush=True)
    return maps, info


# ---------------------------------------------------------------- calibration-only extras
def calib_maps():
    """Map quality on calibration data only (no test docs): val R^2, dense vs low-rank.
    Also a short-gap diagnostic: full-prefill value CE when the queries follow the bindings after
    only `gap` bytes of code (calib split), to see whether any recall was learned at all."""
    t0 = time.time()
    S = streams()
    _, calib, _ = split()
    cal_syn, _ = eval_docs(S["calib"], N_CAL_SYN, T, seed=1, nbind=NB)
    cdocs = cal_syn + [b[:CAL_CAP] for _, b in calib]
    Xs, did = collect(src, cdocs)
    Ys, _ = collect(tgt, cdocs)
    _, info = fit_maps(Xs, Ys, did)
    diag = {}
    for gap_T in [NB * 17 + 8, NB * 17 + 64, NB * 17 + 256, T]:
        docs, vpos = eval_docs(S["calib"], 32, gap_T, seed=5, nbind=NB)
        tok = tt(docs)
        with torch.no_grad():
            out = {}
            for nm, m in [("tgt", tgt), ("src", src)]:
                lo, _, _ = m(tok[:, :-1])
                out[nm] = {k: float(v.mean()) for k, v in metrics(*score_at(lo[:, gap_T - 1:], tok, gap_T), vpos).items()}
        diag[str(gap_T - NB * 17)] = out
        print("gap", gap_T - NB * 17, out, flush=True)
    res = {"map_r2_per_layer_calib_only": info, "short_gap_diagnostic_full_prefill": diag,
           "n_calib_samples": int(len(did)), "runtime_s": time.time() - t0}
    json.dump(res, open("calib_maps_r2.json", "w"), indent=1)


def score_at(lo, tok, t):
    tg = tok[:, t:t + C]
    ce = F.cross_entropy(lo.reshape(-1, 256), tg.reshape(-1), reduction="none").view(tg.shape)
    return ce, (lo.argmax(-1) == tg)


# ---------------------------------------------------------------- run
def run():
    t0 = time.time()
    S = streams()
    _, calib, _ = split()
    cal_syn, _ = eval_docs(S["calib"], N_CAL_SYN, T, seed=1, nbind=NB)
    cal_plain = [b[:CAL_CAP] for _, b in calib][: 10 if os.environ.get("SB_SMOKE") else None]
    cdocs = cal_syn + cal_plain
    Xs, did = collect(src, cdocs)
    Ys, did2 = collect(tgt, cdocs)
    assert (did == did2).all()
    print(f"calib: {len(cdocs)} docs, {len(did)} samples; {time.time()-t0:.0f}s", flush=True)
    maps, layer_info = fit_maps(Xs, Ys, did)
    mean_state = {j: Ys[i].mean(0) for j, i in enumerate(T_IDX)}
    del Xs, Ys
    print(f"maps fitted; {time.time()-t0:.0f}s", flush=True)

    docs, vpos = eval_docs(S["test"], N_TEST, T, seed=2, nbind=NB)
    store = {}
    pairs = {mk: {j: ([], []) for j in range(len(T_IDX))} for mk in maps}
    cuts = [T - w for w in WINDOWS] + [T + C - 1]
    for s in range(0, len(docs), BATCH):
        tok = tt(docs[s:s + BATCH])
        B = len(tok)
        tst, tlo = chain(tgt, tok, cuts)
        sst, slo = chain(src, tok, cuts)
        add(store, "full", metrics(*score(tlo[:, -C:], tok), vpos))
        add(store, "source_alone", metrics(*score(slo[:, -C:], tok), vpos))
        for w in WINDOWS:
            p = T - w
            true = {j: ssm_flat(tst[p], i) for j, i in enumerate(T_IDX)}
            srcf = {k: ssm_flat(sst[p], i) for k, i in enumerate(S_IDX)}
            add(store, f"oracle@{w}", metrics(*window(tok, p, init_from(true, B)), vpos))
            add(store, f"blank@{w}", metrics(*window(tok, p, None), vpos))
            ms = {j: np.tile(mean_state[j], (B, 1)) for j in mean_state}
            add(store, f"mean_state@{w}", metrics(*window(tok, p, init_from(ms, B)), vpos))
            for mk, arm in [("dense", "statebridge"), ("lowrank", "statebridge_lowrank64")]:
                mapped = {j: maps[mk][j](srcf[PAIR[j]]) for j in range(len(T_IDX))}
                add(store, f"{arm}@{w}", metrics(*window(tok, p, init_from(mapped, B)), vpos))
                for j in range(len(T_IDX)):
                    pairs[mk][j][0].append(true[j])
                    pairs[mk][j][1].append(mapped[j])
                # mismatched control: state from a different test doc (next doc in batch), same map
                mis = {j: np.roll(v, -1, 0) for j, v in mapped.items()}
                add(store, f"mismatched{'_lowrank64' if mk == 'lowrank' else ''}@{w}", metrics(*window(tok, p, init_from(mis, B)), vpos))
        print(f"test batch {s//BATCH}: {time.time()-t0:.0f}s", flush=True)

    R = {a: stack(v) for a, v in store.items()}
    for mk in maps:
        for j in range(len(T_IDX)):
            Yt, Yh = np.concatenate(pairs[mk][j][0]), np.concatenate(pairs[mk][j][1])
            layer_info[j][f"{mk}_test_r2"] = r2(Yt, Yh, maps[mk][j].my)

    arms = ["oracle", "statebridge", "statebridge_lowrank64", "blank", "mismatched", "mismatched_lowrank64", "mean_state"]
    by_w = {}
    for w in WINDOWS:
        row = {"means": {a: {k: float(v.mean()) for k, v in R[f"{a}@{w}"].items()} for a in arms}}
        for metric in ["value_ce", "all_ce"]:
            g = lambda a: R[a][metric]
            bl, full = g(f"blank@{w}"), g("full")
            gap = bl.mean() - full.mean()
            d = {"gap_blank_minus_full": float(gap)}
            for a in ["statebridge", "statebridge_lowrank64", "oracle", "mismatched", "mean_state"]:
                d[f"gap_closed_{a}"] = float((bl.mean() - g(f"{a}@{w}").mean()) / gap) if abs(gap) > 1e-9 else None
            for a, ctrls in [("statebridge", [f"blank@{w}", f"mismatched@{w}", "source_alone", f"oracle@{w}", "full", f"mean_state@{w}"]),
                             ("statebridge_lowrank64", [f"blank@{w}", f"mismatched_lowrank64@{w}", "source_alone", f"statebridge@{w}"])]:
                for c in ctrls:
                    d[f"diff_{a}_minus_{c.split('@')[0]}"] = boot(g(f"{a}@{w}") - g(c))
            d["diff_oracle_minus_blank"] = boot(g(f"oracle@{w}") - bl)
            d["diff_blank_minus_full"] = boot(bl - full)
            for a, mis in [("statebridge", "mismatched"), ("statebridge_lowrank64", "mismatched_lowrank64")]:
                d[f"{a}_beats_all_three"] = bool(d[f"diff_{a}_minus_blank"][2] < 0 and d[f"diff_{a}_minus_{mis}"][2] < 0
                                                 and d[f"diff_{a}_minus_source_alone"][2] < 0)
            row[metric] = d
        by_w[str(w)] = row
    res = {
        "protocol": {"T": T, "C": C, "n_bindings": NB, "windows": WINDOWS, "primary_small_window": PRIMARY_W,
                     "lambdas": LAMBDAS, "ranks": RANKS, "primary_rank": PRIMARY_RANK, "snap_every": SNAP_EVERY,
                     "snap_from": SNAP_FROM, "pairing": {str(k): v for k, v in PAIR.items()},
                     "min_distance_value_to_T": T - NB * BIND_LEN,
                     "units": "nats/byte; value_ce = mean over the 24 recalled value bytes per doc, then over docs"},
        "data": {"n_calib_docs_synthetic": len(cal_syn), "n_calib_docs_plain": len(cal_plain),
                 "n_calib_samples": int(len(did)), "n_test_docs": len(docs)},
        "models": {n: {k: v for k, v in json.load(open(f"train_{n}{SUFFIX}.json")).items() if k != "log"} for n in ["src", "tgt"]},
        "gate_attempts": json.load(open(f"gate{SUFFIX}_attempts.json")),
        "full_prefill": {k: float(v.mean()) for k, v in R["full"].items()},
        "source_alone": {k: float(v.mean()) for k, v in R["source_alone"].items()},
        "by_window": by_w,
        "map_r2_per_layer": layer_info,
        "per_doc": {a: {k: v.tolist() for k, v in m.items()} for a, m in R.items()},
        "runtime_s": time.time() - t0,
    }
    json.dump(res, open(f"{OUT}/round2-results.json", "w"), indent=1)
    print("full", res["full_prefill"], "\nsource_alone", res["source_alone"])
    for w in WINDOWS:
        r = by_w[str(w)]
        print(w, {a: round(r["means"][a]["value_ce"], 3) for a in arms})
        print("   exact", {a: round(r["means"][a]["value_exact"], 3) for a in arms})
        print("   ", {k: ([round(x, 3) for x in v] if isinstance(v, list) else (round(v, 3) if isinstance(v, float) else v))
                      for k, v in r["value_ce"].items() if "beats" in k or "gap" in k or "minus_blank" in k or "minus_mismatched" in k or "source_alone" in k})
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    if sys.argv[1] == "gate":
        gate(int(sys.argv[2]))
    elif sys.argv[1] == "calib_maps":
        calib_maps()
    elif sys.argv[1] == "run":
        run()
