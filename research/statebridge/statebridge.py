"""(Round 1. Round 2, with a long-range recall task and a gate, is statebridge2.py.)

StateBridge toy test: map source-model SSM states into the target model and resume with a
repair window.

PROTOCOL (fixed before any test-set number was computed)
- Prefix T = 8192 bytes, scored continuation = the next C = 512 bytes of the same file.
  (Fallback T = 4096 only if the calibration-set precheck shows full prefill is no better than
  blank-state + 2048-byte window, i.e. the models do not use long context at 8K.)
- Repair windows w in {1, 32, 128, 512, 2048}: the target re-reads bytes [T-w, T+C-1) and is
  scored on bytes [T, T+C). w=1 = only the last prefix byte is re-read. Primary "small window": w=32.
- Layer pairing: target SSM layer j (0..7) <- source SSM layer round(j*3/7) (relative depth).
- Map: per target SSM layer, affine ridge map from the flattened source state (H*P*N = 4096) to the
  flattened target state (6144). X standardised per feature; lambda (penalty lambda*n*I) chosen
  per layer from a grid by pooled R^2 on held-out calibration docs (every 5th doc), then refit
  on all calibration docs.
- Calibration samples: both models read each calibration doc (first <= 8192 bytes) from byte 0;
  states taken every 128 bytes from byte 256.
- Mapped state goes into the SSM recurrent state only; the conv tail starts at zero and attention
  KV starts empty; both are rebuilt from the repair window.
- Arms: full target prefill; oracle (true target SSM state at T-w, conv zeroed, empty KV) + w;
  StateBridge (ridge map) + w; blank + w; shuffled (source reads the line-shuffled prefix
  [0,T-w), mapped) + w; mean calibration state + w; source alone (full prefill); sanity: true
  target SSM+conv state after byte T-2, empty KV, re-read byte T-1.
- Ablations: plain least squares (min-norm lstsq, all calib docs) and ridge fitted on 1/8 of the
  fit docs.
- Stats: per-doc mean loss (nats/byte); paired bootstrap over docs (10k) for differences.
- Headline: StateBridge beats all three dossier controls (source alone, blank+w, shuffled+w) at
  w=32 with the 95% CI of every paired difference below zero.

AMENDMENT (made after the calibration-set precheck, before any test-set number was computed)
- Precheck on 16 calibration docs: the trained target gains almost nothing from context beyond
  the last ~100 bytes (T=8192: full 1.248, blank+2048 1.243, blank+32 1.247, blank+1 1.258
  nats/byte). The pre-registered fallback rule fired, so T = 4096 (it fails there too).
- Added w=8 and a secondary metric: loss on the first 64 continuation bytes (where the gap between
  blank and full is concentrated). Calibration states are still read from the first <= 8192 bytes.
- The primary window stays w=32 as registered.
"""
import json
import os
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import split
from model import HybridLM

torch.set_num_threads(int(os.environ.get("SB_THREADS", "4")))
torch.manual_seed(0)
OUT = os.environ.get("SB_OUT", "/mnt/project-files/nebius/statebridge")
T = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
C = 512
WINDOWS = [2048, 512, 128, 32, 8, 1]
HEAD = 64  # secondary metric: loss on the first HEAD continuation bytes
CALIB_CAP = 8192
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
SNAP_EVERY, SNAP_FROM = 128, 256
BATCH = 8


def load(name):
    ck = torch.load(f"ckpt_{name}.pt")
    m = HybridLM(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.eval()


src, tgt = load("src"), load("tgt")
S_IDX, T_IDX = src.ssm_idx, tgt.ssm_idx
PAIR = {j: round(j * (len(S_IDX) - 1) / (len(T_IDX) - 1)) for j in range(len(T_IDX))}
L = tgt.layers[T_IDX[0]].L
assert L == src.layers[S_IDX[0]].L


def tt(bs):
    return torch.tensor(np.stack([np.frombuffer(b, dtype=np.uint8) for b in bs]).astype(np.int64))


@torch.no_grad()
def collect(model, docs, cap):
    """Per SSM layer: (n_samples, H*P*N) float32 state samples; plus doc id per sample."""
    per_layer = {i: [] for i in model.ssm_idx}
    doc_ids = []
    for d, b in enumerate(docs):
        n = min(len(b), cap) // L * L
        _, _, snaps = model(tt([b[:n]]), snap=True)
        pos = np.arange(1, n // L + 1) * L
        keep = np.where((pos % SNAP_EVERY == 0) & (pos >= SNAP_FROM))[0]
        for i in model.ssm_idx:
            per_layer[i].append(snaps[i][0, keep].reshape(len(keep), -1).float())
        doc_ids += [d] * len(keep)
    return {i: torch.cat(v).numpy() for i, v in per_layer.items()}, np.array(doc_ids)


def r2(Y, Yhat, ymean):
    return float(1 - ((Y - Yhat) ** 2).sum() / ((Y - ymean) ** 2).sum())


class Map:
    def __init__(self, X, Y, lam, ols=False):
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
        self.mx = X.mean(0)
        self.sx = X.std(0) + 1e-6
        self.my = Y.mean(0)
        Z = (X - self.mx) / self.sx
        if ols:
            self.W = np.linalg.lstsq(Z, Y - self.my, rcond=None)[0]
        else:
            G = Z.T @ Z
            self.W = np.linalg.solve(G + lam * len(Z) * np.eye(len(G)), Z.T @ (Y - self.my))
        self.lam = lam

    def __call__(self, X):
        return ((X - self.mx) / self.sx) @ self.W + self.my


def ridge_path(X, Y, Xv, Yv):
    """Val R^2 for every lambda, using one eigendecomposition."""
    X = X.astype(np.float64)
    mx, sx = X.mean(0), X.std(0) + 1e-6
    my = Y.mean(0)
    Z = (X - mx) / sx
    Zv = (Xv - mx) / sx
    e, V = np.linalg.eigh(Z.T @ Z)
    VtZY = V.T @ (Z.T @ (Y - my))
    ZvV = Zv @ V
    out = {}
    for lam in LAMBDAS:
        Yh = ZvV @ (VtZY / (e + lam * len(Z))[:, None]) + my
        out[lam] = r2(Yv, Yh, Y.mean(0))
    return out


def shuffle_lines(b, seed):
    lines = b.split(b"\n")
    last = lines.pop()  # keep byte length: re-join with newlines
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lines))
    out = b"\n".join(lines[k] for k in order) + b"\n" + last if lines else b
    assert len(out) == len(b)
    return out


@torch.no_grad()
def chain(model, tok, cuts):
    """Run tok through the model in segments ending at each cut; return states at each cut and
    logits of the last segment."""
    states, st, start, lo = {}, None, 0, None
    for c in cuts:
        lo, st, _ = model(tok[:, start:c], st)
        states[c] = st
        start = c
    return states, lo


def ssm_flat(st, idx):
    return st[idx][0].detach().reshape(st[idx][0].shape[0], -1).numpy()


@torch.no_grad()
def window_loss(tok, p, init):
    """Target reads tok[:, p:T+C-1] from init states; per-doc loss on bytes [T, T+C)."""
    lo, _, _ = tgt(tok[:, p:T + C - 1], init)
    lo = lo[:, -C:]
    return F.cross_entropy(lo.reshape(-1, 256), tok[:, T:T + C].reshape(-1), reduction="none").view(-1, C)


def init_from(flat_by_layer, B):
    st = [None] * len(tgt.pattern)
    for j, i in enumerate(T_IDX):
        m = tgt.layers[i]
        h = torch.from_numpy(np.asarray(flat_by_layer[j], dtype=np.float32)).view(B, m.H, m.P, m.N)
        st[i] = (h, None)
    return st


def boot(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return [float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def precheck(calib):
    """Calibration-set check that long context is used at prefix T (decides T before test)."""
    docs = [b[:T + C] for _, b in calib if len(b) >= T + C][:16]
    tok = tt(docs)
    res = {"n_docs": len(docs)}
    with torch.no_grad():
        for name, model in [("tgt", tgt), ("src", src)]:
            lo, _, _ = model(tok[:, :T + C - 1])
            res[f"{name}_full"] = float(F.cross_entropy(lo[:, -C:].reshape(-1, 256), tok[:, T:].reshape(-1)))
        for w in [2048, 512, 128, 32, 16, 8, 4, 1]:
            res[f"tgt_blank_w{w}"] = float(window_loss(tok, T - w, None).mean())
    return res


def main():
    t0 = time.time()
    train, calib, test = split()
    if sys.argv[1] == "precheck":
        print(json.dumps(precheck(calib), indent=1))
        return

    # ---------------- calibration ----------------
    cdocs = [b for _, b in calib][: int(os.environ.get("SB_MAX_CALIB", "100000"))]
    Xs, did = collect(src, cdocs, CALIB_CAP)
    Ys, did2 = collect(tgt, cdocs, CALIB_CAP)
    assert (did == did2).all()
    val = did % 5 == 4
    fit = ~val
    fit_docs = np.unique(did[fit])
    small = np.isin(did, fit_docs[: max(1, len(fit_docs) // 8)])
    print(f"calib: {len(cdocs)} docs, {len(did)} samples ({fit.sum()} fit / {val.sum()} val); {time.time()-t0:.0f}s", flush=True)

    maps = {"ridge": {}, "ols": {}, "ridge_small": {}}
    layer_info = []
    for j, i in enumerate(T_IDX):
        X = Xs[S_IDX[PAIR[j]]]
        Y = Ys[i]
        path = ridge_path(X[fit], Y[fit], X[val], Y[val])
        lam = max(path, key=path.get)
        path_small = ridge_path(X[small], Y[small], X[val], Y[val])
        lam_s = max(path_small, key=path_small.get)
        # val R^2 of the ablations, fitted on the fit split only (apples to apples)
        ols_fit = Map(X[fit], Y[fit], 0, ols=True)
        r2_ols_val = r2(Y[val], ols_fit(X[val]), Y[fit].mean(0))
        maps["ridge"][j] = Map(X, Y, lam)
        maps["ols"][j] = Map(X, Y, 0, ols=True)
        maps["ridge_small"][j] = Map(X[small], Y[small], lam_s)
        info = {"tgt_layer": i, "tgt_ssm_j": j, "src_layer": S_IDX[PAIR[j]], "src_ssm_k": PAIR[j],
                "lambda": lam, "val_r2_ridge": path[lam], "val_r2_path": path,
                "lambda_small": lam_s, "val_r2_ridge_small": path_small[lam_s], "val_r2_ols": r2_ols_val,
                "calib_r2_ridge_insample": r2(Y, maps["ridge"][j](X), Y.mean(0)),
                "tgt_state_dim": Y.shape[1], "src_state_dim": X.shape[1]}
        layer_info.append(info)
        print(json.dumps({k: v for k, v in info.items() if k != "val_r2_path"}), flush=True)
    mean_state = {j: Ys[i].mean(0) for j, i in enumerate(T_IDX)}
    n_small_docs = len(np.unique(did[small]))
    del Xs, Ys
    print(f"maps fitted; {time.time()-t0:.0f}s", flush=True)

    # ---------------- test ----------------
    tdocs = [b[:T + C] for _, b in test if len(b) >= T + C][: int(os.environ.get("SB_MAX_TEST", "100000"))]
    names = [r for r, b in test if len(b) >= T + C][: len(tdocs)]
    cuts = [T - w for w in WINDOWS] + [T + C - 1]
    arms = ["oracle", "statebridge", "blank", "shuffled", "mean_state", "sb_ols", "sb_ridge_small"]
    L_ = {f"{a}@{w}": [] for a in arms for w in WINDOWS}
    L_["full"], L_["source_alone"], L_["sanity_true_state_no_window"] = [], [], []
    test_pairs = {j: ([], []) for j in range(len(T_IDX))}  # (Y true, Y hat) at all handoff points
    for s in range(0, len(tdocs), BATCH):
        bs = tdocs[s:s + BATCH]
        B = len(bs)
        tok = tt(bs)
        tst, tlo = chain(tgt, tok, cuts)
        sst, slo = chain(src, tok, cuts)
        tgts = tok[:, T:T + C].reshape(-1)
        L_["full"] += F.cross_entropy(tlo[:, -C:].reshape(-1, 256), tgts, reduction="none").view(B, C).tolist()
        L_["source_alone"] += F.cross_entropy(slo[:, -C:].reshape(-1, 256), tgts, reduction="none").view(B, C).tolist()
        # sanity: true SSM + conv state after bytes [0, T-1), attention empty, read byte T-1
        st = [tst[T - 1][i] if c == "M" else None for i, c in enumerate(tgt.pattern)]
        L_["sanity_true_state_no_window"] += window_loss(tok, T - 1, st).tolist()
        for w in WINDOWS:
            p = T - w
            true_flat = {j: ssm_flat(tst[p], i) for j, i in enumerate(T_IDX)}
            src_flat = {k: ssm_flat(sst[p], i) for k, i in enumerate(S_IDX)}
            L_[f"oracle@{w}"] += window_loss(tok, p, init_from(true_flat, B)).tolist()
            L_[f"blank@{w}"] += window_loss(tok, p, None).tolist()
            L_[f"mean_state@{w}"] += window_loss(tok, p, init_from({j: np.tile(mean_state[j], (B, 1)) for j in mean_state}, B)).tolist()
            for arm, mk in [("statebridge", "ridge"), ("sb_ols", "ols"), ("sb_ridge_small", "ridge_small")]:
                mapped = {j: maps[mk][j](src_flat[PAIR[j]]) for j in range(len(T_IDX))}
                L_[f"{arm}@{w}"] += window_loss(tok, p, init_from(mapped, B)).tolist()
                if mk == "ridge":
                    for j in range(len(T_IDX)):
                        test_pairs[j][0].append(true_flat[j])
                        test_pairs[j][1].append(mapped[j])
            # shuffled control: source reads a line-shuffled prefix [0,p)
            sh = tt([shuffle_lines(b[:p], seed=s + k) for k, b in enumerate(bs)])
            with torch.no_grad():
                _, shst, _ = src(sh)
            sh_flat = {k: ssm_flat(shst, i) for k, i in enumerate(S_IDX)}
            mapped = {j: maps["ridge"][j](sh_flat[PAIR[j]]) for j in range(len(T_IDX))}
            L_[f"shuffled@{w}"] += window_loss(tok, p, init_from(mapped, B)).tolist()
        print(f"test batch {s//BATCH}: {time.time()-t0:.0f}s", flush=True)

    Lb = {k: np.array(v) for k, v in L_.items()}  # (n_docs, C) per-byte losses
    for j in range(len(T_IDX)):
        Yt = np.concatenate(test_pairs[j][0])
        Yh = np.concatenate(test_pairs[j][1])
        layer_info[j]["test_r2_ridge"] = r2(Yt, Yh, maps["ridge"][j].my)

    def summarize(Ls):
        summary = {}
        for w in WINDOWS:
            sb, bl, full = Ls[f"statebridge@{w}"], Ls[f"blank@{w}"], Ls["full"]
            row = {a: float(Ls[f"{a}@{w}"].mean()) for a in arms}
            gap = bl.mean() - full.mean()
            row["gap_blank_minus_full"] = float(gap)
            for nm, a in [("quality_kept", "statebridge"), ("quality_kept_oracle", "oracle"),
                          ("quality_kept_ols", "sb_ols"), ("quality_kept_ridge_small", "sb_ridge_small")]:
                row[nm] = float((bl.mean() - Ls[f"{a}@{w}"].mean()) / gap) if abs(gap) > 1e-9 else None
            row["diff_sb_minus_blank"] = boot(sb - bl)
            row["diff_sb_minus_shuffled"] = boot(sb - Ls[f"shuffled@{w}"])
            row["diff_sb_minus_source_alone"] = boot(sb - Ls["source_alone"])
            row["diff_sb_minus_mean_state"] = boot(sb - Ls[f"mean_state@{w}"])
            row["diff_sb_minus_full"] = boot(sb - full)
            row["diff_blank_minus_full"] = boot(bl - full)
            row["diff_oracle_minus_blank"] = boot(Ls[f"oracle@{w}"] - bl)
            row["beats_all_three_controls"] = bool(row["diff_sb_minus_blank"][2] < 0 and row["diff_sb_minus_shuffled"][2] < 0
                                                   and row["diff_sb_minus_source_alone"][2] < 0)
            summary[str(w)] = row
        return summary

    Ls = {k: v.mean(1) for k, v in Lb.items()}
    Lh = {k: v[:, :HEAD].mean(1) for k, v in Lb.items()}
    summary = summarize(Ls)
    summary_head = summarize(Lh)
    per_pos = {k: v.mean(0).tolist() for k, v in Lb.items() if k in ("full", "source_alone") or k.endswith("@1") or k.endswith("@32")}
    res = {
        "protocol": {"T": T, "C": C, "windows": WINDOWS, "primary_small_window": 32, "lambdas": LAMBDAS,
                     "head_bytes": HEAD, "calib_cap": CALIB_CAP, "snap_every": SNAP_EVERY, "snap_from": SNAP_FROM, "pairing": {str(k): v for k, v in PAIR.items()},
                     "units": "nats/byte (mean over the 512 scored bytes, then over docs)"},
        "data": {"n_train_files": len(train), "n_calib_docs": len(cdocs), "n_calib_samples": int(len(did)),
                 "n_calib_fit_samples": int(fit.sum()), "n_calib_val_samples": int(val.sum()),
                 "n_small_calib_docs": int(n_small_docs), "n_small_calib_samples": int(small.sum()),
                 "n_test_docs": len(tdocs), "test_files": names},
        "models": {n: json.load(open(f"train_{n}.json")) for n in ["src", "tgt"]},
        "full_prefill": float(Ls["full"].mean()),
        "source_alone": float(Ls["source_alone"].mean()),
        "sanity_true_state_no_window": float(Ls["sanity_true_state_no_window"].mean()),
        "by_window": summary,
        "by_window_head64": summary_head,
        "head64": {"full_prefill": float(Lh["full"].mean()), "source_alone": float(Lh["source_alone"].mean()),
                   "sanity_true_state_no_window": float(Lh["sanity_true_state_no_window"].mean())},
        "mean_loss_by_position": per_pos,
        "map_r2_per_layer": layer_info,
        "per_doc_losses": {k: v.tolist() for k, v in Ls.items()},
        "per_doc_losses_head64": {k: v.tolist() for k, v in Lh.items()},
        "runtime_s": time.time() - t0,
    }
    for n in res["models"]:
        res["models"][n].pop("log", None)
        res["models"][n]["cfg"] = torch.load(f"ckpt_{n}.pt")["cfg"]
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["full_prefill", "source_alone", "sanity_true_state_no_window"]}))
    for w in WINDOWS:
        r = summary_head[str(w)]
        print("head64", w, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("diff")})
    for w in WINDOWS:
        r = summary[str(w)]
        print(w, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.startswith("diff")})
        print("   ", {k: [round(x, 4) for x in v] for k, v in r.items() if k.startswith("diff")})
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
