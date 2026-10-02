"""StateBridge round 4: a TRAINED bridge in place of the closed-form ridge map.

ROUND 4 PROTOCOL (fixed and committed before any test-document number was computed)

Question. Round 3's ridge maps fit the states well (val R^2 0.57-0.96) but carried almost none of
the bindings (w = 32: bridge 5.07 nats/byte, 5.7% byte acc; source alone 1.41 / 58%; oracle
1.36 / 59%). Is the failure the closed-form state-regression objective, or the approach itself?

Fixed from round 3 (unchanged): models ckpt_{src,tgt}_r3b.pt (frozen, never updated here); task,
T = 864, C = 150, 6 bindings, the 96 test docs (test split, seed 2); windows {1, 8, 32, 128, 256},
primary w = 32; handoff = SSM state only, conv tail zero, attention KV empty; target re-reads
[T-w, T+C-1) and is scored on [T, T+C); layer pairing target SSM j <- source SSM round(3j/7).

Bridges (all per target SSM layer j, input = the paired source SSM layer's flattened state)
- ridge (round 3, re-run as-is): statebridge2.collect + statebridge2.fit_maps on the same 432
  calibration docs (320 synthetic calib-split docs seed 1 + 112 plain calib files), dense map.
- trained (PRIMARY): the same affine map y = std(x) @ W + b, initialised at the ridge solution,
  parametrised in standardised target units (W' = W / sy, b' = 0, y = (z W' + b') * sy + my, sy =
  per-dim calibration std of the target state), all 8 maps (201M params) trained jointly by Adam
  (lr 5e-5, batch 32) on the FROZEN target's downstream loss after the handoff:
      loss = mean CE on the 24 recalled value bytes + mean CE on all bytes after the handoff
             (predictions of bytes p+1 .. T+C-1, p = T - w).
  The window w of each step cycles through {1, 8, 32, 128, 256}.
- trained_mlp (secondary variant): frozen ridge map plus a per-output diagonal gain and bias and a
  small residual MLP, all in standardised target units:
      y = (a * (z W'_ridge) + b' + W2 gelu(W1 z + c1)) * sy + my,
  hidden 128 per target layer, a = 1, b' = 0, W2 = 0 at init (so it starts at the ridge map);
  Adam lr 1e-3, batch 32, same loss and window cycle. (~10.6M trainable params.)

Bridge training data (calibration split only; never test docs)
- train: 2048 synthetic calib-split docs (seed 41); validation: 256 synthetic calib-split docs
  (seed 42), used only for early stopping and diagnostics.
- Source states at T - w for every w are precomputed (the source is frozen).
- Validation every 40 steps (and at step 0 = the ridge init): mean recalled-value CE at w = 32 on
  the 256 val docs. The checkpoint with the best val value CE is kept (step 0 included, so a bridge
  that never improves on val is the ridge map). Early stop after 6 evaluations without improvement
  or at the wall-clock cap (dense 8.5 min, MLP 7 min). The val value CE of the mismatched
  (different-doc) input is logged at each evaluation as a diagnostic only.
- One pre-declared fallback: if the first evaluation after step 0 is worse than step 0 by > 0.5
  nats/byte, training restarts once from the ridge init with lr / 4.
- Calibration-only diagnostics on the val docs at w = 32 (reported, not used for any decision):
  blank, oracle, ridge, ridge mismatched, source alone, and source-own-state (the SOURCE re-reading
  the window from its own true SSM state, conv zero, attention KV empty = how much of the recall
  the source's SSM state itself carries, an upper-bound-ish reference for any bridge).

Test arms (per w in {1, 8, 32, 128, 256}; 96 test docs, read once, in a single final pass)
- trained + w;  trained_mismatched + w: the trained bridge fed the source state of a DIFFERENT
  test doc (next doc in the batch, as round 3) -- the key control;
- trained_mlp + w and trained_mlp_mismatched + w (variant);
- ridge + w and ridge_mismatched + w (round 3 map, re-fitted identically);
- blank + w; oracle (target's own true SSM state) + w; source_own_state + w (diagnostic);
- source alone (full prefill of the source); full target prefill (window-independent).

Metrics (as round 3): per doc, mean CE on the 24 recalled value bytes (primary), per-byte argmax
accuracy, exact-value accuracy (all 4 bytes), CE on all C bytes. Gap closed = (blank - arm) /
(blank - full). Paired bootstrap over test docs (10k resamples, 95% percentile CIs) of the arm
differences, for value CE, value acc and all-bytes CE.

HEADLINE (w = 32, primary = trained dense bridge, value CE): does it beat (a) blank + w,
(b) trained_mismatched + w (a different document's state through the same trained bridge), and
(c) source alone -- each paired-difference 95% CI entirely below 0?  Also: its share of the
oracle's accuracy = value_acc(trained) / value_acc(oracle), and chance/blank-corrected
(acc_trained - acc_blank) / (acc_oracle - acc_blank). The matched-minus-mismatched gap (does the
trained bridge carry the document, or decode generic statistics?) is reported alongside, for
trained vs ridge. Same tests reported for trained_mlp and at every window.

Stages (each run in the foreground):  prep | train dense | train mlp | test.
SB_SMOKE=1 shrinks everything and replaces the test docs by calib-split docs (seed 99), writing
to the scratch dir only; it never reads test documents.
"""
import json
import os
import sys
import time

os.environ.setdefault("SB_SUFFIX", "_r3b")
import numpy as np
import torch
import torch.nn.functional as F

import statebridge2 as sb2
from data import split
from recall import eval_docs, streams

torch.set_num_threads(int(os.environ.get("SB_THREADS", "4")))
SMOKE = bool(os.environ.get("SB_SMOKE"))
OUT = os.environ.get("SB_OUT4", "/mnt/project-files/nebius/statebridge/round4")
TAG = "_r4smoke" if SMOKE else "_r4"
T, C, NB, WINDOWS, PW = sb2.T, sb2.C, sb2.NB, sb2.WINDOWS, sb2.PRIMARY_W
N_TRAIN, N_VAL, N_TEST, N_CAL_SYN = 2048, 256, 96, 320
if SMOKE:
    N_TRAIN, N_VAL, N_TEST, N_CAL_SYN = 64, 32, 16, 24
BATCH, VAL_EVERY, PATIENCE = 32, 40, 6
CFG = {"dense": dict(lr=5e-5, minutes=8.5), "mlp": dict(lr=1e-3, minutes=7.0)}
if SMOKE:
    CFG = {"dense": dict(lr=5e-5, minutes=0.5), "mlp": dict(lr=1e-3, minutes=0.5)}
    VAL_EVERY = 5
HIDDEN = 128
src, tgt = sb2.src, sb2.tgt
for p in list(src.parameters()) + list(tgt.parameters()):
    p.requires_grad_(False)
S_IDX, T_IDX, PAIR = sb2.S_IDX, sb2.T_IDX, sb2.PAIR
NT = len(T_IDX)
PREP = f"states{TAG}.pt"  # gitignored


def tt(bs):
    return sb2.tt(bs)


# ---------------------------------------------------------------- bridges (torch)
class Bridge(torch.nn.Module):
    """kind 'dense': trainable affine map init at ridge; kind 'mlp': frozen ridge + gain/bias + MLP."""

    def __init__(self, ridge, kind):
        super().__init__()
        self.kind = kind
        self.mx = [torch.as_tensor(ridge["mx"][k]).float() for k in range(len(S_IDX))]
        self.sx = [torch.as_tensor(ridge["sx"][k]).float() for k in range(len(S_IDX))]
        self.sy = [torch.as_tensor(ridge["sy"][j]).float() for j in range(NT)]
        self.my = [torch.as_tensor(ridge["my"][j]).float() for j in range(NT)]
        Wn = [(torch.as_tensor(ridge["W"][j]) / torch.as_tensor(ridge["sy"][j])[None, :]).float() for j in range(NT)]
        dy = Wn[0].shape[1]
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(dy)) for _ in range(NT)])
        if kind == "dense":
            self.W = torch.nn.ParameterList([torch.nn.Parameter(w) for w in Wn])
        else:
            self.Wr = Wn  # frozen
            dx = Wn[0].shape[0]
            g = torch.Generator().manual_seed(0)
            self.a = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(dy)) for _ in range(NT)])
            self.W1 = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(dx, HIDDEN, generator=g) / dx ** 0.5) for _ in range(NT)])
            self.c1 = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(HIDDEN)) for _ in range(NT)])
            self.W2 = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(HIDDEN, dy)) for _ in range(NT)])

    def z(self, xs):
        """xs: list over source SSM layers of (B, dx) raw flattened states -> standardised."""
        return [(x - m) / s for x, m, s in zip(xs, self.mx, self.sx)]

    def forward(self, zs):
        out = []
        for j in range(NT):
            z = zs[PAIR[j]]
            if self.kind == "dense":
                u = z @ self.W[j] + self.b[j]
            else:
                u = self.a[j] * (z @ self.Wr[j]) + self.b[j] + F.gelu(z @ self.W1[j] + self.c1[j]) @ self.W2[j]
            out.append(u * self.sy[j] + self.my[j])
        return out


def tgt_init(ys):
    B = ys[0].shape[0]
    st = [None] * len(tgt.pattern)
    for j, i in enumerate(T_IDX):
        m = tgt.layers[i]
        st[i] = (ys[j].reshape(B, m.H, m.P, m.N), None)
    return st


def window_logits(model, tok, p, init):
    lo, _, _ = model(tok[:, p:T + C - 1], init)
    return lo


def loss_fn(lo, tok, p, vpos):
    """lo predicts bytes p+1 .. T+C-1. Returns (value CE mean, all-bytes-after-handoff CE mean)."""
    tg = tok[:, p + 1:T + C]
    ce = F.cross_entropy(lo.reshape(-1, 256), tg.reshape(-1), reduction="none").view(tg.shape)
    return ce[:, -C:][:, vpos].mean(), ce.mean()


# ---------------------------------------------------------------- prep (calibration only)
def prep():
    t0 = time.time()
    S = streams()
    _, calib, _ = split()
    cal_syn, _ = eval_docs(S["calib"], N_CAL_SYN, T, seed=1, nbind=NB)
    cal_plain = [b[:sb2.CAL_CAP] for _, b in calib][: 10 if SMOKE else None]
    cdocs = cal_syn + cal_plain
    Xs, did = sb2.collect(src, cdocs)
    Ys, did2 = sb2.collect(tgt, cdocs)
    assert (did == did2).all()
    print(f"calib: {len(cdocs)} docs, {len(did)} samples; {time.time()-t0:.0f}s", flush=True)
    maps, layer_info = sb2.fit_maps(Xs, Ys, did)
    dense = maps["dense"]
    ridge = {"mx": {}, "sx": {}, "W": {}, "my": {}, "sy": {}}
    for j in range(NT):
        k = PAIR[j]
        ridge["mx"][k], ridge["sx"][k] = dense[j].std.mx, dense[j].std.sx
        ridge["W"][j], ridge["my"][j] = dense[j].W, dense[j].my
        ridge["sy"][j] = Ys[T_IDX[j]].astype(np.float64).std(0) + 1e-6
    # torch tensors (not numpy) so torch.save streams them instead of pickling copies (OOM otherwise)
    ridge = {name: {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in d.items()} for name, d in ridge.items()}
    del Xs, Ys, maps, dense
    print(f"ridge fitted; {time.time()-t0:.0f}s", flush=True)

    tr_docs, vpos = eval_docs(S["calib"], N_TRAIN, T, seed=41, nbind=NB)
    va_docs, _ = eval_docs(S["calib"], N_VAL, T, seed=42, nbind=NB)
    cuts = [T - w for w in WINDOWS] + [T + C - 1]

    def src_states(docs, with_tgt=False):
        Z = {w: [[] for _ in S_IDX] for w in WINDOWS}
        Tt = {w: [[] for _ in T_IDX] for w in WINDOWS}
        for s in range(0, len(docs), 64):
            tok = tt(docs[s:s + 64])
            sst, _ = sb2.chain(src, tok, cuts)
            tst = sb2.chain(tgt, tok, cuts[:-1])[0] if with_tgt else None
            for w in WINDOWS:
                for k, i in enumerate(S_IDX):
                    Z[w][k].append(torch.from_numpy(sb2.ssm_flat(sst[T - w], i)))
                if with_tgt:
                    for j, i in enumerate(T_IDX):
                        Tt[w][j].append(torch.from_numpy(sb2.ssm_flat(tst[T - w], i)))
        Z = {w: [torch.cat(v) for v in Z[w]] for w in WINDOWS}
        Tt = {w: [torch.cat(v) for v in Tt[w]] for w in WINDOWS} if with_tgt else None
        return Z, Tt

    Xtr, _ = src_states(tr_docs)
    Xva, Tva = src_states(va_docs, with_tgt=True)
    print(f"source states precomputed; {time.time()-t0:.0f}s", flush=True)
    torch.save({"ridge": ridge, "ridge_layer_info": layer_info, "vpos": vpos,
                "tr_tok": tt(tr_docs), "va_tok": tt(va_docs), "Xtr": Xtr, "Xva": Xva, "Tva": Tva,
                "n_calib_samples": int(len(did)), "n_calib_docs": [len(cal_syn), len(cal_plain)]}, PREP)
    # calibration-only diagnostics at w = PW on val docs
    D = torch.load(PREP, weights_only=False)
    br = Bridge(ridge, "dense").eval()
    diag = val_diag(D, br, PW, extra=True)
    diag["runtime_s"] = time.time() - t0
    json.dump({"val_diag_w32": diag, "ridge_layer_info": layer_info}, open(f"prep{TAG}.json", "w"), indent=1)
    print(json.dumps(diag), flush=True)


@torch.no_grad()
def val_diag(D, br, w, extra=False, bs=64):
    """Mean value CE / acc on val docs at window w for the bridge (matched and mismatched)."""
    vpos, tok_all, Xva = D["vpos"], D["va_tok"], D["Xva"][w]
    p = T - w
    acc = {}

    def put(name, lo, tok):
        ce, ok = sb2.score(lo[:, -C:], tok)
        acc.setdefault(name, [[], []])
        acc[name][0].append(ce[:, vpos].mean(1))
        acc[name][1].append(ok[:, vpos].float().mean(1))

    for s in range(0, len(tok_all), bs):
        tok = tok_all[s:s + bs]
        zs = br.z([x[s:s + bs] for x in Xva])
        ys = br(zs)
        put("bridge", window_logits(tgt, tok, p, tgt_init(ys)), tok)
        put("bridge_mismatched", window_logits(tgt, tok, p, tgt_init([torch.roll(y, -1, 0) for y in ys])), tok)
        if extra:
            put("blank", window_logits(tgt, tok, p, None), tok)
            put("oracle", window_logits(tgt, tok, p, tgt_init([t[s:s + bs] for t in D["Tva"][w]])), tok)
            B = len(tok)
            sst = [None] * len(src.pattern)
            for k, i in enumerate(S_IDX):
                m = src.layers[i]
                sst[i] = (Xva[k][s:s + bs].reshape(B, m.H, m.P, m.N), None)
            put("source_own_state", window_logits(src, tok, p, sst), tok)
            lo, _, _ = src(tok[:, :T + C - 1])
            put("source_alone", lo, tok)
    return {k: {"value_ce": float(torch.cat(v[0]).mean()), "value_acc": float(torch.cat(v[1]).mean())} for k, v in acc.items()}


# ---------------------------------------------------------------- train (calibration only)
def train(kind):
    D = torch.load(PREP, weights_only=False)
    cfg = CFG[kind]
    log = {"kind": kind, "cfg": cfg, "batch": BATCH, "val_every": VAL_EVERY, "patience": PATIENCE, "attempts": []}
    lr = cfg["lr"]
    for attempt in range(2):
        res = _train(D, kind, lr, cfg["minutes"])
        log["attempts"].append(res["log"])
        ev = res["log"]["evals"]
        if attempt == 0 and len(ev) > 1 and ev[1]["val_value_ce"] > ev[0]["val_value_ce"] + 0.5:
            print(f"fallback: first eval worse by > 0.5; restarting with lr {lr/4}", flush=True)
            lr = lr / 4
            continue
        break
    torch.save(res["best_state"], f"ckpt_bridge_{kind}{TAG}.pt")
    json.dump(log, open(f"train_bridge_{kind}{TAG}.json", "w"), indent=1)


def _train(D, kind, lr, minutes):
    t0 = time.time()
    torch.manual_seed(0)
    br = Bridge(D["ridge"], kind)
    params = [p for p in br.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    try:
        opt = torch.optim.Adam(params, lr=lr, fused=True)
    except Exception:
        opt = torch.optim.Adam(params, lr=lr)
    vpos = D["vpos"]
    tok_all, Xtr = D["tr_tok"], D["Xtr"]
    Ztr = {w: br.z(Xtr[w]) for w in WINDOWS}  # standardisation is fixed (not trained)
    rng = np.random.default_rng(0)

    def evaluate(step):
        br.eval()
        r = val_diag(D, br, PW)
        br.train()
        return {"step": step, "t": round(time.time() - t0, 1), "val_value_ce": r["bridge"]["value_ce"],
                "val_value_acc": r["bridge"]["value_acc"],
                "val_mismatched_value_ce": r["bridge_mismatched"]["value_ce"],
                "val_mismatched_value_acc": r["bridge_mismatched"]["value_acc"]}

    evals = [evaluate(0)]
    print(kind, lr, n_params, evals[-1], flush=True)
    best = (evals[0]["val_value_ce"], 0, {k: v.detach().clone() for k, v in br.state_dict().items()})
    train_log, step, since = [], 0, 0
    order, oi = rng.permutation(len(tok_all)), 0
    while time.time() - t0 < minutes * 60:
        w = WINDOWS[step % len(WINDOWS)]
        p = T - w
        if oi + BATCH > len(order):
            order, oi = rng.permutation(len(tok_all)), 0
        idx = torch.from_numpy(order[oi:oi + BATCH])
        oi += BATCH
        tok = tok_all[idx]
        ys = br([z[idx] for z in Ztr[w]])
        lv, la = loss_fn(window_logits(tgt, tok, p, tgt_init(ys)), tok, p, vpos)
        loss = lv + la
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        train_log.append([step, w, round(float(lv), 4), round(float(la), 4)])
        if step % VAL_EVERY == 0:
            evals.append(evaluate(step))
            recent = np.array(train_log[-VAL_EVERY:])[:, 2:].mean(0)
            print(kind, evals[-1], "train(v,all)", recent.round(3).tolist(), flush=True)
            if evals[-1]["val_value_ce"] < best[0]:
                best = (evals[-1]["val_value_ce"], step, {k: v.detach().clone() for k, v in br.state_dict().items()})
                since = 0
            else:
                since += 1
            if len(evals) == 2 and evals[1]["val_value_ce"] > evals[0]["val_value_ce"] + 0.5:
                break  # fallback check happens in train()
            if since >= PATIENCE:
                break
    return {"best_state": best[2],
            "log": {"lr": lr, "n_params": n_params, "steps": step, "best_step": best[1], "best_val_value_ce": best[0],
                    "evals": evals, "train": train_log, "runtime_s": time.time() - t0}}


# ---------------------------------------------------------------- test (read once)
def boot(d):
    return sb2.boot(d)


def test():
    t0 = time.time()
    D = torch.load(PREP, weights_only=False)
    S = streams()
    if SMOKE:
        docs, vpos = eval_docs(S["calib"], N_TEST, T, seed=99, nbind=NB)
    else:
        docs, vpos = eval_docs(S["test"], N_TEST, T, seed=2, nbind=NB)
    assert (vpos == D["vpos"]).all()
    # ridge exactly as round 3 (numpy float64 Dense maps)
    rg = D["ridge"]
    npf = lambda v: np.asarray(v, dtype=np.float64)
    ridge = {j: sb2.Dense(_Std(npf(rg["mx"][PAIR[j]]), npf(rg["sx"][PAIR[j]])), npf(rg["W"][j]), npf(rg["my"][j])) for j in range(NT)}
    bridges = {}
    for kind, arm in [("dense", "trained"), ("mlp", "trained_mlp")]:
        br = Bridge(rg, kind)
        br.load_state_dict(torch.load(f"ckpt_bridge_{kind}{TAG}.pt"))
        bridges[arm] = br.eval()
    store = {}
    cuts = [T - w for w in WINDOWS] + [T + C - 1]
    for s in range(0, len(docs), sb2.BATCH):
        tok = tt(docs[s:s + sb2.BATCH])
        B = len(tok)
        tst, tlo = sb2.chain(tgt, tok, cuts)
        sst, slo = sb2.chain(src, tok, cuts)
        sb2.add(store, "full", sb2.metrics(*sb2.score(tlo[:, -C:], tok), vpos))
        sb2.add(store, "source_alone", sb2.metrics(*sb2.score(slo[:, -C:], tok), vpos))
        for w in WINDOWS:
            p = T - w
            win = lambda init, model=tgt: sb2.metrics(*sb2.score(window_logits(model, tok, p, init)[:, -C:], tok), vpos)
            with torch.no_grad():
                true = {j: sb2.ssm_flat(tst[p], i) for j, i in enumerate(T_IDX)}
                srcf = {k: sb2.ssm_flat(sst[p], i) for k, i in enumerate(S_IDX)}
                sb2.add(store, f"oracle@{w}", win(sb2.init_from(true, B)))
                sb2.add(store, f"blank@{w}", win(None))
                own = [None] * len(src.pattern)
                for i in S_IDX:
                    own[i] = (sst[p][i][0], None)
                sb2.add(store, f"source_own_state@{w}", win(own, src))
                mapped = {j: ridge[j](srcf[PAIR[j]]) for j in range(NT)}
                sb2.add(store, f"ridge@{w}", win(sb2.init_from(mapped, B)))
                mis = {j: np.roll(v, -1, 0) for j, v in mapped.items()}
                sb2.add(store, f"ridge_mismatched@{w}", win(sb2.init_from(mis, B)))
                xs = [torch.from_numpy(srcf[k]) for k in range(len(S_IDX))]
                for arm, br in bridges.items():
                    ys = br(br.z(xs))
                    sb2.add(store, f"{arm}@{w}", win(tgt_init(ys)))
                    sb2.add(store, f"{arm}_mismatched@{w}", win(tgt_init([torch.roll(y, -1, 0) for y in ys])))
        print(f"test batch {s//sb2.BATCH}: {time.time()-t0:.0f}s", flush=True)

    R = {a: sb2.stack(v) for a, v in store.items()}
    arms = ["oracle", "trained", "trained_mismatched", "trained_mlp", "trained_mlp_mismatched", "ridge",
            "ridge_mismatched", "blank", "source_own_state"]
    comps = [("trained", ["blank", "trained_mismatched", "source_alone", "ridge", "oracle", "full", "trained_mlp"]),
             ("trained_mlp", ["blank", "trained_mlp_mismatched", "source_alone", "ridge", "oracle"]),
             ("ridge", ["blank", "ridge_mismatched", "source_alone", "oracle"]),
             ("oracle", ["blank"]), ("source_own_state", ["blank", "oracle", "source_alone"])]
    by_w = {}
    for w in WINDOWS:
        key = lambda a: a if a in ("source_alone", "full") else f"{a}@{w}"
        row = {"means": {a: {k: float(v.mean()) for k, v in R[f"{a}@{w}"].items()} for a in arms}}
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
                    d[f"diff_{a}_minus_{c}"] = boot(g(a) - g(c))
            if metric == "value_ce":
                for a in ["trained", "trained_mlp", "ridge"]:
                    mis = f"{a}_mismatched"
                    d[f"{a}_beats_all_three"] = bool(d[f"diff_{a}_minus_blank"][2] < 0 and d[f"diff_{a}_minus_{mis}"][2] < 0
                                                     and d[f"diff_{a}_minus_source_alone"][2] < 0)
            row[metric] = d
        m = row["means"]
        row["oracle_acc_share"] = {a: {"ratio": m[a]["value_acc"] / m["oracle"]["value_acc"],
                                       "blank_corrected": (m[a]["value_acc"] - m["blank"]["value_acc"]) /
                                       (m["oracle"]["value_acc"] - m["blank"]["value_acc"])}
                                   for a in ["trained", "trained_mlp", "ridge", "trained_mismatched", "source_own_state"]}
        by_w[str(w)] = row
    trlogs = {k: json.load(open(f"train_bridge_{k}{TAG}.json")) for k in ["dense", "mlp"]}
    for k in trlogs:
        for a in trlogs[k]["attempts"]:
            a["train"] = a["train"]  # kept: [step, w, value CE, all CE]
    res = {
        "protocol": {"T": T, "C": C, "n_bindings": NB, "windows": WINDOWS, "primary_window": PW,
                     "pairing": {str(k): v for k, v in PAIR.items()}, "n_train_calib": N_TRAIN, "n_val_calib": N_VAL,
                     "bridge_cfg": CFG, "batch": BATCH, "hidden_mlp": HIDDEN,
                     "units": "nats/byte; value_ce = mean over the 24 recalled value bytes per doc, then over docs",
                     "code": "statebridge4.py (protocol in docstring)"},
        "data": {"n_test_docs": len(docs), "smoke": SMOKE, "ridge_calib_docs": D["n_calib_docs"],
                 "ridge_calib_samples": D["n_calib_samples"]},
        "models": "ckpt_src_r3b.pt, ckpt_tgt_r3b.pt (frozen)",
        "calib_val_diag_w32": json.load(open(f"prep{TAG}.json"))["val_diag_w32"],
        "bridge_training": trlogs,
        "ridge_layer_info": D["ridge_layer_info"],
        "full_prefill": {k: float(v.mean()) for k, v in R["full"].items()},
        "source_alone": {k: float(v.mean()) for k, v in R["source_alone"].items()},
        "by_window": by_w,
        "per_doc": {a: {k: v.tolist() for k, v in mm.items()} for a, mm in R.items()},
        "runtime_s": time.time() - t0,
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/round4-results.json", "w"), indent=1)
    print("full", res["full_prefill"], "\nsource_alone", res["source_alone"])
    for w in WINDOWS:
        r = by_w[str(w)]
        print(w, {a: round(r["means"][a]["value_ce"], 3) for a in arms})
        print("   acc", {a: round(r["means"][a]["value_acc"], 3) for a in arms})
        print("   ", {k: ([round(x, 3) for x in v] if isinstance(v, list) else (round(v, 3) if isinstance(v, float) else v))
                      for k, v in r["value_ce"].items() if k.startswith("diff_trained") or "beats" in k or k.startswith("diff_ridge")})
        print("   share", {a: {k: round(v, 3) for k, v in s.items()} for a, s in r["oracle_acc_share"].items()})
    print(f"done {time.time()-t0:.0f}s")


class _Std(sb2.Std):
    def __init__(self, mx, sx):
        self.mx, self.sx = mx, sx


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep()
    elif sys.argv[1] == "train":
        train(sys.argv[2])
    elif sys.argv[1] == "test":
        test()
