"""StateBridge round 7, part 2: the round 3-5 protocol, unchanged, on SAME-FAMILY pairs.

ROUND 7 PROTOCOL, BRIDGES AND TEST (fixed in this docstring and in train_family.py's, committed
before any round-7 run, smoke test included)

Question. Do SSM states map across a same-family pair much better than across the independently
trained round 3-6 pair? (Nemotron Nano / Super share family, tokenizer and data; the toy pair did
not.) The targets are built by train_family.py (see its docstring): "grow" (PRIMARY: grown from the
source with function-preserving init, then trained 1,000 more steps on the same stream) and "replay"
(SECONDARY: round 3-6 target architecture, shared init where shapes allow, the source's exact data
order). Source: ckpt_src_r3b.pt, the source of rounds 3-6, frozen. Targets frozen here.

Gate: a pair runs only if its target passed the round-3 gate plus recall bar in train_family.py
(recorded in train_tgt_{pair}_r7.json); otherwise it is reported as failed, with no bridge.

Unchanged from rounds 3-5: task (T = 864, C = 150, 6 bindings); the 96 test docs (test split, seed 2);
windows {1, 8, 32, 128, 256}, primary w = 32; handoff = SSM state only, conv tail zero, attention KV
empty; the target re-reads [T-w, T+C-1) and is scored on [T, T+C); pairing target SSM j <- source
SSM round(3j/7) (for "grow" this is also provenance).

Per pair, in order:
1. Ridge (round 3): statebridge2.collect on the 432 calibration docs (320 synthetic calib-split
   docs, seed 1, + 112 plain calib files), then the dense part of statebridge2.fit_maps copied
   verbatim (per-layer affine ridge, lambda by pooled val R^2, refit on all). The low-rank maps are
   not fitted (no low-rank arm in round 7). Kept as float32 (applied in float32; round 3 used
   float64: negligible).
2. Trained dense bridge (rounds 4-6): initialised exactly at the ridge map, y = (z W' + b) * sy + my,
   W' = W / sy, b = 0; Adam lr 5e-5 (fused when available), batch 32; loss through the frozen target
   = mean CE on the 24 recalled value bytes + mean CE on all bytes after the handoff; window of step s
   = WINDOWS[s % 5]; 2,048 training docs (calib split, seed 41, round 4's), order numpy
   default_rng(0) permutations per epoch; source states computed on the fly (round 6). Validation:
   the 256 calib docs of seed 42 at w = 32, every 40 steps and at step 0; best checkpoint kept (step
   0 included); stop after 6 evaluations without improvement or at 1,000 steps (round 6's cap; round
   4's wall-clock cap is not used because thread counts differ). Round 4's fallback: if the first
   evaluation after step 0 is worse than step 0 by > 0.5 nats/byte, restart once with lr / 4.
3. Calibration-only diagnostics on the val docs at w = 32 (reported, not used for decisions):
   blank, oracle, ridge (+ mismatched), source alone, source's own state.
4. Test (96 test docs, one pass per pair after its bridge is fixed). Arms per w: full target
   prefill; source alone; oracle (target's own SSM state) + w; blank + w; ridge + w and
   ridge_mismatched + w; trained + w and trained_mismatched + w (the same bridge fed the NEXT test
   doc's source state in the batch, as rounds 3-6); source_own_state + w (diagnostic). "grow" only,
   diagnostic: copy + w and copy_mismatched + w, a zero-parameter map that puts the source state into
   the copied heads (0..7) of the copied layers and the calibration-mean target state everywhere else.

Metrics (rounds 4-6): per doc, mean CE on the 24 recalled value bytes (primary), per-byte argmax
accuracy, exact-value accuracy, CE on all C bytes. Gap closed = (blank - arm) / (blank - full).
Paired bootstrap over test docs (10k resamples, 95% percentile CIs). Also, against the independent
pair (round 4's per-doc test results in round4-results.json: same 96 docs, same source, same
windows): paired differences pair-minus-round-4 for trained, ridge and oracle, and the change of the
document-specific gap ((matched - mismatched) - (r4 matched - r4 mismatched)). A built-in check:
source alone must reproduce round 4's per-doc values.

HEADLINE (per pair, w = 32, trained dense bridge, value CE): does it beat (a) blank + w, (b)
trained_mismatched + w and (c) source alone, each paired CI entirely below 0? Reported with it:
oracle share of accuracy (raw and blank-corrected), the document-specific part of the gain, and the
paired difference vs round 4's trained bridge in accuracy and CE. Same for ridge.
Pre-declared reading (grow is primary, replay secondary):
 (i)   trained beats all three -> same-family states map across at toy scale; the round 3-6
       negative result is specific to independently trained pairs.
 (ii)  not (i), but blank-corrected oracle share >= 0.5 and the accuracy gain over round 4 has its CI
       above 0 -> the family relation helps a lot, but the handoff still loses to the source alone.
 (iii) not (ii), but the accuracy gain over round 4 has its CI above 0 -> the family relation helps a
       little.
 (iv)  otherwise -> sharing family (as built here) does not rescue the handoff; the main transfer
       caveat is weakened.

Operation: python3 statebridge7.py [grow|replay|all]   (default all = grow then replay; chdirs here)
- Threads: SB_THREADS (default 2). Log: round7.log; status: round7-status.json (here and in the
  results dir); state: state_r7.json; results: /mnt/project-files/nebius/statebridge/round7/
  round7-results.json, rewritten after every pair's test pass.
- Disk is shared and tight, so large files are written ONLY when at least SB7_MIN_FREE_GB (default
  6) GB would stay free afterwards: the best bridge (ckpt_bridge_{pair}_r7.pt, 0.8 GB, after
  training) and a full training resume checkpoint (3.2 GB, every 15 min). Otherwise they live in RAM
  only. The ridge is never written (0.8 GB) and is not held during bridge training (memory): it is
  refitted for the test pass and on a relaunch (dense fit about 4 min at 2 threads, deterministic;
  the per-layer checksum must match the first fit within 1e-4 relative, else the run stops).
  Resume: a tested pair is skipped; otherwise training resumes from the resume checkpoint, or the
  saved best bridge is used, or (neither on disk) the bridge is retrained from the ridge init
  (deterministic). A saved best bridge is removed after the pair's test pass unless SB7_KEEP=1. A
  second live instance is refused.
- Memory (estimate from component sizes; round 6 measured 4.4 GB anonymous for the same bridge
  training): about 4.5 GB peak (bridge 0.8 + grads 0.8 + Adam 1.6 + best copy 0.8 GB + activations;
  the ridge fit about 3.5 GB). Disk: no large file needed.
SB7_SMOKE=1 (code test only): TINY models from train_family.py's smoke mode (the target's gate is
not blocking there), tiny sizes and caps, plain SGD instead of Adam, test docs replaced by calib-split
docs (seed 99), outputs under the scratch dir; never reads test documents.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("SB_THREADS", "2")
NTH = int(os.environ["SB_THREADS"])
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = str(NTH)
os.environ.pop("SB_SMOKE", None)
os.environ["SB_SUFFIX"] = "_r3b"

import ctypes  # noqa: E402

try:
    _LIBC = ctypes.CDLL("libc.so.6")
    _LIBC.mallopt(-8, 2)  # M_ARENA_MAX = 2
except (OSError, AttributeError):
    _LIBC = None
import json  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(NTH)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
import statebridge4 as sb4  # noqa: E402  (loads sb2 and the frozen r3b models)
import train_family as tf  # noqa: E402
from data import split  # noqa: E402
from recall import eval_docs, streams  # noqa: E402

sb2 = sb4.sb2
torch.set_num_threads(NTH)
SMOKE = tf.SMOKE
WD, OUT = tf.WD, tf.OUT
T, C, NB, WINDOWS, PW = sb2.T, sb2.C, sb2.NB, sb2.WINDOWS, sb2.PRIMARY_W
src = tf.load_ckpt(tf.SRC_CKPT).eval() if SMOKE else sb2.src
for _p in src.parameters():
    _p.requires_grad_(False)
S_IDX = src.ssm_idx
PAIRS = ["grow", "replay"]
N_TRAIN, N_VAL, N_TEST, N_CAL_SYN, N_CAL_PLAIN = 2048, 256, 96, 320, None
BATCH, LR, VAL_EVERY, PATIENCE, MAX_STEPS, DATA_SEED = 32, 5e-5, 40, 6, 1000, 0
SAVE_MIN, LOG_EVERY = 15.0, 20
if SMOKE:
    N_TRAIN, N_VAL, N_TEST, N_CAL_SYN, N_CAL_PLAIN = 64, 32, 16, 24, 10
    MAX_STEPS, VAL_EVERY, LOG_EVERY = 4, 2, 2
MIN_FREE = float(os.environ.get("SB7_MIN_FREE_GB", "6")) * 1e9
KEEP = bool(os.environ.get("SB7_KEEP"))
LOG = f"{WD}/round7.log"
STATE = f"{WD}/state_r7.json"
STATUS_LOCAL = f"{WD}/round7-status.json"
STATUS_OUT = f"{OUT}/round7-status.json"
RESULTS = f"{OUT}/round7-results.json"
LOCK = f"{WD}/r7.pid"
R4_RESULTS = "/mnt/project-files/nebius/statebridge/round4/round4-results.json"
best_path = lambda p: f"{WD}/ckpt_bridge_{p}_r7.pt"  # noqa: E731
resume_path = lambda p: f"{WD}/ckpt_resume_bridge_{p}_r7.pt"  # noqa: E731
STATUS = {}
mem = tf.mem


def persist_ok(nbytes):
    return shutil.disk_usage(WD).free - nbytes >= MIN_FREE


def trim():
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def log(*a):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


atomic_json = tf.atomic_json


def write_status(**kw):
    STATUS.update(kw)
    STATUS["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS["elapsed_s"] = round(time.time() - STATUS["t_start"], 1)
    STATUS["memory_gb"] = mem()
    STATUS["disk_free_gb"] = round(shutil.disk_usage(WD).free / 1e9, 2)
    atomic_json(STATUS_LOCAL, STATUS)
    try:
        os.makedirs(OUT, exist_ok=True)
        atomic_json(STATUS_OUT, STATUS)
        shutil.copyfile(LOG, f"{OUT}/round7.log")
    except Exception as e:
        print("warning: could not mirror status/log to", OUT, repr(e), flush=True)


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"pairs": {}}


def take_lock():
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip())
            if pid != os.getpid() and os.path.exists(f"/proc/{pid}") and \
                    "statebridge7" in open(f"/proc/{pid}/cmdline").read():
                sys.exit(f"another statebridge7 run is live (pid {pid}); not starting")
        except (ValueError, OSError):
            pass
    open(LOCK, "w").write(str(os.getpid()))


# ---------------------------------------------------------------- models and helpers
def load_target(pair):
    m = tf.load_ckpt(f"{WD}/ckpt_tgt_{pair}_r7.pt").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def tgt_init(tgt, ys):
    B = ys[0].shape[0]
    st = [None] * len(tgt.pattern)
    for j, i in enumerate(tgt.ssm_idx):
        m = tgt.layers[i]
        st[i] = (ys[j].reshape(B, m.H, m.P, m.N), None)
    return st


def win_logits(model, tok, p, init):
    lo, _, _ = model(tok[:, p:T + C - 1], init)
    return lo


@torch.no_grad()
def src_x(tok, w):
    _, st, _ = src(tok[:, :T - w])
    return [st[i][0].reshape(len(tok), -1) for i in S_IDX]


def make_bridge(ridge):
    return sb4.Bridge(ridge, "dense")


def copy_map(tgt, xs, my):
    """grow pair diagnostic: source state into the copied heads of the copied layers, calib-mean
    target state elsewhere. xs: list over source SSM layers of (B, dx) flattened states."""
    B = xs[0].shape[0]
    out = []
    for j, i in enumerate(tgt.ssm_idx):
        m = tgt.layers[i]
        h = my[j].float().view(1, m.H, m.P, m.N).repeat(B, 1, 1, 1)
        if i in tf.GROW_PROV:
            k = S_IDX.index(tf.GROW_PROV[i])
            sl = src.layers[S_IDX[k]]
            h[:, :sl.H] = xs[k].view(B, sl.H, sl.P, sl.N)
        out.append(h.reshape(B, -1))
    return out


# ---------------------------------------------------------------- prep: ridge + val slice
def prep(pair, tgt, P):
    t0 = time.time()
    write_status(phase="prep", pair=pair)
    S = streams()
    _, calib, _ = split()
    cal_syn, _ = eval_docs(S["calib"], N_CAL_SYN, T, seed=1, nbind=NB)
    cal_plain = [b[:sb2.CAL_CAP] for _, b in calib][:N_CAL_PLAIN]
    cdocs = cal_syn + cal_plain
    Xs, did = sb2.collect(src, cdocs)
    Ys, did2 = sb2.collect(tgt, cdocs)
    assert (did == did2).all()
    log(f"{pair} calib: {len(cdocs)} docs, {len(did)} samples; {time.time() - t0:.0f}s", mem())
    ridge, layer_info = fit_dense(Xs, Ys, did, tgt)
    del Xs, Ys
    trim()
    ridge["checksum"] = [float(w.double().abs().sum()) for w in ridge["W"]]
    fit = {"layer_info": layer_info, "n_calib_samples": int(len(did)), "n_calib_docs": [len(cal_syn), len(cal_plain)],
           "ridge_s": round(time.time() - t0, 1), "checksum": ridge["checksum"]}
    if "prep" in P:  # relaunch: the refit must reproduce the first fit
        a, b = np.array(P["prep"]["checksum"]), np.array(ridge["checksum"])
        rel = float(np.abs(a - b).max() / np.abs(a).max())
        P.setdefault("refits", []).append({"max_rel_checksum_diff": rel, "ridge_s": fit["ridge_s"]})
        if rel > 1e-4:
            raise RuntimeError(f"{pair}: ridge refit differs from the first fit (rel {rel:.2e})")
        log(f"{pair} ridge refitted; checksum matches the first fit (rel {rel:.1e}); {time.time() - t0:.0f}s", mem())
    else:
        P["prep"] = fit
        log(f"{pair} ridge fitted; {time.time() - t0:.0f}s", mem())
    return ridge


def fit_dense(Xs, Ys, did, tgt):
    """The dense part of statebridge2.fit_maps, verbatim in its arithmetic (float64), with each map
    converted to float32 torch as soon as it is fitted (memory)."""
    T_IDX, NT = tgt.ssm_idx, len(tgt.ssm_idx)
    val = did % 5 == 4
    fit = ~val
    ridge = {"mx": [None] * len(S_IDX), "sx": [None] * len(S_IDX), "W": [None] * NT, "my": [None] * NT, "sy": [None] * NT}
    info = [dict() for _ in T_IDX]
    f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
    for k, si in enumerate(S_IDX):
        js = [j for j in range(NT) if sb2.PAIR[j] == k]
        if not js:
            continue
        t0 = time.time()
        X = Xs[si].astype(np.float64)
        sf, sa = sb2.Std(X[fit]), sb2.Std(X)
        Zf, Zv, Za = sf(X[fit]), sf(X[val]), sa(X)
        del X
        ef, Vf = np.linalg.eigh(Zf.T @ Zf)
        ea, Va = np.linalg.eigh(Za.T @ Za)
        ridge["mx"][k], ridge["sx"][k] = f32(sa.mx), f32(sa.sx)
        for j in js:
            Y = Ys[T_IDX[j]].astype(np.float64)
            myf, mya = Y[fit].mean(0), Y.mean(0)
            VtZY = Vf.T @ (Zf.T @ (Y[fit] - myf))
            ZvV = Zv @ Vf
            path = {lam: sb2.r2(Y[val], ZvV @ (VtZY / (ef + lam * len(Zf))[:, None]) + myf, myf) for lam in sb2.LAMBDAS}
            lam = max(path, key=path.get)
            W = Va @ ((Va.T @ (Za.T @ (Y - mya))) / (ea + lam * len(Za))[:, None])
            ins = sb2.r2(Y, Za @ W + mya, mya)
            ridge["W"][j], ridge["my"][j], ridge["sy"][j] = f32(W), f32(mya), f32(Y.std(0) + 1e-6)
            info[j] = {"tgt_layer": T_IDX[j], "tgt_ssm_j": j, "src_layer": si, "src_ssm_k": k, "dense_lambda": lam,
                       "dense_val_r2": path[lam], "dense_val_r2_path": {str(a): b for a, b in path.items()},
                       "dense_calib_r2_insample": ins, "src_dim": Za.shape[1], "tgt_dim": Y.shape[1]}
            print(json.dumps({a: b for a, b in info[j].items() if "path" not in a}), flush=True)
            del Y, W, VtZY, ZvV
        del Zf, Zv, Za, Vf, Va, ef, ea
        trim()
        print(f"source layer {si}: {time.time() - t0:.0f}s", flush=True)
    return ridge, info


def val_slice(tgt):
    S = streams()
    va_docs, vpos = eval_docs(S["calib"], N_VAL, T, seed=42, nbind=NB)
    tok = sb2.tt(va_docs)
    Xva, Tva = [], []
    for s in range(0, len(tok), 64):
        tb = tok[s:s + 64]
        Xva.append(src_x(tb, PW))
        with torch.no_grad():
            _, st, _ = tgt(tb[:, :T - PW])
        Tva.append([st[i][0].reshape(len(tb), -1) for i in tgt.ssm_idx])
    Xva = [torch.cat([x[k] for x in Xva]) for k in range(len(S_IDX))]
    Tva = [torch.cat([x[j] for x in Tva]) for j in range(len(tgt.ssm_idx))]
    return {"vpos": vpos, "va_tok": tok, "Xva": Xva, "Tva": Tva}


@torch.no_grad()
def val_diag(V, br, tgt, extra=False, bs=64):
    vpos, tok_all, Xva = V["vpos"], V["va_tok"], V["Xva"]
    p = T - PW
    acc = {}

    def put(name, lo, tok):
        ce, ok = sb2.score(lo[:, -C:], tok)
        acc.setdefault(name, [[], []])
        acc[name][0].append(ce[:, vpos].mean(1))
        acc[name][1].append(ok[:, vpos].float().mean(1))

    for s in range(0, len(tok_all), bs):
        tok = tok_all[s:s + bs]
        ys = br(br.z([x[s:s + bs] for x in Xva]))
        put("bridge", win_logits(tgt, tok, p, tgt_init(tgt, ys)), tok)
        put("bridge_mismatched", win_logits(tgt, tok, p, tgt_init(tgt, [torch.roll(y, -1, 0) for y in ys])), tok)
        if extra:
            put("blank", win_logits(tgt, tok, p, None), tok)
            put("oracle", win_logits(tgt, tok, p, tgt_init(tgt, [t[s:s + bs] for t in V["Tva"]])), tok)
            B = len(tok)
            sst = [None] * len(src.pattern)
            for k, i in enumerate(S_IDX):
                m = src.layers[i]
                sst[i] = (Xva[k][s:s + bs].reshape(B, m.H, m.P, m.N), None)
            put("source_own_state", win_logits(src, tok, p, sst), tok)
            lo, _, _ = src(tok[:, :T + C - 1])
            put("source_alone", lo, tok)
    return {k: {"value_ce": float(torch.cat(v[0]).mean()), "value_acc": float(torch.cat(v[1]).mean())} for k, v in acc.items()}


# ---------------------------------------------------------------- bridge training (calibration only)
def make_opt(br, scale):
    if SMOKE:
        return torch.optim.SGD(br.parameters(), lr=LR * scale)
    try:
        return torch.optim.Adam(br.parameters(), lr=LR * scale, fused=True)
    except Exception:
        return torch.optim.Adam(br.parameters(), lr=LR * scale)


def train_bridge(pair, tgt, ridge, V, P, st):
    t0 = time.time()
    S_ = streams()
    tr_docs, vpos = eval_docs(S_["calib"], N_TRAIN, T, seed=41, nbind=NB)
    assert (vpos == V["vpos"]).all()
    tok_all = sb2.tt(tr_docs).to(torch.uint8)
    n = len(tok_all)
    rpath = resume_path(pair)
    br = make_bridge(ridge)
    ridge["W"] = None  # not held during training (memory); refitted for the fallback and the test
    trim()
    if os.path.exists(rpath):
        R = torch.load(rpath, weights_only=False)
        S = R["S"]
        br.load_state_dict(R["model"])
        opt = make_opt(br, S["scale"])
        opt.load_state_dict(R["opt"])
        best_state = R["best_state"]
        rng = np.random.default_rng()
        rng.bit_generator.state = S["rng"]
        del R
        log(f"{pair}: bridge resumed at step {S['step']}", mem())
    else:
        S = {"step": 0, "t_acc": 0.0, "evals": [], "train": [], "since": 0, "best": None, "scale": 1.0,
             "attempts": [], "oi": 0, "order": None, "n_docs": n, "n_params": sum(p.numel() for p in br.parameters()),
             "resume_saves": 0, "resume_skipped_low_disk": 0}
        opt = make_opt(br, 1.0)
        rng = np.random.default_rng(DATA_SEED)
        best_state = None
        log(f"{pair}: bridge training start, {n} docs, {S['n_params']} params", mem())
    br.train()
    t_seg, t_base, t_saved = time.time(), S["t_acc"], time.time()
    elapsed = lambda: t_base + time.time() - t_seg  # noqa: E731

    def status():
        sp = S.get("t_acc_step")
        write_status(phase="train", pair=pair, step=S["step"], max_steps=MAX_STEPS, best=S["best"], since_best=S["since"],
                     sec_per_step=sp, eta_upper_bound_s=None if sp is None else round((MAX_STEPS - S["step"]) * sp * 1.1),
                     last_eval=S["evals"][-1] if S["evals"] else None)

    def evaluate():
        br.eval()
        r = val_diag(V, br, tgt)
        br.train()
        trim()
        e = {"step": S["step"], "t": round(elapsed(), 1), "val_value_ce": r["bridge"]["value_ce"],
             "val_value_acc": r["bridge"]["value_acc"], "val_mismatched_value_ce": r["bridge_mismatched"]["value_ce"],
             "val_mismatched_value_acc": r["bridge_mismatched"]["value_acc"], "rss_anon_gb": mem().get("RssAnon")}
        S["evals"].append(e)
        log(f"{pair} eval", e)
        return e

    snapshot = lambda: {kk: v.detach().clone() for kk, v in br.state_dict().items()}  # noqa: E731

    def keep_best():
        for kk, v in br.state_dict().items():
            best_state[kk].copy_(v.detach())

    if S["step"] == 0 and not S["evals"]:
        e = evaluate()
        S["best"], best_state = [e["val_value_ce"], 0], snapshot()
        status()
    stop = None
    t_steps, n_steps = 0.0, 0
    while True:
        if S["step"] >= MAX_STEPS:
            stop = "cap"
            break
        ts = time.time()
        w = WINDOWS[S["step"] % len(WINDOWS)]
        p = T - w
        if S["order"] is None or S["oi"] + BATCH > len(S["order"]):
            S["order"], S["oi"] = rng.permutation(n), 0
        idx = torch.from_numpy(S["order"][S["oi"]:S["oi"] + BATCH])
        S["oi"] += BATCH
        tok = tok_all[idx].long()
        ys = br(br.z(src_x(tok, w)))
        lv, la = sb4.loss_fn(win_logits(tgt, tok, p, tgt_init(tgt, ys)), tok, p, vpos)
        loss = lv + la
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        S["step"] += 1
        S["train"].append([S["step"], w, round(float(lv.detach()), 4), round(float(la.detach()), 4)])
        t_steps += time.time() - ts
        n_steps += 1
        S["t_acc_step"] = round(t_steps / n_steps, 2)
        if S["step"] % 5 == 0:
            trim()
        if S["step"] % LOG_EVERY == 0:
            recent = np.array(S["train"][-LOG_EVERY:])[:, 2:].mean(0)
            log(f"{pair} step {S['step']} train(value, all) {recent.round(3).tolist()} {S['t_acc_step']} s/step "
                f"t_acc {elapsed():.0f}s")
        if S["step"] % VAL_EVERY == 0:
            e = evaluate()
            first = [x for x in S["evals"] if x["step"] > 0]
            if len(first) == 1 and S["scale"] == 1.0 and e["val_value_ce"] > S["evals"][0]["val_value_ce"] + 0.5:
                log(f"{pair} fallback: first eval worse by > 0.5; restarting from the ridge init with lr / 4")
                S["attempts"].append({"evals": S["evals"], "train": S["train"]})
                del br, opt
                trim()
                ridge.update(prep(pair, tgt, P))
                br = make_bridge(ridge).train()
                ridge["W"] = None
                trim()
                opt = make_opt(br, 0.25)
                rng = np.random.default_rng(DATA_SEED)
                S.update(step=0, evals=[S["evals"][0]], train=[], since=0, scale=0.25, order=None, oi=0)
                continue
            if e["val_value_ce"] < S["best"][0]:
                S["best"], S["since"] = [e["val_value_ce"], S["step"]], 0
                keep_best()
            else:
                S["since"] += 1
            if S["since"] >= PATIENCE:
                stop = "patience"
                break
            if not SMOKE and S["step"] < MAX_STEPS and time.time() - t_saved >= SAVE_MIN * 60:
                t_saved = time.time()
                if persist_ok(3.4e9):
                    S["t_acc"], S["rng"] = elapsed(), rng.bit_generator.state
                    torch.save({"S": S, "model": br.state_dict(), "opt": opt.state_dict(), "best_state": best_state},
                               rpath + ".tmp")
                    os.replace(rpath + ".tmp", rpath)
                    S["resume_saves"] += 1
                    log(f"{pair} resume checkpoint saved at step {S['step']}")
                else:
                    S["resume_skipped_low_disk"] += 1
                    log(f"{pair} resume checkpoint skipped (low disk: keeps >= {MIN_FREE / 1e9:.1f} GB free for other jobs)")
            status()
    S["t_acc"] = elapsed()
    saved = False
    if persist_ok(0.9e9):
        torch.save(best_state, best_path(pair) + ".tmp")
        os.replace(best_path(pair) + ".tmp", best_path(pair))
        saved = True
    tl = {kk: v for kk, v in S.items() if kk not in ("order", "rng")}
    tl.update(stop=stop, best_step=S["best"][1], best_val_value_ce=S["best"][0], lr=LR * S["scale"], batch=BATCH,
              val_every=VAL_EVERY, patience=PATIENCE, max_steps=MAX_STEPS, data_seed=DATA_SEED, memory_gb=mem(),
              wall_s=round(time.time() - t0, 1), optimizer="SGD (smoke only)" if SMOKE else "Adam",
              best_saved_to_disk=saved)
    P["trained"] = tl
    atomic_json(STATE, st)
    if os.path.exists(rpath):
        os.remove(rpath)
    log(f"{pair} TRAINING_DONE: stop {stop} at step {S['step']}, best {S['best']}, t_acc {S['t_acc']:.0f}s, "
        f"best saved to disk: {saved}", mem())
    del br, opt
    trim()
    return best_state


# ---------------------------------------------------------------- test (read once per pair)
def test_pair(pair, tgt, ridge, best_state, P, st):
    t0 = time.time()
    write_status(phase="test", pair=pair)
    P.setdefault("test_passes", []).append({"started": time.strftime("%Y-%m-%d %H:%M:%S")})
    atomic_json(STATE, st)
    if len(P["test_passes"]) > 1:
        log(f"{pair}: NOTE a previous test pass did not finish; redoing it")
    S = streams()
    if SMOKE:
        docs, vpos = eval_docs(S["calib"], N_TEST, T, seed=99, nbind=NB)
    else:
        docs, vpos = eval_docs(S["test"], N_TEST, T, seed=2, nbind=NB)
    my = ridge["my"]
    bridges = {"ridge": make_bridge(ridge).eval()}  # step-0 bridge == the ridge map (float32)
    trained = make_bridge(ridge)
    ridge["W"] = None
    trained.load_state_dict(best_state)
    best_state.clear()
    bridges["trained"] = trained.eval()
    trim()
    store = {}
    cuts = [T - w for w in WINDOWS] + [T + C - 1]
    for s in range(0, len(docs), sb2.BATCH):
        tok = sb2.tt(docs[s:s + sb2.BATCH])
        B = len(tok)
        with torch.no_grad():
            tst, tlo = sb2.chain(tgt, tok, cuts)
            sst, slo = sb2.chain(src, tok, cuts)
        sb2.add(store, "full", sb2.metrics(*sb2.score(tlo[:, -C:], tok), vpos))
        sb2.add(store, "source_alone", sb2.metrics(*sb2.score(slo[:, -C:], tok), vpos))
        for w in WINDOWS:
            p = T - w
            with torch.no_grad():
                win = lambda init, model=tgt: sb2.metrics(*sb2.score(win_logits(model, tok, p, init)[:, -C:], tok), vpos)  # noqa: E731
                true = [tst[p][i][0].reshape(B, -1) for i in tgt.ssm_idx]
                sb2.add(store, f"oracle@{w}", win(tgt_init(tgt, true)))
                sb2.add(store, f"blank@{w}", win(None))
                own = [None] * len(src.pattern)
                for i in S_IDX:
                    own[i] = (sst[p][i][0], None)
                sb2.add(store, f"source_own_state@{w}", win(own, src))
                xs = [sst[p][i][0].reshape(B, -1) for i in S_IDX]
                for arm, br in bridges.items():
                    ys = br(br.z(xs))
                    sb2.add(store, f"{arm}@{w}", win(tgt_init(tgt, ys)))
                    sb2.add(store, f"{arm}_mismatched@{w}", win(tgt_init(tgt, [torch.roll(y, -1, 0) for y in ys])))
                if pair == "grow":
                    ys = copy_map(tgt, xs, my)
                    sb2.add(store, f"copy@{w}", win(tgt_init(tgt, ys)))
                    sb2.add(store, f"copy_mismatched@{w}", win(tgt_init(tgt, [torch.roll(y, -1, 0) for y in ys])))
        log(f"{pair} test batch {s // sb2.BATCH}: {time.time() - t0:.0f}s", mem())
    P["perdoc"] = {a: {m: x.tolist() for m, x in sb2.stack(v).items()} for a, v in store.items()}
    P["tested"] = True
    P["test_passes"][-1]["finished_s"] = round(time.time() - t0, 1)
    atomic_json(STATE, st)
    log(f"{pair} TEST_DONE in {time.time() - t0:.0f}s")
    del bridges, trained
    trim()
    if not KEEP and os.path.exists(best_path(pair)):
        os.remove(best_path(pair))
        log(f"{pair}: removed the saved best bridge (disk; SB7_KEEP=1 keeps it)")


# ---------------------------------------------------------------- analysis
def sdiv(a, b):
    return None if b == 0 else a / b


def load_r4():
    if SMOKE or not os.path.exists(R4_RESULTS):
        return None
    d = json.load(open(R4_RESULTS))["per_doc"]
    return {a: {m: np.array(v) for m, v in mm.items()} for a, mm in d.items()}


def analyse_pair(pair, P, R4):
    R = {a: {m: np.array(v) for m, v in d.items()} for a, d in P["perdoc"].items()}
    boot = sb2.boot
    bridges = ["trained", "ridge"] + (["copy"] if f"copy@{PW}" in R else [])
    arms = ["oracle", "blank", "source_own_state"] + [x for b in bridges for x in (b, f"{b}_mismatched")]
    checks = {}
    if R4 is not None:
        checks["source_alone_vs_round4_max_abs_diff_value_ce"] = float(np.abs(R["source_alone"]["value_ce"] -
                                                                             R4["source_alone"]["value_ce"]).max())
    by_w = {}
    for w in WINDOWS:
        key = lambda a: a if a in ("source_alone", "full") else f"{a}@{w}"  # noqa: E731
        g = lambda a, m: R[key(a)][m]  # noqa: E731
        row = {"means": {a: {m: float(v.mean()) for m, v in R[key(a)].items()} for a in arms + ["source_alone", "full"]}}
        for metric in ["value_ce", "value_acc", "all_ce"]:
            d = {}
            if metric != "value_acc":
                gap = g("blank", metric).mean() - g("full", metric).mean()
                d["gap_blank_minus_full"] = float(gap)
                for a in arms:
                    d[f"gap_closed_{a}"] = sdiv(float(g("blank", metric).mean() - g(a, metric).mean()), float(gap))
            for b in bridges:
                for c in ["blank", f"{b}_mismatched", "source_alone", "oracle", "full"]:
                    d[f"diff_{b}_minus_{c}"] = boot(g(b, metric) - g(c, metric))
            d["diff_trained_minus_ridge"] = boot(g("trained", metric) - g("ridge", metric))
            d["diff_oracle_minus_blank"] = boot(g("oracle", metric) - g("blank", metric))
            d["diff_oracle_minus_source_alone"] = boot(g("oracle", metric) - g("source_alone", metric))
            d["diff_source_own_state_minus_oracle"] = boot(g("source_own_state", metric) - g("oracle", metric))
            if R4 is not None and f"trained@{w}" in R4:
                r4 = lambda a: R4[f"{a}@{w}"][metric]  # noqa: E731
                for b in ["trained", "ridge", "oracle"]:
                    d[f"diff_{b}_minus_r4_{b}"] = boot(g(b, metric) - r4(b))
                for b in ["trained", "ridge"]:
                    d[f"docgap_change_{b}_vs_r4"] = boot((g(b, metric) - g(f"{b}_mismatched", metric)) -
                                                         (r4(b) - r4(f"{b}_mismatched")))
            if metric == "value_ce":
                for b in bridges:
                    d[f"{b}_beats_all_three"] = bool(d[f"diff_{b}_minus_blank"][2] < 0 and d[f"diff_{b}_minus_{b}_mismatched"][2] < 0
                                                     and d[f"diff_{b}_minus_source_alone"][2] < 0)
            row[metric] = d
        m = row["means"]
        row["oracle_acc_share"] = {a: {"ratio": sdiv(m[a]["value_acc"], m["oracle"]["value_acc"]),
                                       "blank_corrected": sdiv(m[a]["value_acc"] - m["blank"]["value_acc"],
                                                               m["oracle"]["value_acc"] - m["blank"]["value_acc"])}
                                   for a in arms if a not in ("oracle", "blank")}
        row["doc_specific"] = {}
        for b in bridges:
            ce_gain = m["blank"]["value_ce"] - m[b]["value_ce"]
            ds = m[f"{b}_mismatched"]["value_ce"] - m[b]["value_ce"]
            row["doc_specific"][b] = {"ce_gain_over_blank": ce_gain, "ce_matched_minus_mismatched": -ds,
                                      "frac_of_ce_gain_doc_specific": sdiv(ds, ce_gain),
                                      "acc_matched_minus_mismatched": m[b]["value_acc"] - m[f"{b}_mismatched"]["value_acc"]}
        if R4 is not None and f"trained@{w}" in R4:
            r4m = {a: float(R4[f"{a}@{w}"]["value_acc"].mean()) for a in ["trained", "ridge", "oracle", "blank"]}
            row["round4_reference"] = {"value_acc": r4m, "value_ce": {a: float(R4[f"{a}@{w}"]["value_ce"].mean())
                                                                      for a in ["trained", "ridge", "oracle", "blank"]},
                                       "trained_oracle_share_blank_corrected": sdiv(r4m["trained"] - r4m["blank"], r4m["oracle"] - r4m["blank"])}
        by_w[str(w)] = row
    r = by_w[str(PW)]
    sh = r["oracle_acc_share"]["trained"]["blank_corrected"]
    acc_r4 = r["value_acc"].get("diff_trained_minus_r4_trained")
    better = bool(acc_r4 is not None and acc_r4[1] > 0)
    if r["value_ce"]["trained_beats_all_three"]:
        reading = "(i) trained beats all three: same-family states map across at toy scale"
    elif sh is not None and sh >= 0.5 and better:
        reading = "(ii) large help from the family relation, but the handoff still loses to the source alone"
    elif better:
        reading = "(iii) the family relation helps a little (accuracy above round 4), share < 0.5 or not all controls"
    else:
        reading = "(iv) no significant accuracy gain over the independent pair: family (as built) does not rescue the handoff"
    head = {"pair": pair, "w": PW,
            "trained_beats_all_three": r["value_ce"]["trained_beats_all_three"],
            "ridge_beats_all_three": r["value_ce"]["ridge_beats_all_three"],
            "trained": r["means"]["trained"], "trained_mismatched": r["means"]["trained_mismatched"],
            "ridge": r["means"]["ridge"], "oracle": r["means"]["oracle"], "source_alone": r["means"]["source_alone"],
            "blank": r["means"]["blank"],
            "trained_value_ce_diffs": {c: r["value_ce"][f"diff_trained_minus_{c}"] for c in ["blank", "trained_mismatched", "source_alone"]},
            "trained_oracle_share": sh, "trained_oracle_share_raw": r["oracle_acc_share"]["trained"]["ratio"],
            "ridge_oracle_share": r["oracle_acc_share"]["ridge"]["blank_corrected"],
            "trained_minus_r4_value_acc": acc_r4, "trained_minus_r4_value_ce": r["value_ce"].get("diff_trained_minus_r4_trained"),
            "ridge_minus_r4_value_acc": r["value_acc"].get("diff_ridge_minus_r4_ridge"),
            "reading": reading}
    if "copy" in bridges:
        head["copy_diagnostic"] = {"means": r["means"]["copy"], "oracle_share": r["oracle_acc_share"]["copy"]["blank_corrected"]}
    return by_w, head, checks, R


def write_results(st):
    R4 = load_r4()
    res = {"complete": all(st["pairs"].get(p, {}).get("tested") or st["pairs"].get(p, {}).get("skipped") for p in PAIRS),
           "pairs": {}, "headlines": {},
           "protocol": {"T": T, "C": C, "n_bindings": NB, "windows": WINDOWS, "primary_window": PW,
                        "pairing": {str(k): v for k, v in sb2.PAIR.items()}, "n_train_calib": N_TRAIN, "n_val_calib": N_VAL,
                        "batch": BATCH, "lr": LR, "val_every": VAL_EVERY, "patience": PATIENCE, "max_steps": MAX_STEPS,
                        "data_seed": DATA_SEED, "threads": NTH,
                        "units": "nats/byte; value_ce = mean over the 24 recalled value bytes per doc, then over docs",
                        "code": "train_family.py + statebridge7.py (protocol in docstrings)"},
           "smoke": SMOKE, "source": "ckpt_src_r3b.pt (frozen; rounds 3-6 source)"}
    for pair in PAIRS:
        P = st["pairs"].get(pair)
        if not P:
            continue
        entry = {k: v for k, v in P.items() if k not in ("perdoc",)}
        if P.get("tested"):
            by_w, head, checks, R = analyse_pair(pair, P, R4)
            entry.update(by_window=by_w, checks=checks, per_doc=P["perdoc"])
            res["headlines"][pair] = head
            log("HEADLINE", pair, json.dumps({k: v for k, v in head.items() if k in (
                "trained_beats_all_three", "trained_oracle_share", "trained_minus_r4_value_acc", "reading")}))
        res["pairs"][pair] = entry
    os.makedirs(OUT, exist_ok=True)
    atomic_json(RESULTS, res)
    return res


# ---------------------------------------------------------------- main
def run_pair(pair, st):
    P = st["pairs"].setdefault(pair, {})
    if P.get("tested"):
        log(f"{pair}: already tested; skipping")
        return
    jpath = f"{WD}/train_tgt_{pair}_r7.json"
    if not (os.path.exists(jpath) and os.path.exists(f"{WD}/ckpt_tgt_{pair}_r7.pt")):
        raise RuntimeError(f"{pair}: target not built yet; run python3 train_family.py {pair} first")
    J = json.load(open(jpath))
    P["target_gate"] = {k: v for k, v in J["gates"][-1].items() if k != "detail"}
    P["target_training"] = {"segments": J["segments"], "checks": J["checks"], "snapshots": J.get("snapshots"),
                            "train_minutes": J.get("train_minutes"), "n_params": J.get("n_params")}
    if not J.get("final_gate_pass") and SMOKE:
        log(f"{pair}: smoke test: target gate failed (tiny random model); continuing for the code test only")
    elif not J.get("final_gate_pass"):
        P["skipped"] = "target failed the gate after the allowed extensions; no bridge run"
        atomic_json(STATE, st)
        log(f"{pair}: SKIPPED ({P['skipped']})")
        return
    tgt = load_target(pair)
    log(f"{pair}: target loaded ({sum(p.numel() for p in tgt.parameters())} params, pattern {tgt.pattern})", mem())
    ridge = prep(pair, tgt, P)
    atomic_json(STATE, st)
    V = val_slice(tgt)
    if "val_diag_w32" not in P:
        P["val_diag_w32"] = val_diag(V, make_bridge(ridge).eval(), tgt, extra=True)
        log(f"{pair} calib val diag (w = 32)", P["val_diag_w32"])
        atomic_json(STATE, st)
        trim()
    if P.get("trained") and os.path.exists(best_path(pair)):
        best_state = torch.load(best_path(pair))
        log(f"{pair}: trained bridge loaded from disk (training already done)")
    else:
        if P.get("trained"):
            log(f"{pair}: NOTE training finished in an earlier launch but its bridge was not on disk; retraining")
            P.setdefault("retrained", []).append(P.pop("trained"))
        best_state = train_bridge(pair, tgt, ridge, V, P, st)
    del V
    trim()
    if ridge["W"] is None:
        log(f"{pair}: refitting the ridge for the test pass (not held during training)")
        ridge = prep(pair, tgt, P)
    test_pair(pair, tgt, ridge, best_state, P, st)
    write_results(st)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    assert what in ("grow", "replay", "all"), "usage: statebridge7.py [grow|replay|all]"
    os.makedirs(WD, exist_ok=True)
    take_lock()
    signal.signal(signal.SIGTERM, tf.on_term)
    signal.signal(signal.SIGHUP, tf.on_term)
    STATUS.update(state="running", t_start=time.time(), pid=os.getpid(), threads=NTH, pairs=what, smoke=SMOKE,
                  started=time.strftime("%Y-%m-%d %H:%M:%S"), log=LOG, results=RESULTS,
                  expected_runtime="about 45-60 min per pair at 2 threads (ridge ~6 min, bridge <= 1,000 steps at ~2.8 s/step, "
                                   "usually stops on patience earlier; test ~3 min)")
    if os.path.exists(STATUS_LOCAL):
        old = json.load(open(STATUS_LOCAL))
        STATUS["previous_launches"] = old.get("previous_launches", []) + [
            {k: old.get(k) for k in ("started", "state", "updated", "phase", "pair", "step", "error")}]
    log(f"round 7 start: pid {os.getpid()}, threads {NTH}, pairs {what}, smoke {SMOKE}", mem())
    write_status(phase="setup")
    try:
        st = load_state()
        for pair in (PAIRS if what == "all" else [what]):
            run_pair(pair, st)
            atomic_json(STATE, st)
        res = write_results(st)
        write_status(state="done", phase="done", headlines=res["headlines"])
        log("ROUND7_DONE")
    except BaseException as e:
        tb = traceback.format_exc()
        log("FAILED", repr(e), "\n" + tb)
        write_status(state="failed", error=repr(e), traceback=tb[-4000:], hint="relaunch the same command to resume")
        sys.exit(1)
    finally:
        try:
            if open(LOCK).read().strip() == str(os.getpid()):
                os.remove(LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    main()
