"""StateBridge round 6: does far MORE CALIBRATION DATA lift the trained bridge off its plateau?

ROUND 6 PROTOCOL (fixed and committed before any round-6 run, smoke test included)

Background (rounds 4-5, w = 32, 96 test docs). The paired dense bridge, warm-started at the ridge
map and trained on the frozen target's loss, reached 3.25-3.27 nats/byte value CE and 14-16%
per-byte accuracy (about 27% of the oracle's 59.1%; source alone 57.8%). In every run validation
value CE flattened and then rose while training CE kept falling: the bridge (201.4M params)
overfits the 2,048 calibration training docs. Untested caveat: far more calibration data.
Round 6 changes ONLY the amount of bridge-training data and measures a data-scaling curve.

Unchanged from rounds 4-5
- Frozen models ckpt_{src,tgt}_r3b.pt. Task T = 864, C = 150, 6 bindings; the 96 test docs (test
  split, seed 2); windows {1, 8, 32, 128, 256}, primary w = 32; handoff = SSM state only, conv tail
  zero, attention KV empty; the target re-reads [T-w, T+C-1) and is scored on [T, T+C).
- Bridge = round 4's paired dense map (target SSM layer j <- source SSM layer round(3j/7)),
  y = (z W + b) * sy + my, 201.4M params, warm start EXACTLY at round 4's ridge solution (from
  states_r4.pt: ridge fit on 432 calib docs, the same init for every data size; the ridge is not
  refit), b = 0. Adam lr 5e-5 (fused when available), batch 32.
- Loss through the frozen target: mean CE on the 24 recalled value bytes + mean CE on all bytes
  after the handoff; the window of step s is WINDOWS[s % 5] with WINDOWS = [256, 128, 32, 8, 1].
- Early stopping on the FIXED validation slice of rounds 4-5: the 256 calib-split docs (seed 42)
  with their precomputed source states from states_r4.pt. Every 40 steps, and at step 0, mean
  recalled-value CE at w = 32; best checkpoint kept (step 0 included); stop after 6 evaluations
  without improvement. Mismatched (different-doc) val CE/acc logged as a diagnostic.
  Pre-declared fallback as before: if the first eval after step 0 is worse than step 0 by > 0.5
  nats/byte, restart once from the ridge init with lr / 4.

Changed: the bridge-training set
- A nested pool of 8 chunks x 2,048 synthetic calib-split docs (6 bindings, T = 864, eval_docs on
  the calibration stream). Chunk 0 = round 4's 2,048 training docs (seed 41, checked equal to
  states_r4.pt). Chunks 1..7 use seeds 1001..1007, never used before (other seeds: test split
  seed 2; calib split seeds 1, 5, 41, 42, 99). Data size kx = the first k chunks (k * 2,048 docs).
  The script asserts that no pool doc is byte-identical to a validation doc and reports
  within-pool duplicates.
- Sizes: 1x, 8x, 4x, run in that order (the 1x-vs-8x contrast first). Optional pre-declared
  extension, same settings, only after the default run has finished: 2x (SB6_SIZES=1,8,4,2).
- Compute: every size gets the same cap of 1,000 steps (32,000 samples; 2.4x round 4's 417
  steps; round 5's 1x run bested near 840 cumulative steps). The stop reason is reported: a size
  that stops at the cap was still improving on val (compute-limited at that size).
- Data order: numpy default_rng(0) permutations of the size's docs, a new permutation per epoch
  (round 4's scheme). At 1x this is round 4's order, so 1x replays round 4's 417 steps and then
  continues: a built-in reproduction check (round 4 val CE: 5.025 at step 0, 3.657 at 40, 3.266
  at 400).
- Source states for training batches are computed on the fly by the frozen source (final SSM
  states of src(tok[:, :T-w]), fp32) instead of being precomputed, which would need about 5.4 GB
  of disk or RAM at 8x. Startup check (logged): on 32 round-4 training docs they match
  states_r4.pt to about 2e-6 relative or better. The val slice keeps round 4's precomputed states.

Test (96 test docs; read once per data size)
- After a size's training has stopped and its best checkpoint is fixed, ONE pass over the test
  docs evaluates that bridge (matched and mismatched) at every window. The first pass also
  evaluates the fixed controls (blank + w, oracle + w, source alone, full target prefill) and
  round 4's dense bridge (trained_r4, reference: should reproduce 3.270 / 13.9% at w = 32); these
  are cached, not recomputed. All settings for later sizes are fixed here, so no test number can
  influence any later training. (If the process dies inside a test pass, that pass is redone on
  resume, and the log says so.)
- Arms per size kx: xk + w, xk_mismatched + w (the SAME bridge fed the next test doc's source
  state in the batch, as rounds 3-5); plus trained_r4 (+ mismatched), blank, oracle, source alone,
  full.

Metrics (as rounds 4-5): per doc, mean CE on the 24 recalled value bytes (primary), per-byte
argmax accuracy, exact-value accuracy, CE on all C bytes. Gap closed = (blank - arm) / (blank -
full). Paired bootstrap over test docs (10k resamples, 95% percentile CIs) for value CE, value acc
and all-bytes CE, of: each size vs blank, its own mismatched arm, source alone, oracle, full; each
size vs 1x and vs trained_r4; and the change of the document-specific gap vs 1x,
((xk - xk_mismatched) - (x1 - x1_mismatched)).
Per size also: beats all three (blank, own mismatched, source alone; value CE CIs below 0),
oracle share of accuracy (raw and blank-corrected), document-specific part of the CE gain
((mismatched - matched) / (blank - matched)) and matched-minus-mismatched accuracy.

HEADLINE (w = 32). Does more calibration data raise test recall? Primary contrast: 8x - 1x in
value accuracy and value CE (paired 95% CIs). "Data helps" = accuracy CI entirely above 0 AND CE
CI entirely below 0. Reported with it: the curve over 1x/4x/8x (value CE, accuracy, oracle share,
matched-minus-mismatched gap), val CE at matched steps, stop reasons, and beats-all-three per
size. Pre-declared reading: data helps and 8x stopped at the cap -> the round-5 plateau is at
least partly data-limited and still compute-limited at 8x; data helps and 8x stopped on patience
-> data-limited, and the 8x plateau is the new ceiling at this compute; no significant gain ->
up to 8x more data at equal compute does not lift the plateau, so overfitting 2,048 docs was not
the binding constraint. Same numbers at every window.

Operation (one long unattended process, resumable)
- python statebridge6.py        (from any cwd; it chdirs to its own directory)
- Threads: SB_THREADS (default 2; OMP/MKL set to match, 1 inter-op thread).
- Progress log: round6.log (next to this file), mirrored to the results directory.
- Status: round6-status.json in the results directory (and a local copy), state
  running | done | failed, with results so far; refreshed at every val evaluation.
- Resume: training state (weights, Adam state, best checkpoint, data order, rng, step, logs) is
  saved every 15 minutes and at each size's end; completed sizes and test results live in
  state_r6.json. Relaunching the same command resumes; a second live instance is refused.
- Memory: about 4.5-6 GB anonymous (params, grads, Adam, best copy); no calibration states held.
- Disk: resume checkpoint 3.2 GB (plus a temporary copy while saving, skipped when free space is
  under 4.5 GB), 0.8 GB per finished size; refuses to start with under 5.5 GB free.
- Results: /mnt/project-files/nebius/statebridge/round6/round6-results.json, rewritten after
  every size's test pass (complete = true at the end).
SB6_SMOKE=1: tiny sizes and caps, val subset, test docs replaced by calib-split docs (seed 99),
all outputs under the scratch dir; never reads test documents.
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
os.environ.pop("SB_SMOKE", None)  # rounds 2/4 smoke switches must stay off; round 6 uses SB6_SMOKE
os.environ["SB_SUFFIX"] = "_r3b"

import ctypes  # noqa: E402

try:  # fewer malloc arenas and explicit trims keep freed heap from piling up across sizes
    _LIBC = ctypes.CDLL("libc.so.6")
    _LIBC.mallopt(-8, 2)  # M_ARENA_MAX = 2
except (OSError, AttributeError):
    _LIBC = None
import hashlib  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
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
import statebridge4 as sb4  # noqa: E402  (loads the frozen _r3b models)
from recall import eval_docs, streams  # noqa: E402

sb2 = sb4.sb2
torch.set_num_threads(NTH)

SMOKE = bool(os.environ.get("SB6_SMOKE"))
SCR = "/tmp/claude-0/-home-claude/85fced09-8eb7-5482-be2c-9bc85e817f60/scratchpad"
OUT = f"{SCR}/r6smoke/out" if SMOKE else "/mnt/project-files/nebius/statebridge/round6"
WD = f"{SCR}/r6smoke" if SMOKE else HERE
TAG = "r6smoke" if SMOKE else "r6"
T, C, NB, WINDOWS, PW = sb2.T, sb2.C, sb2.NB, sb2.WINDOWS, sb2.PRIMARY_W
S_IDX, T_IDX = sb2.S_IDX, sb2.T_IDX
src, tgt = sb2.src, sb2.tgt

CHUNK, CHUNK_SEEDS = 2048, [41, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
SIZES = [int(x) for x in os.environ.get("SB6_SIZES", "1,8,4").split(",")]
BATCH, LR, VAL_EVERY, PATIENCE, MAX_STEPS, DATA_SEED = 32, 5e-5, 40, 6, 1000, 0
N_VAL, N_TEST, SAVE_MIN, LOG_EVERY = 256, 96, 15.0, 20
if SMOKE:
    CHUNK, MAX_STEPS, VAL_EVERY, N_VAL, N_TEST, LOG_EVERY = 32, 10, 5, 32, 16, 5
    SAVE_MIN = float(os.environ.get("SB6_SMOKE_SAVE_MIN", "0"))
    SIZES = [int(x) for x in os.environ.get("SB6_SIZES", "1,2").split(",")]
assert MAX_STEPS % VAL_EVERY == 0 and all(1 <= k <= len(CHUNK_SEEDS) for k in SIZES)

LOG = f"{WD}/round6.log"
STATE = f"{WD}/state_{TAG}.json"
STATUS_LOCAL = f"{WD}/round6-status.json"
STATUS_OUT = f"{OUT}/round6-status.json"
RESULTS = f"{OUT}/round6-results.json"
LOCK = f"{WD}/{TAG}.pid"
resume_path = lambda k: f"{WD}/ckpt_resume_{TAG}_x{k}.pt"  # noqa: E731
best_path = lambda k: f"{WD}/ckpt_bridge_{TAG}_x{k}.pt"  # noqa: E731
STATUS = {}


# ---------------------------------------------------------------- bookkeeping
def mem():
    d = {}
    try:
        for line in open("/proc/self/status"):
            if line.startswith(("VmRSS", "RssAnon")):
                k, v = line.split(":")
                d[k] = round(int(v.split()[0]) / 1e6, 2)
    except OSError:
        pass
    d["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    return d


def trim():
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def log(*a):
    s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")


def atomic_json(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def write_status(**kw):
    STATUS.update(kw)
    STATUS["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATUS["elapsed_s"] = round(time.time() - STATUS["t_start"], 1)
    STATUS["memory_gb"] = mem()
    atomic_json(STATUS_LOCAL, STATUS)
    try:
        os.makedirs(OUT, exist_ok=True)
        atomic_json(STATUS_OUT, STATUS)
        shutil.copyfile(LOG, f"{OUT}/round6.log")
    except Exception as e:  # the project mount is not essential to the run
        print("warning: could not mirror status/log to", OUT, repr(e), flush=True)


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"trained": {}, "tested": [], "perdoc": {}, "controls_done": False, "checks": {}, "test_passes": []}


class Terminated(Exception):
    pass


def on_term(signum, frame):
    for sg in (signal.SIGTERM, signal.SIGHUP):  # a second signal must not interrupt the failure report
        signal.signal(sg, signal.SIG_IGN)
    raise Terminated(f"terminated by signal {signum}")


def take_lock():
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip())
            if pid != os.getpid() and os.path.exists(f"/proc/{pid}") and \
                    "statebridge6" in open(f"/proc/{pid}/cmdline").read():
                sys.exit(f"another statebridge6 run is live (pid {pid}); not starting")
        except (ValueError, OSError):
            pass
    open(LOCK, "w").write(str(os.getpid()))


# ---------------------------------------------------------------- data (calibration only)
def prep_file():
    return torch.load(sb4.PREP, weights_only=False, mmap=True)


def val_data():
    D = prep_file()
    V = {"vpos": np.array(D["vpos"]), "va_tok": D["va_tok"][:N_VAL].clone(),
         "Xva": {PW: [x[:N_VAL].clone() for x in D["Xva"][PW]]}}
    return V


def build_pool(kmax, V):
    """Nested training pool: chunk 0 = round 4's docs (seed 41), chunks 1.. = seeds 1001.. ."""
    t0 = time.time()
    S = streams()
    D = prep_file()
    chunks = []
    for c in range(kmax):
        docs, vpos = eval_docs(S["calib"], CHUNK, T, seed=CHUNK_SEEDS[c], nbind=NB)
        assert (vpos == V["vpos"]).all()
        tok = sb2.tt(docs)
        if c == 0:
            assert torch.equal(tok, D["tr_tok"][:CHUNK]), "chunk 0 differs from round 4's training docs"
        chunks.append(tok.to(torch.uint8))
    pool = torch.cat(chunks)
    h = lambda row: hashlib.sha1(row.numpy().tobytes()).hexdigest()  # noqa: E731
    hp = [h(r) for r in pool]
    hv = {h(r.to(torch.uint8)) for r in V["va_tok"]}
    n_val_overlap = sum(x in hv for x in hp)
    assert n_val_overlap == 0, f"{n_val_overlap} training docs identical to validation docs"
    checks = {"pool_docs": len(pool), "pool_within_duplicates": len(hp) - len(set(hp)), "pool_val_overlap": 0,
              "pool_build_s": round(time.time() - t0, 1)}
    # on-the-fly source states vs round 4's precomputed ones (first 32 round-4 training docs)
    idx = torch.arange(min(32, CHUNK))
    rel = {}
    for w in WINDOWS:
        xs = src_x(pool[idx].long(), w)
        rel[str(w)] = max(float((a - b[idx]).abs().max() / b[idx].abs().max()) for a, b in zip(xs, D["Xtr"][w]))
    checks["onthefly_vs_prep_max_rel_diff"] = rel
    del D
    return pool, checks


@torch.no_grad()
def src_x(tok, w):
    _, st, _ = src(tok[:, :T - w])
    return [st[i][0].reshape(len(tok), -1) for i in S_IDX]


def new_bridge():
    D = prep_file()
    br = sb4.Bridge(D["ridge"], "dense")
    for name in ("mx", "sx", "sy", "my"):
        setattr(br, name, [t.clone() for t in getattr(br, name)])
    del D
    return br


def make_opt(br, scale):
    try:
        return torch.optim.Adam(br.parameters(), lr=LR * scale, fused=True)
    except Exception:
        return torch.optim.Adam(br.parameters(), lr=LR * scale)


def save_resume(k, obj):
    path = resume_path(k)
    if os.path.exists(path) and shutil.disk_usage(WD).free < 4.5e9:
        log("low disk: replacing the resume checkpoint in place (no temporary copy)")
        os.remove(path)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------- train one data size
def train_size(k, pool, V, st):
    n = k * CHUNK
    tok_all = pool[:n]
    vpos = V["vpos"]
    rpath = resume_path(k)
    if os.path.exists(rpath + ".tmp"):
        os.remove(rpath + ".tmp")
    br = new_bridge()
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
        log(f"x{k}: resumed at step {S['step']} (t_acc {S['t_acc']:.0f}s)", mem())
    else:
        S = {"step": 0, "t_acc": 0.0, "evals": [], "train": [], "since": 0, "best": None, "scale": 1.0,
             "attempts": [], "oi": 0, "order": None, "n_docs": n,
             "n_params": sum(p.numel() for p in br.parameters())}
        opt = make_opt(br, 1.0)
        rng = np.random.default_rng(DATA_SEED)
        best_state = None
        log(f"x{k}: start, {n} training docs, {S['n_params']} params", mem())
    br.train()
    t_seg, t_base, t_saved = time.time(), S["t_acc"], time.time()
    elapsed = lambda: t_base + time.time() - t_seg  # noqa: E731

    def status(phase):
        sp = S.get("t_acc_step")
        later = sum(1 for kk in SIZES[SIZES.index(k) + 1:] if str(kk) not in st["trained"])
        eta = None if sp is None else round(((MAX_STEPS - S["step"]) + later * MAX_STEPS) * sp * 1.1)
        write_status(phase=phase, size=f"x{k}", step=S["step"], max_steps=MAX_STEPS,
                     best=S["best"], since_best=S["since"], sec_per_step=sp,
                     eta_upper_bound_s=eta, last_eval=S["evals"][-1] if S["evals"] else None)

    def evaluate():
        br.eval()
        r = sb4.val_diag(V, br, PW)
        br.train()
        trim()
        e = {"step": S["step"], "t": round(elapsed(), 1), "val_value_ce": r["bridge"]["value_ce"],
             "val_value_acc": r["bridge"]["value_acc"], "val_mismatched_value_ce": r["bridge_mismatched"]["value_ce"],
             "val_mismatched_value_acc": r["bridge_mismatched"]["value_acc"],
             "rss_anon_gb": mem().get("RssAnon")}
        S["evals"].append(e)
        log(f"x{k} eval", e)
        return e

    snapshot = lambda: {kk: v.detach().clone() for kk, v in br.state_dict().items()}  # noqa: E731

    def keep_best():  # copy in place: no second 0.8 GB copy while the new best is taken
        for kk, v in br.state_dict().items():
            best_state[kk].copy_(v.detach())

    if S["step"] == 0 and not S["evals"]:
        e = evaluate()
        S["best"], best_state = [e["val_value_ce"], 0], snapshot()
        status("train")
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
        lv, la = sb4.loss_fn(sb4.window_logits(tgt, tok, p, sb4.tgt_init(ys)), tok, p, vpos)
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
            log(f"x{k} step {S['step']} train(value, all) {recent.round(3).tolist()} "
                f"{S['t_acc_step']} s/step t_acc {elapsed():.0f}s")
        if S["step"] % VAL_EVERY == 0:
            e = evaluate()
            first = [x for x in S["evals"] if x["step"] > 0]
            if len(first) == 1 and S["scale"] == 1.0 and e["val_value_ce"] > S["evals"][0]["val_value_ce"] + 0.5:
                log(f"x{k} fallback: first eval worse by > 0.5; restarting from the ridge init with lr / 4")
                S["attempts"].append({"evals": S["evals"], "train": S["train"]})
                br = new_bridge().train()
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
            if S["step"] < MAX_STEPS and time.time() - t_saved >= SAVE_MIN * 60:
                S["t_acc"], S["rng"] = elapsed(), rng.bit_generator.state
                save_resume(k, {"S": S, "model": br.state_dict(), "opt": opt.state_dict(), "best_state": best_state})
                t_saved = time.time()
                log(f"x{k} resume checkpoint saved at step {S['step']}")
            status("train")
    S["t_acc"] = elapsed()
    if os.path.exists(rpath) and shutil.disk_usage(WD).free < 2e9:
        log(f"x{k} low disk: removing the resume checkpoint before saving the best checkpoint")
        os.remove(rpath)
    torch.save(best_state, best_path(k) + ".tmp")
    os.replace(best_path(k) + ".tmp", best_path(k))
    tl = {kk: v for kk, v in S.items() if kk not in ("order", "rng")}
    tl.update(stop=stop, best_step=S["best"][1], best_val_value_ce=S["best"][0], lr=LR * S["scale"],
              batch=BATCH, val_every=VAL_EVERY, patience=PATIENCE, max_steps=MAX_STEPS, data_seed=DATA_SEED,
              memory_gb=mem())
    st["trained"][str(k)] = tl
    atomic_json(STATE, st)
    if os.path.exists(rpath):
        os.remove(rpath)
    log(f"x{k} TRAINING_DONE: stop {stop} at step {S['step']}, best {S['best']}, t_acc {S['t_acc']:.0f}s", mem())
    del br, opt, best_state
    trim()


# ---------------------------------------------------------------- test (read once per size)
def test_size(k, V, st):
    t0 = time.time()
    first = not st["controls_done"]
    write_status(phase="test", size=f"x{k}")
    st["test_passes"].append({"size": k, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "controls": first})
    atomic_json(STATE, st)
    if sum(p["size"] == k for p in st["test_passes"]) > 1:
        log(f"x{k}: NOTE a previous test pass for this size did not finish; redoing it")
    S = streams()
    if SMOKE:
        docs, vpos = eval_docs(S["calib"], N_TEST, T, seed=99, nbind=NB)
    else:
        docs, vpos = eval_docs(S["test"], N_TEST, T, seed=2, nbind=NB)
    assert (vpos == V["vpos"]).all()
    bridges = {}
    br = new_bridge()
    br.load_state_dict(torch.load(best_path(k)))
    bridges[f"x{k}"] = br.eval()
    if first:
        r4 = new_bridge()
        r4.load_state_dict(torch.load("ckpt_bridge_dense_r4.pt"))
        bridges["trained_r4"] = r4.eval()
    store = {}
    cuts = [T - w for w in WINDOWS] + [T + C - 1]
    for s in range(0, len(docs), sb2.BATCH):
        tok = sb2.tt(docs[s:s + sb2.BATCH])
        B = len(tok)
        sst, slo = sb2.chain(src, tok, cuts)
        if first:
            tst, tlo = sb2.chain(tgt, tok, cuts)
            sb2.add(store, "full", sb2.metrics(*sb2.score(tlo[:, -C:], tok), vpos))
            sb2.add(store, "source_alone", sb2.metrics(*sb2.score(slo[:, -C:], tok), vpos))
        for w in WINDOWS:
            p = T - w
            with torch.no_grad():
                win = lambda init: sb2.metrics(*sb2.score(sb4.window_logits(tgt, tok, p, init)[:, -C:], tok), vpos)  # noqa: E731
                if first:
                    true = {j: sb2.ssm_flat(tst[p], i) for j, i in enumerate(T_IDX)}
                    sb2.add(store, f"oracle@{w}", win(sb2.init_from(true, B)))
                    sb2.add(store, f"blank@{w}", win(None))
                xs = [torch.from_numpy(sb2.ssm_flat(sst[p], i)) for i in S_IDX]
                for arm, b in bridges.items():
                    ys = b(b.z(xs))
                    sb2.add(store, f"{arm}@{w}", win(sb4.tgt_init(ys)))
                    sb2.add(store, f"{arm}_mismatched@{w}", win(sb4.tgt_init([torch.roll(y, -1, 0) for y in ys])))
        log(f"x{k} test batch {s // sb2.BATCH}: {time.time() - t0:.0f}s", mem())
    for a, v in store.items():
        st["perdoc"][a] = {m: x.tolist() for m, x in sb2.stack(v).items()}
    st["tested"].append(k)
    st["controls_done"] = True
    st["test_passes"][-1]["finished_s"] = round(time.time() - t0, 1)
    atomic_json(STATE, st)
    log(f"x{k} TEST_DONE in {time.time() - t0:.0f}s")
    del bridges
    trim()


# ---------------------------------------------------------------- analysis
def analyse(st):
    R = {a: {m: np.array(v) for m, v in d.items()} for a, d in st["perdoc"].items()}
    sizes = sorted(st["tested"])
    xs = [f"x{k}" for k in sizes]
    bridges = xs + ["trained_r4"]
    arms = ["oracle", "blank"] + [x for b in bridges for x in (b, f"{b}_mismatched")]
    boot = sb2.boot
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
                    d[f"gap_closed_{a}"] = float((g("blank", metric).mean() - g(a, metric).mean()) / gap)
            for b in bridges:
                for c in ["blank", f"{b}_mismatched", "source_alone", "oracle", "full"]:
                    d[f"diff_{b}_minus_{c}"] = boot(g(b, metric) - g(c, metric))
            for b in xs:
                d[f"diff_{b}_minus_trained_r4"] = boot(g(b, metric) - g("trained_r4", metric))
                if b != "x1" and "x1" in xs:
                    d[f"diff_{b}_minus_x1"] = boot(g(b, metric) - g("x1", metric))
                    d[f"docgap_change_{b}_minus_x1"] = boot((g(b, metric) - g(f"{b}_mismatched", metric)) -
                                                            (g("x1", metric) - g("x1_mismatched", metric)))
            if metric == "value_ce":
                for b in bridges:
                    d[f"{b}_beats_all_three"] = bool(d[f"diff_{b}_minus_blank"][2] < 0 and
                                                     d[f"diff_{b}_minus_{b}_mismatched"][2] < 0 and
                                                     d[f"diff_{b}_minus_source_alone"][2] < 0)
            row[metric] = d
        m = row["means"]
        row["oracle_acc_share"] = {a: {"ratio": m[a]["value_acc"] / m["oracle"]["value_acc"],
                                       "blank_corrected": (m[a]["value_acc"] - m["blank"]["value_acc"]) /
                                       (m["oracle"]["value_acc"] - m["blank"]["value_acc"])}
                                   for a in arms if a not in ("oracle", "blank")}
        row["doc_specific"] = {}
        for b in bridges:
            ce_gain = m["blank"]["value_ce"] - m[b]["value_ce"]
            ds = m[f"{b}_mismatched"]["value_ce"] - m[b]["value_ce"]
            row["doc_specific"][b] = {"ce_gain_over_blank": ce_gain, "ce_matched_minus_mismatched": -ds,
                                      "frac_of_ce_gain_doc_specific": ds / ce_gain,
                                      "acc_matched_minus_mismatched": m[b]["value_acc"] - m[f"{b}_mismatched"]["value_acc"],
                                      "acc_gain_over_blank": m[b]["value_acc"] - m["blank"]["value_acc"]}
        by_w[str(w)] = row
    r = by_w[str(PW)]
    scaling = []
    for k in sizes:
        b, tl = f"x{k}", st["trained"][str(k)]
        m = r["means"]
        scaling.append({
            "size": f"{k}x", "train_docs": k * CHUNK, "steps": tl["step"], "stop": tl["stop"],
            "best_step": tl["best_step"], "best_val_value_ce": tl["best_val_value_ce"],
            "epochs_at_best": tl["best_step"] * BATCH / (k * CHUNK),
            "test_value_ce": m[b]["value_ce"], "test_value_acc": m[b]["value_acc"],
            "test_value_exact": m[b]["value_exact"],
            "mismatched_value_ce": m[f"{b}_mismatched"]["value_ce"], "mismatched_value_acc": m[f"{b}_mismatched"]["value_acc"],
            "oracle_share": r["oracle_acc_share"][b]["ratio"], "oracle_share_blank_corrected": r["oracle_acc_share"][b]["blank_corrected"],
            "ce_matched_minus_mismatched": r["value_ce"][f"diff_{b}_minus_{b}_mismatched"],
            "acc_matched_minus_mismatched": r["value_acc"][f"diff_{b}_minus_{b}_mismatched"],
            "ce_minus_x1": r["value_ce"].get(f"diff_{b}_minus_x1"), "acc_minus_x1": r["value_acc"].get(f"diff_{b}_minus_x1"),
            "beats_all_three": r["value_ce"][f"{b}_beats_all_three"]})
    head = None
    top = max(sizes)
    if 1 in sizes and top > 1:
        ce, acc = r["value_ce"][f"diff_x{top}_minus_x1"], r["value_acc"][f"diff_x{top}_minus_x1"]
        head = {"contrast": f"x{top} - x1 at w = {PW}", "value_ce_diff": ce, "value_acc_diff": acc,
                "data_helps": bool(acc[1] > 0 and ce[2] < 0), f"x{top}_stop": st["trained"][str(top)]["stop"]}
    ref = {"trained_r4_test_value_ce": r["means"]["trained_r4"]["value_ce"],
           "trained_r4_test_value_acc": r["means"]["trained_r4"]["value_acc"],
           "expected_from_round4": [3.270, 0.139]}
    return by_w, scaling, head, ref, R


def write_results(st, complete):
    by_w, scaling, head, ref, R = analyse(st)
    res = {
        "complete": complete,
        "headline": head,
        "scaling_w32": scaling,
        "trained_r4_reproduction": ref,
        "protocol": {"T": T, "C": C, "n_bindings": NB, "windows": WINDOWS, "primary_window": PW,
                     "pairing": {str(k): v for k, v in sb2.PAIR.items()}, "sizes_run_order": SIZES,
                     "chunk_docs": CHUNK, "chunk_seeds": CHUNK_SEEDS, "batch": BATCH, "lr": LR,
                     "val_every": VAL_EVERY, "patience": PATIENCE, "max_steps": MAX_STEPS, "data_seed": DATA_SEED,
                     "n_val": N_VAL, "threads": NTH,
                     "units": "nats/byte; value_ce = mean over the 24 recalled value bytes per doc, then over docs",
                     "code": "statebridge6.py (protocol in docstring)"},
        "data": {"n_test_docs": N_TEST, "smoke": SMOKE, "checks": st["checks"], "test_passes": st["test_passes"]},
        "models": "ckpt_src_r3b.pt, ckpt_tgt_r3b.pt (frozen)",
        "bridge_training": st["trained"],
        "full_prefill": {k: float(v.mean()) for k, v in R["full"].items()},
        "source_alone": {k: float(v.mean()) for k, v in R["source_alone"].items()},
        "by_window": by_w,
        "per_doc": {a: {k: v.tolist() for k, v in mm.items()} for a, mm in R.items()},
    }
    os.makedirs(OUT, exist_ok=True)
    atomic_json(RESULTS, res)
    for s in scaling:
        log("scaling w=32", {k: (round(v, 4) if isinstance(v, float) else
                                 ([round(x, 4) for x in v] if isinstance(v, list) else v)) for k, v in s.items()})
    log("trained_r4 reproduction", ref)
    if head:
        log("HEADLINE", head)
    return scaling, head, ref


# ---------------------------------------------------------------- main
def run():
    st = load_state()
    kmax = max(SIZES)
    V = val_data()
    pool, checks = build_pool(kmax, V)
    st["checks"].update(checks)
    atomic_json(STATE, st)
    log("checks", checks, mem())
    free = shutil.disk_usage(WD).free
    if not SMOKE and free < float(os.environ.get("SB6_MIN_FREE_GB", "5.5")) * 1e9 and any(str(k) not in st["trained"] for k in SIZES):
        raise RuntimeError(f"only {free / 1e9:.1f} GB free under {WD}; round 6 needs at least 5.5 GB, 8+ GB "
                           "preferred (resume checkpoint 3.2 GB + a temporary copy, 0.8 GB per finished size)")
    for k in SIZES:
        if k in st["tested"]:
            log(f"x{k}: already trained and tested; skipping")
            continue
        if str(k) not in st["trained"]:
            train_size(k, pool, V, st)
        test_size(k, V, st)
        scaling, head, ref = write_results(st, complete=all(kk in st["tested"] for kk in SIZES))
        write_status(results_so_far={"scaling_w32": scaling, "headline": head, "trained_r4_reproduction": ref},
                     sizes_done=sorted(st["tested"]))


def main():
    os.makedirs(WD, exist_ok=True)
    take_lock()
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)
    STATUS.update(state="running", t_start=time.time(), pid=os.getpid(), threads=NTH, sizes=SIZES, smoke=SMOKE,
                  started=time.strftime("%Y-%m-%d %H:%M:%S"), log=LOG, results=RESULTS,
                  expected_runtime="about 2.5 h at 2 threads for 1x, 8x, 4x (upper bound 3 x 1,000 steps at ~3 s/step)")
    if os.path.exists(STATUS_LOCAL):
        old = json.load(open(STATUS_LOCAL))
        STATUS["previous_launches"] = old.get("previous_launches", []) + [
            {k: old.get(k) for k in ("started", "state", "updated", "phase", "size", "step", "error")}]
    log(f"round 6 start: pid {os.getpid()}, threads {NTH}, sizes {SIZES}, smoke {SMOKE}", mem())
    write_status(phase="setup")
    try:
        run()
        write_status(state="done", phase="done")
        log("ROUND6_DONE")
    except BaseException as e:
        tb = traceback.format_exc()
        log("FAILED", repr(e), "\n" + tb)
        write_status(state="failed", error=repr(e), traceback=tb[-4000:],
                     hint="relaunch the same command to resume from the last checkpoint")
        sys.exit(1)
    finally:
        try:
            if open(LOCK).read().strip() == str(os.getpid()):
                os.remove(LOCK)
        except OSError:
            pass


if __name__ == "__main__":
    main()
