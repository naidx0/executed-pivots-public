"""StateBridge round 7, part 1: build and train SAME-FAMILY targets for the round-3 source.

ROUND 7 PROTOCOL, TARGETS (fixed in this docstring and in statebridge7.py's, committed before any
round-7 run, smoke test included)

Why. In rounds 3-6 the toy source and target were trained apart (different seeds, widths, depths,
data order). Nemotron Nano and Super share family, tokenizer and data. That is the main untested
reason the round 3-6 negative result might not transfer. Round 7 builds targets that are related to
the SAME source (ckpt_src_r3b.pt, 0.68M params, d 128, MMAMM, the source of rounds 3-6) and runs the
round 3-5 protocol unchanged on them (statebridge7.py).

Both targets have the round 3-6 target's size: d 192, 8 SSM + 2 attention(+MLP) layers, headdim 32,
d_state 16, chunk 32, 2,844,512 params.

(a) "grow" (PRIMARY): the target is grown FROM the source, then trained on.
  - Function-preserving growth (net2net / zero-padding style) of ckpt_src_r3b.pt to d 192 and
    pattern MMMMAMMMMA: target layer 0 <- src 0 (M), 1 new M, 2 <- src 1 (M), 3 new M,
    4 <- src 2 (attention + MLP), 5 <- src 3 (M), 6 new M, 7 <- src 4 (M), 8 new M, 9 new attention.
    Copied layers keep the source weights in the leading blocks. New residual dims (128..191) start
    at exactly 0 (every writer's new rows are 0). New SSM heads (8..11) get zero x-input weights from
    the old dims and zero conv bias, so they output 0. New attention heads write nothing (o columns
    0). New MLP units write nothing. RMSNorm weights on old dims are divided by sqrt(new width / old
    width) to cancel the zero padding. Inserted layers: random init with a zero output projection
    (identity residual branch). Every other new weight is random (torch seed 7) so gradients flow.
    With this layout the round-3 pairing rule (target SSM j <- source SSM round(3j/7) =
    0,0,1,1,2,2,3,3) coincides with provenance: each source SSM layer feeds its copy and the new
    layer inserted after it.
  - Growth check (logged; the run stops if it fails): on 16 calib-split docs, with every RMSNorm eps
    rescaled by old/new width (so the zero padding is cancelled exactly), max |logit difference|
    grown vs source < 1e-3 and the copied layers' SSM states (heads 0..7) equal the source's (max
    relative difference < 1e-4). The model is then used with the standard eps (1e-5, as HybridLM
    builds it); that leaves a tiny eps-only difference, which is logged and must stay below 1e-2 in
    max |probability difference| (smoke test: 5e-4).
  - Continued training: the source's last-phase recipe (train_r3.py worker, final curriculum stage 7
    with rehearsal, LM_FRAC 0.2, handoff rows, VALW 4, T 1024, batch 8, AdamW betas (0.9, 0.95), wd
    0.1, clip 1) at the round-3 target's lr 1.5e-3 (warm-up 30 steps, constant, then cosine to 0.2x
    over the last 25%), for 1,000 steps, on the SAME data stream: the source's r3b numpy stream
    (default_rng(30)), replayed without a model through the source's 635 r3b batches with its recorded
    stage schedule, then continued.
  - Snapshots at steps 0, 250, 500, 1000 (diagnostics only: copy-slice R^2 of the copied SSM layers
    vs the source, full-prefill recall on 16 gate docs).

(b) "replay" (SECONDARY): the round 3-6 target architecture (MMMAMMMMAM), trained apart from the
  source but on the IDENTICAL data order, from a SHARED INIT where shapes allow.
  - Init: the round-1 target init (torch seed 1) with the source's round-1 init (torch seed 0)
    copied into the leading blocks of every matching tensor (target SSM j <- source SSM
    round(3j/7); both attention layers and MLPs <- the source's; embedding, head, norms). Not
    function-preserving; this is "shared seed where shapes allow".
  - Training: replays the source's whole history batch for batch: phases r1 (LM, T 2048), r2a1, r2
    (train.py recall mix, options as recorded), r3, r3b (train_r3.py curriculum, the source's
    recorded stage schedule), with the source's numpy seeds and step counts (212, 449, 688, 584,
    635). Hyperparameters are the round 1-3 target's own, identical to the source's except lr 1.5e-3
    vs 2e-3 in r3/r3b. train.py's wall-clock cosine is reproduced by interpolating the source's
    logged elapsed times. Fresh AdamW per phase, as originally.
  - Replay fidelity check (every run start, logged, not a stop condition): the source itself, run
    through the replay code from its recorded phase-start checkpoints, reproduces its logged step-0
    loss (r1, r2) and 25-step training-loss EMA (r3, r3b) to 4 decimals (checked before commit;
    r1 needs round 1's batch rule, one vectorised index draw, which the replay uses). Known
    exception: r2a1 ran on a pre-commit working copy and does not reproduce (step-0 loss 2.05 vs
    logged 1.59); its 449 steps use the committed code with the same seed (same distribution, not the
    same batches). Everything else in (b)'s data order is identical to the source's.

Gate (both targets; the round-3 gate unchanged): on the 64 gate docs (calib split, seed 1), at w =
32, the target's oracle (own SSM state, conv tail zero, attention KV empty) must beat blank + w and
mismatched-own + w by >= 0.5 nats/byte with 95% paired-bootstrap CIs below 0, AND full-prefill value
CE < 3.08 (round 3's recall bar). If a target fails: continue its final-stage training on the same
stream in blocks of 250 steps, gate after each, at most 4 blocks. Still failing: the pair is
reported as failed and statebridge7.py runs no bridge for it.

Outputs: ckpt_tgt_{grow,replay}_r7.pt (final), ckpt_tgt_grow_r7_s{0,250,500,1000}.pt,
train_tgt_{pair}_r7.json (logs, checks, gates).

Operation: python3 train_family.py {grow|replay|check}   (chdirs to its own directory)
- Threads: SB_THREADS (default 2). Status: family7-status.json (here and in the round-7 results
  dir); log: family7.log. Resume: model, optimizer, rng and logs saved every 100 steps
  (ckpt_resume_family_{pair}_r7.pt, about 35 MB); relaunching resumes. A second live instance is
  refused. Memory: about 2 GB peak (batches run as gradient-accumulated micro-batches of
SB7_MICRO_TOKENS = 4096 tokens: 2 rows at T 2048, 4 rows at T 1024; same gradient up to summation
order; measured per-row activation memory 0.67 GB at T 2048 and 0.34 GB at T 1024 for this target).
`check` runs only the replay fidelity check.
SB7_SMOKE=1 (code test only, under 2 GB RSS): TINY models (a random d-32 source grown to d 48, and a
d-48 replay target; the real source's checkpoints are used only for the forward-only replay check of
r1/r2), a few steps per phase, 16 gate docs, gate failure not blocking, outputs under the scratch
dir; calib docs only.
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
os.environ["SB_SUFFIX"] = "_r3b"
os.environ.pop("SB_SMOKE", None)

import ctypes  # noqa: E402

try:
    ctypes.CDLL("libc.so.6").mallopt(-8, 2)  # M_ARENA_MAX = 2 (freed heap does not pile up)
except (OSError, AttributeError):
    pass
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(NTH)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
import statebridge2 as sb2  # noqa: E402  (loads the r3b models; only helpers are used here)
import train_r3 as r3  # noqa: E402
from data import train_stream  # noqa: E402
from model import CONFIGS, HybridLM  # noqa: E402
from recall import eval_docs, streams, train_row  # noqa: E402

torch.set_num_threads(NTH)
SMOKE = bool(os.environ.get("SB7_SMOKE"))
SCR = "/tmp/claude-0/-home-claude/85fced09-8eb7-5482-be2c-9bc85e817f60/scratchpad"
WD = f"{SCR}/r7smoke" if SMOKE else HERE
OUT = f"{SCR}/r7smoke/out" if SMOKE else "/mnt/project-files/nebius/statebridge/round7"
T, C, NB, WINDOWS, PW = sb2.T, sb2.C, sb2.NB, sb2.WINDOWS, sb2.PRIMARY_W
GROW_CFG = dict(d=192, pattern="MMMMAMMMMA", headdim=32, d_state=16, attn_headdim=32, chunk=32)
SRC_CFG, TGT_CFG, SRC_CKPT = CONFIGS["src"], CONFIGS["tgt"], "ckpt_src_r3b.pt"
if SMOKE:  # tiny models: the smoke test checks code paths only (memory: shared machine)
    SRC_CFG = dict(d=32, pattern="MMAMM", headdim=16, d_state=8, attn_headdim=16, chunk=32)
    TGT_CFG = dict(d=48, pattern="MMMAMMMMAM", headdim=16, d_state=8, attn_headdim=16, chunk=32)
    GROW_CFG = dict(d=48, pattern="MMMMAMMMMA", headdim=16, d_state=8, attn_headdim=16, chunk=32)
    SRC_CKPT = f"{WD}/ckpt_src_smoke.pt"
GROW_PROV = {0: 0, 2: 1, 4: 2, 5: 3, 7: 4}  # grown target layer -> source layer (others are new)
GROW_SEED, GROW_STEPS, GROW_SNAPS = 7, 1000, [0, 250, 500, 1000]
TGT_LR3, TGT_WARM3 = 1.5e-3, 30  # the round-3 target's train_r3 settings
EXT_BLOCK, EXT_MAX = 250, 4
RECALL_BAR = 3.08
MICRO_TOKENS = int(os.environ.get("SB7_MICRO_TOKENS", "4096"))  # rows per micro-batch = this // T
N_GATE, SAVE_EVERY, LOG_EVERY = 64, 100, 25
if SMOKE:
    GROW_STEPS, GROW_SNAPS, N_GATE, SAVE_EVERY, LOG_EVERY, EXT_BLOCK, EXT_MAX = 6, [0, 3, 6], 16, 3, 2, 2, 0
LOG = f"{WD}/family7.log"
STATUS_LOCAL = f"{WD}/family7-status.json"
STATUS_OUT = f"{OUT}/family7-status.json"
LOCK = f"{WD}/family7.pid"
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
        shutil.copyfile(LOG, f"{OUT}/family7.log")
    except Exception as e:  # the project mount is not essential
        print("warning: could not mirror status/log to", OUT, repr(e), flush=True)


class Terminated(Exception):
    pass


def on_term(signum, frame):
    for sg in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sg, signal.SIG_IGN)
    raise Terminated(f"terminated by signal {signum}")


def take_lock():
    if os.path.exists(LOCK):
        try:
            pid = int(open(LOCK).read().strip())
            if pid != os.getpid() and os.path.exists(f"/proc/{pid}") and \
                    "train_family" in open(f"/proc/{pid}/cmdline").read():
                sys.exit(f"another train_family run is live (pid {pid}); not starting")
        except (ValueError, OSError):
            pass
    open(LOCK, "w").write(str(os.getpid()))


def load_ckpt(path):
    ck = torch.load(path)
    m = HybridLM(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m


def save_model(m, cfg, path, extra=None):
    obj = {"cfg": cfg, "state": m.state_dict()}
    if extra:
        obj.update(extra)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


# ---------------------------------------------------------------- growth / shared init
def _mamba_segments(m):
    di, N, H = m.di, m.N, m.H
    return {"z": (0, di), "x": (di, 2 * di), "B": (2 * di, 2 * di + N), "C": (2 * di + N, 2 * di + 2 * N),
            "dt": (2 * di + 2 * N, 2 * di + 2 * N + H)}


@torch.no_grad()
def copy_mamba(s, t, ds, preserve):
    ss, ts = _mamba_segments(s), _mamba_segments(t)
    W, Ws = t.in_proj.weight, s.in_proj.weight
    for k in ss:
        a, b = ss[k]
        c = ts[k][0]
        W[c:c + b - a, :ds] = Ws[a:b]
    if preserve:  # new heads get no x input from the (non-zero) old residual dims
        c = ts["x"][0]
        W[c + s.di:c + t.di, :ds] = 0
    # conv channels: [x (di), B (N), C (N)]
    t.conv.weight[:s.di] = s.conv.weight[:s.di]
    t.conv.bias[:s.di] = s.conv.bias[:s.di]
    t.conv.weight[t.di:t.di + 2 * s.N] = s.conv.weight[s.di:s.di + 2 * s.N]
    t.conv.bias[t.di:t.di + 2 * s.N] = s.conv.bias[s.di:s.di + 2 * s.N]
    if preserve:
        t.conv.bias[s.di:t.di] = 0  # silu(0) = 0: new heads' x is exactly 0
    t.dt_bias[:s.H] = s.dt_bias
    t.A_log[:s.H] = s.A_log
    t.D[:s.H] = s.D
    if preserve:
        t.norm.w[:s.di] = s.norm.w / math.sqrt(t.di / s.di)
        t.norm.w[s.di:] = 1.0
        t.out_proj.weight[ds:] = 0
    else:
        t.norm.w[:s.di] = s.norm.w
    t.out_proj.weight[:ds, :s.di] = s.out_proj.weight


@torch.no_grad()
def copy_attn(s, t, ds, dt_, preserve):
    for g in range(3):  # q, k, v blocks
        t.qkv.weight[g * dt_:g * dt_ + ds, :ds] = s.qkv.weight[g * ds:(g + 1) * ds]
    t.o.weight[:ds, :ds] = s.o.weight
    if preserve:
        t.o.weight[:ds, ds:] = 0
        t.o.weight[ds:] = 0


@torch.no_grad()
def copy_mlp(s, t, ds, preserve, f):
    hs = s[1].weight.shape[0]
    t[0].w[:ds] = s[0].w / f if preserve else s[0].w
    if preserve:
        t[0].w[ds:] = 1.0
    t[1].weight[:hs, :ds] = s[1].weight
    t[1].bias[:hs] = s[1].bias
    t[3].weight[:ds, :hs] = s[3].weight
    t[3].bias[:ds] = s[3].bias
    if preserve:
        t[3].weight[:ds, hs:] = 0
        t[3].weight[ds:] = 0
        t[3].bias[ds:] = 0


@torch.no_grad()
def embed_source(src, tgt, prov, preserve):
    """Copy src weights into the leading blocks of tgt. prov: tgt layer -> src layer (same type).
    preserve=True: function-preserving growth (zeros / norm rescaling as in the docstring)."""
    ds, dt_ = src.emb.weight.shape[1], tgt.emb.weight.shape[1]
    f = math.sqrt(dt_ / ds)
    tgt.emb.weight[:, :ds] = src.emb.weight
    if preserve:
        tgt.emb.weight[:, ds:] = 0
    for i, c in enumerate(tgt.pattern):
        if i in prov:
            k = prov[i]
            assert src.pattern[k] == c
            tgt.norms[i].w[:ds] = src.norms[k].w / f if preserve else src.norms[k].w
            if preserve:
                tgt.norms[i].w[ds:] = 1.0
            if c == "M":
                copy_mamba(src.layers[k], tgt.layers[i], ds, preserve)
            else:
                copy_attn(src.layers[k], tgt.layers[i], ds, dt_, preserve)
                copy_mlp(src.mlps[str(k)], tgt.mlps[str(i)], ds, preserve, f)
        elif preserve:  # inserted layer: identity residual branch
            if c == "M":
                tgt.layers[i].out_proj.weight.zero_()
            else:
                tgt.layers[i].o.weight.zero_()
                tgt.mlps[str(i)][3].weight.zero_()
                tgt.mlps[str(i)][3].bias.zero_()
    tgt.fnorm.w[:ds] = src.fnorm.w / f if preserve else src.fnorm.w
    if preserve:
        tgt.fnorm.w[ds:] = 1.0
    tgt.head.weight[:, :ds] = src.head.weight


def grow(src):
    torch.manual_seed(GROW_SEED)
    t = HybridLM(**GROW_CFG)
    embed_source(src, t, GROW_PROV, preserve=True)
    return t


def replay_init():
    torch.manual_seed(0)
    s0 = HybridLM(**SRC_CFG)  # the source's round-1 init (train.py: manual_seed(0) then build)
    torch.manual_seed(1)
    t0 = HybridLM(**TGT_CFG)  # the round-1 target init
    s_ssm, t_ssm = s0.ssm_idx, t0.ssm_idx
    pair = {j: round(j * (len(s_ssm) - 1) / (len(t_ssm) - 1)) for j in range(len(t_ssm))}
    prov = {t_ssm[j]: s_ssm[k] for j, k in pair.items()}
    s_att = [i for i, c in enumerate(s0.pattern) if c == "A"]
    for i, c in enumerate(t0.pattern):
        if c == "A":
            prov[i] = s_att[0]
    embed_source(s0, t0, prov, preserve=False)
    return t0, prov


@torch.no_grad()
def growth_check(src, t):
    docs, _ = eval_docs(streams()["calib"], 16, T, seed=1, nbind=NB)
    tok = sb2.tt(docs)
    ls, ss, _ = src(tok)
    out = {}
    for mode in ("eps_rescaled", "standard_eps"):
        tm = t
        if mode == "eps_rescaled":
            tm = HybridLM(**GROW_CFG)
            tm.load_state_dict(t.state_dict())
            for mod in tm.modules():
                if mod.__class__.__name__ == "RMSNorm":
                    n = mod.w.shape[0]
                    mod.eps = mod.eps * (src.emb.weight.shape[1] / n if n == GROW_CFG["d"] else
                                         src.layers[0].di / n)
            tm.eval()
        lt, st, _ = tm(tok)
        rel = {}
        for ti, si in GROW_PROV.items():
            if t.pattern[ti] == "M":
                hs, ht = ss[si][0], st[ti][0][:, :src.layers[si].H]
                rel[str(ti)] = float((hs - ht).abs().max() / hs.abs().max())
        out[mode] = {"max_abs_logit_diff": float((ls - lt).abs().max()),
                     "max_abs_prob_diff": float((ls.softmax(-1) - lt.softmax(-1)).abs().max()),
                     "copied_state_max_rel_diff": rel}
    e, sd = out["eps_rescaled"], out["standard_eps"]
    out["pass"] = bool(e["max_abs_logit_diff"] < 1e-3 and max(e["copied_state_max_rel_diff"].values()) < 1e-4
                       and sd["max_abs_prob_diff"] < 1e-2)
    return out


# ---------------------------------------------------------------- gate (round-3 gate, explicit model)
def init_state(model, hs):
    st = [None] * len(model.pattern)
    for i, h in zip(model.ssm_idx, hs):
        st[i] = (h, None)
    return st


@torch.no_grad()
def gate(model, n_docs=None):
    t0 = time.time()
    docs, vpos = eval_docs(streams()["calib"], n_docs or N_GATE, T, seed=1, nbind=NB)
    store = {}
    cuts = [T - w for w in WINDOWS] + [T + C - 1]
    for s in range(0, len(docs), 16):
        tok = sb2.tt(docs[s:s + 16])
        tst, tlo = sb2.chain(model, tok, cuts)
        sb2.add(store, "full", sb2.metrics(*sb2.score(tlo[:, -C:], tok), vpos))
        for w in WINDOWS:
            p = T - w
            win = lambda init: sb2.metrics(*sb2.score(model(tok[:, p:T + C - 1], init)[0][:, -C:], tok), vpos)  # noqa: E731
            true = [tst[p][i][0] for i in model.ssm_idx]
            sb2.add(store, f"oracle@{w}", win(init_state(model, true)))
            sb2.add(store, f"blank@{w}", win(None))
            sb2.add(store, f"mismatched_own@{w}", win(init_state(model, [torch.roll(h, -1, 0) for h in true])))
    R = {a: sb2.stack(v) for a, v in store.items()}
    res = {"n_docs": len(docs), "full": {k: float(v.mean()) for k, v in R["full"].items()}, "by_window": {}}
    for w in WINDOWS:
        o, b, m = (R[f"{a}@{w}"]["value_ce"] for a in ["oracle", "blank", "mismatched_own"])
        row = {a: {k: float(v.mean()) for k, v in R[f"{a}@{w}"].items()} for a in ["oracle", "blank", "mismatched_own"]}
        row["diff_oracle_minus_blank"] = sb2.boot(o - b)
        row["diff_oracle_minus_mismatched_own"] = sb2.boot(o - m)
        res["by_window"][str(w)] = row
    r = res["by_window"][str(PW)]
    d1, d2 = r["diff_oracle_minus_blank"], r["diff_oracle_minus_mismatched_own"]
    res["registered_gate_pass"] = bool(d1[0] <= -sb2.GATE_MARGIN and d2[0] <= -sb2.GATE_MARGIN and d1[2] < 0 and d2[2] < 0)
    res["recall_bar_pass"] = bool(res["full"]["value_ce"] < RECALL_BAR)
    res["pass"] = res["registered_gate_pass"] and res["recall_bar_pass"]
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


def gate_summary(g):
    r = g["by_window"][str(PW)]
    return {"pass": g["pass"], "full_value_ce": round(g["full"]["value_ce"], 4), "full_value_acc": round(g["full"]["value_acc"], 4),
            "w32_oracle_ce": round(r["oracle"]["value_ce"], 4), "w32_oracle_acc": round(r["oracle"]["value_acc"], 4),
            "w32_blank_ce": round(r["blank"]["value_ce"], 4), "w32_mismatched_own_ce": round(r["mismatched_own"]["value_ce"], 4)}


# ---------------------------------------------------------------- data replay
_DATA = {}


def data():
    if not _DATA:
        S = streams()
        _DATA["rstream"] = S["train"]
        _DATA["lm"] = train_stream()
    return _DATA


def tp_opts(o):
    """train.py options (env strings as recorded) -> values with train.py's defaults."""
    o = o or {}
    g = o.get
    return dict(BS=int(g("SB_BS", "8")), T=int(g("SB_T", "2048")), MIX=float(g("SB_MIX", "0")), SEGP=float(g("SB_SEGP", "0")),
                VALW=float(g("SB_VALW", "0")), SEGMIN=int(g("SB_SEGMIN", "32")), SEGMAX=int(g("SB_SEGMAX", "256")),
                MAXGAP=int(g("SB_MAXGAP", "800")), LR=float(g("SB_LR", "2e-3")), WARM=int(g("SB_WARM", "100")),
                LOGGAP=g("SB_LOGGAP") == "1", INIT=bool(g("SB_INIT")))


def tp_batch(rng, o):
    """train.py batch(), verbatim in its rng use (round-1 version when o["R1"])."""
    D = data()
    Tn = o["T"]
    if o.get("R1"):  # round-1 train.py (commit a2e3dfc): one vectorised draw per batch, no segments
        idx = rng.integers(0, len(D["lm"]) - Tn - 1, o["BS"])
        tok = torch.stack([torch.from_numpy(D["lm"][i:i + Tn + 1].astype(np.int64)) for i in idx])
        return tok, torch.zeros(o["BS"], Tn, dtype=torch.bool), None
    toks, masks, segs = [], [], []
    for _ in range(o["BS"]):
        if rng.random() < o["MIX"]:
            r, m = train_row(rng, D["rstream"], Tn + 1, o["MAXGAP"], o["LOGGAP"])
            toks.append(torch.from_numpy(r.astype(np.int64)))
            masks.append(torch.from_numpy(m[1:].astype(bool)))
        else:
            i = int(rng.integers(0, len(D["lm"]) - Tn - 1))
            toks.append(torch.from_numpy(D["lm"][i:i + Tn + 1].astype(np.int64)))
            masks.append(torch.zeros(Tn, dtype=torch.bool))
        if rng.random() < o["SEGP"]:
            b = np.cumsum(rng.integers(o["SEGMIN"], o["SEGMAX"] + 1, Tn // o["SEGMIN"] + 1))
            segs.append(torch.from_numpy(np.searchsorted(b, np.arange(Tn), side="right")))
        else:
            segs.append(torch.zeros(Tn, dtype=torch.long))
    return torch.stack(toks), torch.stack(masks), (torch.stack(segs) if o["SEGP"] > 0 else None)


def r3_batch(rng, stage):
    """train_r3.py worker rows, verbatim in their rng use."""
    D = data()
    Tn = r3.T
    rows = []
    for _ in range(r3.BS):
        if rng.random() < r3.LM_FRAC:
            i = int(rng.integers(0, len(D["lm"]) - Tn - 1))
            tok = D["lm"][i:i + Tn + 1]
            if rng.random() < 0.5:
                b = np.cumsum(rng.integers(32, 257, Tn // 32 + 1))
                seg = np.searchsorted(b, np.arange(Tn), side="right")
            else:
                seg = np.zeros(Tn, np.int64)
            rows.append((tok, np.zeros(Tn + 1, np.uint8), seg))
        else:
            si = stage if (stage == 0 or rng.random() >= r3.REHEARSAL) else int(rng.integers(0, stage))
            rows.append(r3.recall_row(rng, D["rstream"], si, handoff=rng.random() >= r3.FULLATTN_FRAC,
                                      alph=r3.STAGES[stage]["alph"]))
    tok = torch.from_numpy(np.stack([r[0] for r in rows]).astype(np.int64))
    msk = torch.from_numpy(np.stack([r[1][1:] for r in rows]).astype(bool))
    seg = torch.from_numpy(np.stack([r[2] for r in rows]).astype(np.int64))
    return tok, msk, seg


def src_phases():
    """The source's recorded history (rounds 1-3), in order."""
    P = []
    for name, js, seed in [("r1", "train_src.json", 0), ("r2a1", "train_src_r2a1.json", 0), ("r2", "train_src_r2.json", 10)]:
        d = json.load(open(js))
        o = tp_opts(d.get("options"))
        o["R1"] = name == "r1"
        lg = np.array([[e[0], e[1]] for e in d["log"]], dtype=np.float64)
        P.append(dict(name=name, kind="tp", seed=seed, rng_seed=seed + 1000 * o["INIT"], steps=d["steps"], opts=o,
                      budget_s=d["minutes"] * 60, log_it=lg[:, 0].tolist(), log_el=lg[:, 1].tolist(),
                      src_log0=d["log"][0], src_ckpt_in={"r1": None, "r2a1": "ckpt_src.pt", "r2": "ckpt_src_r2a1.pt"}[name]))
    for name, js in [("r3", "train_src_r3.json"), ("r3b", "train_src_r3b.json")]:
        d = json.load(open(js))
        adv = [h["step"] for h in d["curriculum"]["history"] if str(h.get("event", "")).startswith("advance")]
        P.append(dict(name=name, kind="r3", rng_seed=r3.SEEDS["src"], steps=d["steps"], advances=adv,
                      budget_s=d["options"]["budget_min"] * 60, log_it=[e[0] for e in d["log"]], log_el=[e[1] for e in d["log"]],
                      src_log25=d["log"][0], src_ckpt_in=d["options"]["init"]))
    return P


def phase_lr(ph, it, lr_base, warm):
    """LR at step index it (0-based) of a phase, as the original run computed it."""
    el = float(np.interp(it, ph["log_it"], ph["log_el"],
                         right=ph["log_el"][-1] + (it - ph["log_it"][-1]) * (ph["log_el"][-1] / max(ph["log_it"][-1], 1))))
    frac = min(1.0, el / ph["budget_s"])
    if ph["kind"] == "tp":
        return lr_base * min(1.0, (it + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
    sched = 1.0 if frac < r3.DECAY_FROM else 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * (frac - r3.DECAY_FROM) / (1 - r3.DECAY_FROM)))
    return lr_base * min(1.0, (it + 1) / warm) * sched


def stage_at(ph, b):
    """Stage used for batch b (1-based step count after the update) in a replayed r3 phase."""
    return sum(1 for a in ph["advances"] if a < b)


def make_opt(model, lr):
    decay = [p for _, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for _, p in model.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95))


def train_step(model, opt, lr, tok, msk, seg, valw):
    """One optimizer step on the whole batch, as train.py / train_r3.py: loss = mean CE over all
    bytes (+ valw x mean CE over value bytes), clip 1, AdamW. The batch is processed in micro-batches
    of MICRO_TOKENS // T rows with gradient accumulation (memory): the gradient is the full-batch
    gradient up to floating-point summation order."""
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad(set_to_none=True)
    B, Tn = tok.shape[0], tok.shape[1] - 1
    micro = max(1, MICRO_TOKENS // Tn)
    nall, nmask = B * Tn, int(msk.sum())
    use_v = valw > 0 and nmask > 0
    lsum = vsum = 0.0
    for a in range(0, B, micro):
        sl = slice(a, a + micro)
        lo, _, _ = model(tok[sl, :-1], seg=None if seg is None else seg[sl])
        ce = F.cross_entropy(lo.reshape(-1, 256), tok[sl, 1:].reshape(-1), reduction="none").view(-1, Tn)
        vs = ce[msk[sl]].sum()
        part = ce.sum() / nall + (valw * vs / nmask if use_v else 0.0)
        part.backward()
        lsum += float(ce.sum().detach())
        vsum += float(vs.detach())
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return lsum / nall, (vsum / nmask if nmask else None)


# ---------------------------------------------------------------- replay fidelity check (source)
def replay_check(n25=25):
    """The source, through this file's replay code, vs its own logged losses."""
    t0 = time.time()
    out = {}
    for ph in src_phases():
        if ph["src_ckpt_in"] is None:
            torch.manual_seed(0)
            m = HybridLM(**CONFIGS["src"])
        else:
            m = load_ckpt(ph["src_ckpt_in"])
        rng = np.random.default_rng(ph["rng_seed"])
        if ph["kind"] == "tp":
            o = ph["opts"]
            tok, msk, seg = tp_batch(rng, o)
            with torch.no_grad():
                lo, _, _ = m(tok[:, :-1], seg=seg)
                ce = F.cross_entropy(lo.reshape(-1, 256), tok[:, 1:].reshape(-1), reduction="none")
                loss = float(ce.mean())
                vl = float(ce[msk.reshape(-1)].mean()) if msk.any() else None
            out[ph["name"]] = {"step0_loss": round(loss, 4), "logged": ph["src_log0"][2],
                               "step0_value_loss": None if vl is None else round(vl, 4),
                               "logged_value": ph["src_log0"][3] if len(ph["src_log0"]) > 3 else None,
                               "match": bool(abs(loss - ph["src_log0"][2]) < 2e-3)}
        elif n25 == 0:
            out[ph["name"]] = {"skipped": "smoke test: forward-only checks", "match": True}
        else:
            opt = make_opt(m, r3.LR["src"])
            ema = vema = None
            for it in range(n25):  # through train_step (micro-batched), as the replayed target is trained
                tok, msk, seg = r3_batch(rng, stage_at(ph, it + 1))
                loss, vloss = train_step(m, opt, phase_lr(ph, it, r3.LR["src"], r3.WARM), tok, msk, seg, r3.VALW)
                ema = loss if ema is None else 0.98 * ema + 0.02 * loss
                if vloss is not None:
                    vema = vloss if vema is None else 0.95 * vema + 0.05 * vloss
            lg = ph["src_log25"]
            out[ph["name"]] = {"ema25_loss": round(ema, 4), "logged": lg[3], "ema25_value_loss": round(vema, 4),
                               "logged_value": lg[4], "match": bool(abs(ema - lg[3]) < 5e-3 and abs(vema - lg[4]) < 5e-2)}
        log("replay check", ph["name"], out[ph["name"]])
    out["runtime_s"] = round(time.time() - t0, 1)
    out["r2a1_note"] = "r2a1 is known not to reproduce (pre-commit code); excluded from all_match"
    out["all_match"] = all(v["match"] for k, v in out.items() if isinstance(v, dict) and k != "r2a1")
    return out


def replay_rng_after(ph):
    """numpy rng state after the source's whole r3-type phase (no model involved)."""
    rng = np.random.default_rng(ph["rng_seed"])
    for it in range(ph["steps"]):
        r3_batch(rng, stage_at(ph, it + 1))
    return rng


# ---------------------------------------------------------------- training driver (resumable)
def diag_copy_r2(src, t):
    """Grow pair only: R^2 of the source SSM state vs the copied target layer's heads 0..7 (16 gate docs, T-32)."""
    docs, _ = eval_docs(streams()["calib"], 16, T, seed=1, nbind=NB)
    tok = sb2.tt(docs)
    with torch.no_grad():
        _, ss, _ = src(tok[:, :T - PW])
        _, st, _ = t(tok[:, :T - PW])
    out = {}
    for ti, si in GROW_PROV.items():
        if t.pattern[ti] == "M":
            a = ss[si][0].reshape(len(tok), -1)
            b = st[ti][0][:, :src.layers[si].H].reshape(len(tok), -1)
            out[str(ti)] = round(float(1 - ((a - b) ** 2).sum() / ((a - a.mean(0)) ** 2).sum()), 4)
    return out


def run_pair(pair):
    rpath = f"{WD}/ckpt_resume_family_{pair}_r7.pt"
    final = f"{WD}/ckpt_tgt_{pair}_r7.pt"
    jpath = f"{WD}/train_tgt_{pair}_r7.json"
    if os.path.exists(final) and os.path.exists(jpath) and json.load(open(jpath)).get("done"):
        log(f"{pair}: already done ({final}); nothing to do")
        return json.load(open(jpath))
    if SMOKE and not os.path.exists(SRC_CKPT):
        torch.manual_seed(0)
        save_model(HybridLM(**SRC_CFG), SRC_CFG, SRC_CKPT)
    src = load_ckpt(SRC_CKPT).eval()
    for p in src.parameters():
        p.requires_grad_(False)
    phases = src_phases()
    if SMOKE:
        for ph in phases:
            ph["steps"] = 3
            if ph["kind"] == "tp":
                ph["opts"]["T"] = 256  # smoke only: attention memory grows with T^2
            if ph["kind"] == "r3":
                ph["advances"] = [a for a in ph["advances"] if a < 3] + [1]
    # plan: list of segments (name, kind, steps, ...) executed in order
    if pair == "grow":
        r3b = phases[-1]
        plan = [dict(name="grow", kind="r3", steps=GROW_STEPS, lr=TGT_LR3, warm=TGT_WARM3, stage=len(r3.STAGES) - 1,
                     cosine_tail=True)]
        cfg = GROW_CFG
    else:
        r3b = phases[-1]
        plan = []
        for ph in phases:
            if ph["kind"] == "tp":
                plan.append(dict(name=ph["name"], kind="tp", steps=ph["steps"], ph=ph))
            else:
                plan.append(dict(name=ph["name"], kind="r3", steps=ph["steps"], ph=ph, lr=TGT_LR3, warm=TGT_WARM3))
        cfg = TGT_CFG
    ext = lambda n: dict(name=f"ext{n}", kind="r3", steps=EXT_BLOCK, lr=TGT_LR3, warm=TGT_WARM3,  # noqa: E731
                         stage=len(r3.STAGES) - 1)
    if os.path.exists(rpath):
        R = torch.load(rpath, weights_only=False)
        model = HybridLM(**cfg)
        model.load_state_dict(R["model"])
        J = R["J"]
        seg_i, it = J["seg_i"], J["it"]
        opt = None
        if R["opt"] is not None:
            opt = make_opt(model, 1e-3)
            opt.load_state_dict(R["opt"])
        rng = None
        if R["rng"] is not None:
            rng = np.random.default_rng()
            rng.bit_generator.state = R["rng"]
        del R
        log(f"{pair}: resumed at segment {seg_i} step {it}", mem())
    else:
        J = {"pair": pair, "cfg": cfg, "segments": [], "gates": [], "checks": {}, "snapshots": {}, "log": [], "done": False,
             "threads": NTH, "smoke": SMOKE}
        if pair == "grow":
            model = grow(src)
            chk = growth_check(src, model)
            J["checks"]["growth"] = chk
            log("growth check", chk)
            if not chk["pass"]:
                raise RuntimeError(f"growth is not function-preserving: {chk}")
            J["snapshots"]["0"] = {"copy_r2": diag_copy_r2(src, model)}
            save_model(model, cfg, f"{WD}/ckpt_tgt_grow_r7_s0.pt")
        else:
            model, prov = replay_init()
            J["checks"]["shared_init_prov"] = {str(k): v for k, v in prov.items()}
        J["checks"]["replay_fidelity"] = replay_check(0 if SMOKE else 25)
        J["n_params"] = sum(p.numel() for p in model.parameters())
        seg_i, it, opt, rng = 0, 0, None, None
    plan += [ext(n + 1) for n in range(sum(1 for g in J["gates"] if not g["pass"]))]
    t_saved = time.time()

    def save_resume():
        J.update(seg_i=seg_i, it=it)
        torch.save({"model": model.state_dict(), "opt": None if opt is None else opt.state_dict(),
                    "rng": None if rng is None else rng.bit_generator.state, "J": J}, rpath + ".tmp")
        os.replace(rpath + ".tmp", rpath)

    def start_segment(sg):
        """Fresh AdamW per segment (as the original phases); the segment's own data stream."""
        if sg["kind"] == "tp":
            o = sg["ph"]["opts"]
            new_rng = np.random.default_rng(sg["ph"]["rng_seed"])
            new_opt = make_opt(model, o["LR"])
        elif sg["name"] == "grow":
            log("grow: replaying the source's r3b data stream (no model) to continue it")
            new_rng, new_opt = replay_rng_after(r3b), make_opt(model, sg["lr"])
        elif sg["name"].startswith("ext"):
            new_rng, new_opt = rng, make_opt(model, sg["lr"])  # the current stream continues
        else:
            new_rng, new_opt = np.random.default_rng(sg["ph"]["rng_seed"]), make_opt(model, sg["lr"])
        J["segments"].append({"name": sg["name"], "steps": sg["steps"], "started": time.strftime("%Y-%m-%d %H:%M:%S")})
        log(f"{pair}: segment {sg['name']} ({sg['steps']} steps) start", mem())
        return new_opt, new_rng

    model.train()
    t_run = time.time()
    while True:
        if seg_i >= len(plan):  # every planned segment done: gate
            model.eval()
            write_status(phase="gate", pair=pair)
            g = gate(model)
            gs = gate_summary(g)
            J["gates"].append(dict(after=plan[-1]["name"], **gs, detail=g))
            log(f"{pair} GATE after {plan[-1]['name']}:", gs)
            if g["pass"] or sum(1 for x in J["gates"] if not x["pass"]) > EXT_MAX:
                break
            plan.append(ext(sum(1 for x in J["gates"] if not x["pass"])))
            model.train()
            save_resume()
            continue
        sg = plan[seg_i]
        if opt is None:
            opt, rng = start_segment(sg)
        while it < sg["steps"]:
            ts = time.time()
            if sg["kind"] == "tp":
                o = sg["ph"]["opts"]
                lr = phase_lr(sg["ph"], it, o["LR"], o["WARM"])
                tok, msk, seg = tp_batch(rng, o)
                valw = o["VALW"]
            else:
                if "ph" in sg:  # replayed r3/r3b: the source's lr profile and stage schedule
                    lr = phase_lr(sg["ph"], it, sg["lr"], sg["warm"])
                    stage = stage_at(sg["ph"], it + 1)
                else:  # grow / extension: final stage, step-based schedule
                    frac = it / sg["steps"]
                    tail = sg.get("cosine_tail") and frac >= r3.DECAY_FROM
                    sched = 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * (frac - r3.DECAY_FROM) / (1 - r3.DECAY_FROM))) if tail else 1.0
                    lr = sg["lr"] * min(1.0, (it + 1) / sg["warm"]) * sched
                    stage = sg["stage"]
                tok, msk, seg = r3_batch(rng, stage)
                valw = r3.VALW
            loss, vloss = train_step(model, opt, lr, tok, msk, seg, valw)
            it += 1
            J["log"].append([sg["name"], it, round(loss, 4), None if vloss is None else round(vloss, 4), round(lr, 7),
                             round(time.time() - ts, 2)])
            if it % LOG_EVERY == 0:
                rec = J["log"][-LOG_EVERY:]
                vl = [x[3] for x in rec if x[3] is not None]
                sps = float(np.mean([x[5] for x in rec]))
                log(f"{pair} {sg['name']} step {it}/{sg['steps']} loss {np.mean([x[2] for x in rec]):.4f} "
                    f"value {np.mean(vl) if vl else float('nan'):.4f} lr {lr:.2e} {sps:.2f} s/step")
                left = sg["steps"] - it + sum(p["steps"] for p in plan[seg_i + 1:])
                write_status(phase="train", pair=pair, segment=sg["name"], step=it, seg_steps=sg["steps"],
                             steps_left_in_plan=left, sec_per_step=round(sps, 2), eta_s=round(left * sps))
            if pair == "grow" and sg["name"] == "grow" and it in GROW_SNAPS:
                model.eval()
                sn = {"copy_r2": diag_copy_r2(src, model), "gate16": gate_summary(gate(model, 16))}
                J["snapshots"][str(it)] = sn
                log(f"grow snapshot at step {it}", sn)
                save_model(model, cfg, f"{WD}/ckpt_tgt_grow_r7_s{it}.pt")
                model.train()
            if it % SAVE_EVERY == 0 or time.time() - t_saved > 600:
                save_resume()
                t_saved = time.time()
        J["segments"][-1]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        seg_i, it, opt = seg_i + 1, 0, None
        save_resume()
    J["done"] = True
    J["final_gate_pass"] = J["gates"][-1]["pass"]
    J["train_minutes"] = round((time.time() - t_run) / 60, 1)
    save_model(model, cfg, final, {"gate_pass": J["final_gate_pass"]})
    atomic_json(jpath, J)
    if os.path.exists(rpath):
        os.remove(rpath)
    log(f"{pair} FAMILY_DONE: gate pass {J['final_gate_pass']}; {final}", mem())
    return J


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "grow"
    assert what in ("grow", "replay", "check"), "usage: train_family.py {grow|replay|check}"
    os.makedirs(WD, exist_ok=True)
    take_lock()
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGHUP, on_term)
    STATUS.update(state="running", t_start=time.time(), pid=os.getpid(), threads=NTH, pair=what, smoke=SMOKE,
                  started=time.strftime("%Y-%m-%d %H:%M:%S"), log=LOG,
                  expected_runtime={"grow": "about 45 min at 2 threads (1,000 steps at ~2.5 s/step, plus snapshots and gate)",
                                    "replay": "about 2 h at 2 threads (2,568 replayed steps, r1 at T 2048, plus gate)",
                                    "check": "about 2 min"}[what])
    if os.path.exists(STATUS_LOCAL):
        old = json.load(open(STATUS_LOCAL))
        STATUS["previous_launches"] = old.get("previous_launches", []) + [
            {k: old.get(k) for k in ("started", "state", "pair", "updated", "segment", "step", "error")}]
    log(f"train_family start: {what}, pid {os.getpid()}, threads {NTH}, smoke {SMOKE}", mem())
    write_status(phase="setup")
    try:
        if what == "check":
            res = replay_check(0 if SMOKE else 25)
            write_status(state="done", phase="done", replay_check=res)
        else:
            J = run_pair(what)
            write_status(state="done", phase="done", final_gate_pass=J.get("final_gate_pass"),
                         final_gate=J["gates"][-1] if J.get("gates") else None)
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
