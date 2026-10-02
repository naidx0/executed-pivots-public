#!/usr/bin/env python3
"""StateBridge real-model check: do a small SSM's states map onto a bigger same-family SSM's?

Why. Five toy rounds found that a map from an independently trained small model's SSM state to a
bigger model's does not match running the small model alone. The main untested caveat is family:
real same-family models (Nemotron Nano / Super) share tokenizer, data and recipe. Nano (30B-A3B)
and Super do NOT fit an 8 GB GPU for state extraction, so this runs the cheapest real-scale version
on the best same-family pure-SSM pair that fits and that Hugging Face transformers supports with
accessible SSM states (out.cache_params.ssm_states): Mamba-2 130m -> 370m by default (same Pile
data, same GPT-NeoX tokenizer). Forward passes and ridge fits only; no training.

What it does (protocol of the toy rounds 3-5, adapted):
1. Prompts. Synthetic key-value recall built at token level (equal lengths): intro, nb bindings
   "<key>: <value>" (single-token words), a natural-text gap, then the keys again in random order;
   the model must produce each value (primary metric: CE and top-1 accuracy on the value tokens).
   Also natural-text windows (ctx tokens, then cont scored tokens; metric: CE on the continuation).
   Calibration prompts (fit only) and test prompts come from disjoint text and seeds.
2. Handoff point p = w tokens before the queries (natural text: before the continuation), w = 32.
   "Window" arms re-read tokens [p, end) one token at a time through the model's own cache decode
   path, starting from a given SSM state in every layer, conv state zero (the toy protocol).
   Arms: full prefill (target, and source = "source alone"); target oracle (its own SSM state at p);
   blank (zero state); mismatched-own (own state of the next prompt in the batch); bridge (source
   state at p mapped by per-layer ridge); bridge-mismatched (same map, next prompt's source state);
   source oracle (source's own state, diagnostic).
3. Ridge. Target layer j <- source layer round(j (Ls-1)/(Lt-1)). Features: the source layer's
   flattened SSM state, standardised on calibration samples; target: the flattened target state.
   States are far too wide for a primal map (e.g. 196,608 -> 262,144 per layer), so the ridge is
   fitted in its exact dual (kernel) form: pred = K_test (K + lam n I)^-1 (Y - mean) + mean, with
   lam chosen per layer from {1e-4..10} by validation R^2 (every 5th calibration prompt held out),
   then refitted on all calibration samples. Calibration samples: several offsets before the
   handoff point per calibration prompt. Layers are processed in groups sized to --ram-gb.
4. Self-check (API): re-reading the window from the target's own state WITH its true conv state
   must reproduce full-prefill logits; a mismatch means the transformers cache API differs from what
   this script assumes, and the results are then invalid (flagged in the JSON).
5. Stats: paired bootstrap over test prompts (10k resamples, 95% CIs).

Reading (value CE at w = 32, recall prompts):
- gate: oracle beats blank and mismatched-own by >= 0.5 nats with CIs below 0. If not, the SSM state
  alone carries little recall at this scale, and no SSM-state bridge can work here, whatever the map.
- bridge "beats all three": bridge minus blank, minus bridge-mismatched and minus source alone, each
  CI below 0. The document-specific gap is bridge minus bridge-mismatched. A clear document-specific
  gap here, unlike in the toy, would justify a GPU run with a trained bridge on the real pair.

Verified vs assumed. Written without network access to Hugging Face and without transformers
installed; the logic was tested against a small stand-in module with the same interface. ASSUMED
(unverified): repo ids AntonV/mamba2-{130m,370m,780m,1.3b}-hf (HF-format Mamba-2 conversions; the
script falls back to state-spaces/mamba-{130m,370m}-hf, Mamba-1, if they cannot be loaded);
out.cache_params exposes ssm_states[i] and conv_states[i] (dict, list or stacked tensor); passing
cache_params + cache_position (> 0) with one token runs the recurrent decode step. The self-check
tests these at runtime.

Usage:  python run_real_pair.py [--pair auto] [--out-dir statebridge_real] [--gpu-lock PATH]
Needs: torch (CUDA build for the GPU), transformers; optional: datasets (wikitext-2 as natural text;
falls back to Python standard-library source text). Writes <out-dir>/results.json and log.txt.
"""
import argparse
import atexit
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

PAIRS = {
    "mamba2-130m-370m": ("AntonV/mamba2-130m-hf", "AntonV/mamba2-370m-hf"),
    "mamba2-370m-780m": ("AntonV/mamba2-370m-hf", "AntonV/mamba2-780m-hf"),
    "mamba2-780m-1.3b": ("AntonV/mamba2-780m-hf", "AntonV/mamba2-1.3b-hf"),
    "mamba1-130m-370m": ("state-spaces/mamba-130m-hf", "state-spaces/mamba-370m-hf"),
    "mamba1-370m-790m": ("state-spaces/mamba-370m-hf", "state-spaces/mamba-790m-hf"),
}
AUTO = ["mamba2-130m-370m", "mamba1-130m-370m"]
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
LOGF = None


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOGF:
        with open(LOGF, "a", encoding="utf-8") as f:
            f.write(s + "\n")


def args_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="auto", help="auto | " + " | ".join(PAIRS))
    ap.add_argument("--small"), ap.add_argument("--big")
    ap.add_argument("--tokenizer", help="default: the small model's own, else EleutherAI/gpt-neox-20b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--out-dir", default="statebridge_real")
    ap.add_argument("--n-cal-kv", type=int, default=256), ap.add_argument("--n-test-kv", type=int, default=128)
    ap.add_argument("--n-cal-nat", type=int, default=64), ap.add_argument("--n-test-nat", type=int, default=64)
    ap.add_argument("--nb", type=int, default=4, help="bindings per recall prompt")
    ap.add_argument("--gap", type=int, default=200, help="natural-text tokens between bindings and queries")
    ap.add_argument("--w", type=int, default=32, help="handoff window (tokens re-read by the target)")
    ap.add_argument("--cal-offsets", default="8,32,64,128", help="calibration state positions before the queries")
    ap.add_argument("--ctx", type=int, default=512), ap.add_argument("--cont", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16, help="prefill batch for state collection")
    ap.add_argument("--eval-batch", type=int, default=8, help="batch for the window arms (holds full state sets)")
    ap.add_argument("--ram-gb", type=float, default=6.0, help="CPU RAM budget for one layer group of states")
    ap.add_argument("--text-file", help="natural text (UTF-8) instead of wikitext-2 / stdlib source")
    ap.add_argument("--gpu-lock", default=os.environ.get("GPU_LOCK"), help="lock file shared with other GPU users")
    ap.add_argument("--min-free-gb", type=float, default=6.0, help="wait until this much VRAM is free")
    ap.add_argument("--max-wait-h", type=float, default=6.0)
    ap.add_argument("--chunk-size", type=int, default=0, help="override the Mamba-2 scan chunk size (0 keeps the model's)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ---------------------------------------------------------------- GPU etiquette
def gpu_guard(a):
    """Wait until the shared lock file is absent and enough VRAM is free (Ollama may hold models),
    then take the lock (atomic create) for the whole run; released at exit."""
    if not a.device.startswith("cuda"):
        return
    t0 = time.time()
    while True:
        busy = []
        if a.gpu_lock and os.path.exists(a.gpu_lock):
            try:
                busy.append(f"lock {a.gpu_lock} held: {open(a.gpu_lock).read()[:160]!r}")
            except OSError:
                busy.append(f"lock {a.gpu_lock} held")
        free = torch.cuda.mem_get_info()[0] / 1e9
        if free < a.min_free_gb:
            busy.append(f"{free:.1f} GB VRAM free < {a.min_free_gb} GB")
        if not busy and a.gpu_lock:
            try:
                fd = os.open(a.gpu_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                busy.append("lock taken meanwhile")
            else:
                os.write(fd, (f"lane=executed-pivots since={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                              f"pid={os.getpid()} what=statebridge run_real_pair.py\n").encode())
                os.close(fd)
                atexit.register(lambda: os.path.exists(a.gpu_lock) and os.remove(a.gpu_lock))
        if not busy:
            log(f"GPU ok: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB free; lock {a.gpu_lock or '(none given)'}")
            return
        if time.time() - t0 > a.max_wait_h * 3600:
            sys.exit("GPU still busy after --max-wait-h: " + "; ".join(busy))
        log("waiting for the GPU:", "; ".join(busy))
        time.sleep(30)


# ---------------------------------------------------------------- models, cache access
def load_pair(a, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def get_tok(i):
        for t in ([a.tokenizer] if a.tokenizer else [i, "EleutherAI/gpt-neox-20b"]):
            try:
                return AutoTokenizer.from_pretrained(t)
            except Exception as e:
                log(f"tokenizer {t}: {e!r}")
        raise RuntimeError("no tokenizer")

    def get_model(i):
        try:
            m = AutoModelForCausalLM.from_pretrained(i, dtype=dtype)
        except TypeError:  # older transformers
            m = AutoModelForCausalLM.from_pretrained(i, torch_dtype=dtype)
        if a.chunk_size:  # the pure-torch Mamba-2 path (no mamba_ssm kernels, e.g. Windows) scales with chunk^2
            for mod in m.modules():
                if hasattr(mod, "chunk_size"):
                    mod.chunk_size = a.chunk_size
        return m.to(a.device).eval()

    names = [a.pair] if a.pair != "auto" else AUTO
    for name in names:
        ids = (a.small, a.big) if a.small and a.big else PAIRS[name]
        try:
            tok = get_tok(ids[0])
            ms = [get_model(i) for i in ids]
            tok_b = get_tok(ids[1])
            probe = "The quick brown fox: 12345\nkey = value"
            assert tok.encode(probe) == tok_b.encode(probe), "tokenizers differ: the handoff needs one token sequence"
            log(f"loaded {name}: {ids[0]} -> {ids[1]}")
            return name, ids, tok, ms[0], ms[1]
        except Exception as e:  # repo missing, unsupported arch, ...
            log(f"could not load {name} ({ids}): {e!r}")
            if a.pair != "auto":
                raise
    sys.exit("no pair could be loaded")


def n_layers(m):
    return m.config.num_hidden_layers


def cache_get(cache, name, i):
    if not hasattr(cache, name):
        raise RuntimeError(f"cache object {type(cache).__name__} has no '{name}' (attributes: "
                           f"{[k for k in dir(cache) if not k.startswith('_')]}); transformers cache API changed")
    return getattr(cache, name)[i]


def cache_set(cache, name, i, t):
    dst = getattr(cache, name)[i]
    if t is None:
        dst.zero_()
    else:
        dst.copy_(t.to(dst.dtype).reshape(dst.shape))


@torch.no_grad()
def prefill(model, ids):
    out = model(input_ids=ids, use_cache=True)
    return out.cache_params, out.logits


@torch.no_grad()
def ssm_states(model, ids, layers):
    cache, _ = prefill(model, ids)
    return {i: cache_get(cache, "ssm_states", i).reshape(len(ids), -1).clone() for i in layers}, cache


@torch.no_grad()
def run_window(model, ids, p, states, score_pos, keep_conv_from=None):
    """Re-read ids[:, p:-1] one token at a time from the given per-layer SSM states (None = zeros),
    conv zero (or the conv state of keep_conv_from). Returns per-prompt CE and correct at score_pos,
    and the raw logits at score_pos."""
    B = ids.shape[0]
    cache, _ = prefill(model, ids[:, :1])  # a cache object of the right shape; its content is overwritten
    for i in range(n_layers(model)):
        cache_set(cache, "ssm_states", i, None if states is None else states[i])
        cache_set(cache, "conv_states", i, None if keep_conv_from is None else cache_get(keep_conv_from, "conv_states", i))
    ce, ok, lg = [], [], []
    for t in range(p, ids.shape[1] - 1):
        out = step(model, ids[:, t:t + 1], cache, t)
        cache = out.cache_params
        if t + 1 in score_pos:
            lo = out.logits[:, -1].float()
            ce.append(F.cross_entropy(lo, ids[:, t + 1], reduction="none"))
            ok.append((lo.argmax(-1) == ids[:, t + 1]).float())
            lg.append(lo)
    return torch.stack(ce, 1).mean(1).cpu(), torch.stack(ok, 1).mean(1).cpu(), torch.stack(lg, 1)


_NO_CACHE_POSITION = []


def step(model, tok1, cache, t):
    """One recurrent decode step from `cache` (decode path: cache_position > 0 in recent transformers,
    cache.seqlen_offset > 0 in older ones, which also lack the cache_position argument)."""
    if not _NO_CACHE_POSITION:
        try:
            return model(input_ids=tok1, cache_params=cache, use_cache=True, cache_position=torch.tensor([t], device=tok1.device))
        except TypeError:
            _NO_CACHE_POSITION.append(True)
            log("model.forward takes no cache_position: using the cache's seqlen_offset path")
    return model(input_ids=tok1, cache_params=cache, use_cache=True)


@torch.no_grad()
def full_scores(model, ids, score_pos):
    lo = model(input_ids=ids[:, :-1]).logits.float()
    idx = torch.tensor(sorted(score_pos), device=ids.device) - 1
    lo = lo[:, idx]
    tg = ids[:, idx + 1]
    ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction="none").view(tg.shape)
    return ce.mean(1).cpu(), (lo.argmax(-1) == tg).float().mean(1).cpu(), lo


# ---------------------------------------------------------------- prompts (token level)
def load_text(a):
    if a.text_file:
        return open(a.text_file, encoding="utf-8").read(), a.text_file
    for name in [("wikitext", "wikitext-2-raw-v1"), ("Salesforce/wikitext", "wikitext-2-raw-v1")]:
        try:
            from datasets import load_dataset
            d = load_dataset(*name)
            return "\n".join(d["train"]["text"] + d["validation"]["text"] + d["test"]["text"]), "/".join(name)
        except Exception as e:
            log(f"datasets {name} unavailable: {e!r}")
    import glob
    import sysconfig
    fs = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    return "\n".join(open(f, encoding="utf-8", errors="ignore").read() for f in fs), "python-stdlib-source"


def build_prompts(a, tok):
    rng = np.random.default_rng(a.seed)
    text, src = load_text(a)
    stream = np.array(tok.encode(text[:6_000_000]), dtype=np.int64)
    cut = int(len(stream) * 0.7)
    streams = {"cal": stream[:cut], "test": stream[cut:]}  # disjoint text for calibration and test
    one = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    colon, nl = one(":"), one("\n")
    assert len(colon) == 1 and len(nl) == 1, "':' and newline must be single tokens"
    colon, nl = colon[0], nl[0]
    words = []
    for s, i in tok.get_vocab().items():
        w = tok.convert_tokens_to_string([s])
        if w.startswith(" ") and w[1:].isalpha() and w[1:].islower() and 4 <= len(w) - 1 <= 8 and one(w) == [i]:
            words.append(i)
    words = sorted(words)
    rng.shuffle(words)
    assert len(words) >= 400, f"only {len(words)} single-token words"
    keys, vals = words[: len(words) // 2], words[len(words) // 2:]
    intro = one("Remember these codes.\n")
    offs = [int(x) for x in a.cal_offsets.split(",")]

    def kv(part, n, seed, gap=None):
        gap = a.gap if gap is None else gap
        r = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            k = r.choice(keys, a.nb, replace=False)
            v = r.choice(vals, a.nb)
            ids = list(intro)
            for kk, vv in zip(k, v):
                ids += [kk, colon, vv, nl]
            s = int(r.integers(0, len(streams[part]) - gap - 1))
            ids += streams[part][s:s + gap].tolist() + [nl]
            q0 = len(ids)
            for i in r.permutation(a.nb):
                ids += [k[i], colon, v[i], nl]
            out.append(ids)
        score = {q0 + 4 * i + 2 for i in range(a.nb)}  # value tokens (same positions in every prompt)
        return np.array(out), q0, score

    def nat(part, n, seed):
        r = np.random.default_rng(seed)
        L = a.ctx + a.cont
        st = r.integers(0, len(streams[part]) - L, n)
        return np.stack([streams[part][s:s + L] for s in st]), a.ctx, set(range(a.ctx, L))

    P = {}
    for task, fn, nc, nt in [("kv", kv, a.n_cal_kv, a.n_test_kv), ("nat", nat, a.n_cal_nat, a.n_test_nat)]:
        ids_c, q0, score = fn("cal", nc, a.seed + 1 if task == "kv" else a.seed + 3)
        ids_t, _, _ = fn("test", nt, a.seed + 2 if task == "kv" else a.seed + 4)
        P[task] = {"cal": ids_c, "test": ids_t, "q0": q0, "score": score, "p": q0 - a.w,
                   "cal_pos": [q0 - o for o in offs if q0 - o > 0]}
    P["meta"] = {"text_source": src, "n_words": len(words), "stream_tokens": int(len(stream))}
    P["gap_diag"] = {g: kv("test", 32, a.seed + 10 + g) for g in sorted({0, 50, a.gap})}
    return P


def samples(P):
    """Calibration samples (prompt, position) then test samples (handoff point), with prompt ids for
    the validation split."""
    cal, test, pid = [], [], []
    for ti, task in enumerate(["kv", "nat"]):
        for r in range(len(P[task]["cal"])):
            for pos in P[task]["cal_pos"]:
                cal.append((task, "cal", r, pos))
                pid.append(ti * 100000 + r)
        for r in range(len(P[task]["test"])):
            test.append((task, "test", r, P[task]["p"]))
    return cal, test, np.array(pid)


def collect(model, P, smp, layers, a):
    """States of `layers` for every sample, fp16 on CPU: {layer: (n, d)}."""
    out = {}
    groups = {}
    for n, (task, part, r, pos) in enumerate(smp):
        groups.setdefault((task, part, pos), []).append((n, r))
    for (task, part, pos), items in groups.items():
        for b in range(0, len(items), a.batch):
            chunk = items[b:b + a.batch]
            ids = torch.tensor(P[task][part][[r for _, r in chunk]][:, :pos], device=a.device)
            st, _ = ssm_states(model, ids, layers)
            rows = torch.tensor([n for n, _ in chunk])
            for i in layers:
                if i not in out:
                    out[i] = torch.empty(len(smp), st[i].shape[1], dtype=torch.float16)
                out[i][rows] = st[i].half().cpu()
    return out


# ---------------------------------------------------------------- ridge (dual form)
def kernel(X, n_cal, dev, ch=16384):
    """Linear kernel of per-feature standardised states (stats from calibration rows)."""
    N, d = X.shape
    K = torch.zeros(N, N, dtype=torch.float64, device=dev)
    for c in range(0, d, ch):
        x = X[:, c:c + ch].to(dev).float()
        m, s = x[:n_cal].mean(0), x[:n_cal].std(0) + 1e-6
        z = (x - m) / s
        K += (z @ z.T).double()
    return K


def solve_mat(Kab, Kbb, lam):
    e, U = torch.linalg.eigh(Kbb)
    return ((Kab @ U) / (e + lam * len(Kbb))) @ U.T


def fit_layer(K, Y, n_cal, val, dev, pred_out, ch=16384):
    """Choose lam by val R^2, refit on all calibration rows, write test predictions (fp16) into
    pred_out (n_test, dy). Returns info."""
    fi, vi = np.where(~val)[0], np.where(val)[0]
    ti = np.arange(n_cal, K.shape[0])
    fi_t, vi_t, ti_t = (torch.tensor(x, device=dev) for x in (fi, vi, ti))
    Kff, Kvf = K[fi_t][:, fi_t], K[vi_t][:, fi_t]
    A = {lam: solve_mat(Kvf, Kff, lam).float() for lam in LAMBDAS}
    cal_t = torch.arange(n_cal, device=dev)
    M = None
    sse = {lam: 0.0 for lam in LAMBDAS}
    sst = 0.0
    for c in range(0, Y.shape[1], ch):
        y = Y[:n_cal, c:c + ch].to(dev).float()
        yf, yv = y[fi_t], y[vi_t]
        mf = yf.mean(0)
        sst += float(((yv - mf) ** 2).sum())
        for lam in LAMBDAS:
            sse[lam] += float(((A[lam] @ (yf - mf) + mf - yv) ** 2).sum())
    r2 = {lam: 1 - sse[lam] / sst for lam in LAMBDAS}
    lam = max(r2, key=r2.get)
    M = solve_mat(K[ti_t][:, cal_t], K[cal_t][:, cal_t], lam).float()
    sse_t = sst_t = 0.0
    for c in range(0, Y.shape[1], ch):
        y = Y[:n_cal, c:c + ch].to(dev).float()
        my = y.mean(0)
        pr = M @ (y - my) + my
        pred_out[:, c:c + ch] = pr.half().cpu()
        yt = Y[n_cal:, c:c + ch].to(dev).float()
        sse_t += float(((pr - yt) ** 2).sum())
        sst_t += float(((yt - my) ** 2).sum())
    return {"lambda": lam, "val_r2": r2[lam], "val_r2_path": {str(k): v for k, v in r2.items()}, "test_r2": 1 - sse_t / sst_t}


def layer_groups(layers, dim, n, ram_gb):
    per = max(1, int(ram_gb * 1e9 // (dim * n * 2)))
    return [layers[i:i + per] for i in range(0, len(layers), per)]


# ---------------------------------------------------------------- statistics
def boot(d, n=10000, seed=0):
    d = np.asarray(d, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return [float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    global LOGF
    a = args_()
    os.makedirs(a.out_dir, exist_ok=True)
    LOGF = os.path.join(a.out_dir, "log.txt")
    t0 = time.time()
    torch.manual_seed(a.seed)
    log("start", vars(a))
    gpu_guard(a)
    dtype = getattr(torch, a.dtype)
    name, ids, tok, ms, mt = load_pair(a, dtype)
    Ls, Lt = n_layers(ms), n_layers(mt)
    pair = {j: round(j * (Ls - 1) / (Lt - 1)) for j in range(Lt)}
    P = build_prompts(a, tok)
    cal, test, pid = samples(P)
    n_cal, n_test = len(cal), len(test)
    val = (pid % 5) == 4
    smp = cal + test
    log(f"prompts: {P['meta']}; recall prompt length {P['kv']['cal'].shape[1]}, calibration samples {n_cal}, test {n_test}")
    dims, gap_diag = {}, {}
    for tag, m in (("s", ms), ("t", mt)):
        st, cache = ssm_states(m, torch.tensor(P["kv"]["cal"][:1, :8], device=a.device), [0])
        dims[tag], dims[tag + "_shape"] = st[0].shape[1], list(cache_get(cache, "ssm_states", 0).shape)
    log("state dims per layer", dims, "pairing", pair)
    for g, (x_, _, sc_) in P["gap_diag"].items():  # does full-prefill recall exist at all, by gap?
        x_ = torch.tensor(x_, device=a.device)
        gap_diag[str(g)] = {tag: round(float(torch.cat([full_scores(m, x_[i:i + a.batch], sc_)[1]
                                                        for i in range(0, len(x_), a.batch)]).mean()), 4)
                            for tag, m in (("small", ms), ("big", mt))}
    log("full-prefill value accuracy by gap (32 prompts)", gap_diag)
    # source kernels (all source layers), layer group by group
    Ks, t1 = {}, time.time()
    for g in layer_groups(list(range(Ls)), dims["s"], n_cal + n_test, a.ram_gb):
        X = collect(ms, P, smp, g, a)
        for i in g:
            Ks[i] = kernel(X[i], n_cal, a.device).cpu()
        del X
        log(f"source layers {g[0]}-{g[-1]}: kernels done ({time.time() - t1:.0f}s)")
    # target: fit per layer, predictions for the test samples (disk-backed, fp16)
    pred = np.lib.format.open_memmap(os.path.join(a.out_dir, "pred_states.npy"), mode="w+", dtype=np.float16,
                                     shape=(n_test, Lt * dims["t"]))
    info = {}
    for g in layer_groups(list(range(Lt)), dims["t"], n_cal + n_test, a.ram_gb):
        Y = collect(mt, P, smp, g, a)
        for j in g:
            out = torch.empty(n_test, dims["t"], dtype=torch.float16)
            info[j] = fit_layer(Ks[pair[j]].to(a.device), Y[j], n_cal, val, a.device, out)
            info[j]["src_layer"] = pair[j]
            pred[:, j * dims["t"]:(j + 1) * dims["t"]] = out.numpy()
        del Y
        log(f"target layers {g[0]}-{g[-1]} fitted ({time.time() - t1:.0f}s):",
            {j: (info[j]["lambda"], round(info[j]["val_r2"], 3), round(info[j]["test_r2"], 3)) for j in g})
    pred.flush()
    # evaluation on test prompts
    res = {"kv": {}, "nat": {}}
    selfcheck = None
    trow = {(task, r): n for n, (task, _, r, _) in enumerate(test)}
    for task in ("kv", "nat"):
        T = P[task]
        for b in range(0, len(T["test"]), a.eval_batch):
            rows = list(range(b, min(b + a.eval_batch, len(T["test"]))))
            x = torch.tensor(T["test"][rows], device=a.device)
            p, sc = T["p"], T["score"]
            arms = {}
            arms["full"] = full_scores(mt, x, sc)
            arms["source_alone"] = full_scores(ms, x, sc)
            own_t, cache_t = ssm_states(mt, x[:, :p], range(Lt))
            own_s, _ = ssm_states(ms, x[:, :p], range(Ls))
            own_t = [own_t[i] for i in range(Lt)]
            if selfcheck is None:
                _, _, lg = run_window(mt, x, p, own_t, sc, keep_conv_from=cache_t)
                ref = arms["full"][2]
                selfcheck = {"max_abs_logit_diff": float((lg - ref).abs().max()),
                             "logit_scale": float(ref.abs().max()),
                             "argmax_agreement": float((lg.argmax(-1) == ref.argmax(-1)).float().mean())}
                selfcheck["pass"] = bool(selfcheck["argmax_agreement"] > 0.98 and
                                         selfcheck["max_abs_logit_diff"] < 0.02 * selfcheck["logit_scale"] + 0.05)
                log("self-check (own state + true conv vs full prefill):", selfcheck)
            pr = torch.from_numpy(np.asarray(pred[[trow[(task, r)] for r in rows]])).to(a.device)  # fp16
            br = [pr[:, j * dims["t"]:(j + 1) * dims["t"]] for j in range(Lt)]
            roll = lambda L_: [torch.roll(s, -1, 0) for s in L_]  # noqa: E731
            arms["oracle"] = run_window(mt, x, p, own_t, sc)
            arms["blank"] = run_window(mt, x, p, None, sc)
            arms["mismatched_own"] = run_window(mt, x, p, roll(own_t), sc)
            arms["bridge"] = run_window(mt, x, p, br, sc)
            arms["bridge_mismatched"] = run_window(mt, x, p, roll(br), sc)
            arms["source_oracle"] = run_window(ms, x, p, [own_s[i] for i in range(Ls)], sc)
            del own_t, own_s, cache_t, pr, br
            for k, v in arms.items():
                res[task].setdefault(k, {"ce": [], "acc": []})
                res[task][k]["ce"] += v[0].tolist()
                res[task][k]["acc"] += v[1].tolist()
        log(f"{task}: evaluated {len(T['test'])} test prompts ({time.time() - t0:.0f}s)")
    # summary
    summ = {}
    for task in ("kv", "nat"):
        R = {k: {m: np.array(v) for m, v in d.items()} for k, d in res[task].items()}
        s = {"means": {k: {m: float(v.mean()) for m, v in d.items()} for k, d in R.items()}, "diffs_ce": {}, "diffs_acc": {}}
        for x_, y_ in [("oracle", "blank"), ("oracle", "mismatched_own"), ("bridge", "blank"), ("bridge", "bridge_mismatched"),
                       ("bridge", "source_alone"), ("bridge", "oracle"), ("oracle", "full"), ("source_oracle", "source_alone")]:
            s["diffs_ce"][f"{x_}_minus_{y_}"] = boot(R[x_]["ce"] - R[y_]["ce"])
            s["diffs_acc"][f"{x_}_minus_{y_}"] = boot(R[x_]["acc"] - R[y_]["acc"])
        d = s["diffs_ce"]
        s["gate_pass"] = bool(d["oracle_minus_blank"][0] <= -0.5 and d["oracle_minus_blank"][2] < 0 and
                              d["oracle_minus_mismatched_own"][0] <= -0.5 and d["oracle_minus_mismatched_own"][2] < 0)
        s["bridge_beats_all_three"] = bool(d["bridge_minus_blank"][2] < 0 and d["bridge_minus_bridge_mismatched"][2] < 0
                                           and d["bridge_minus_source_alone"][2] < 0)
        s["doc_specific_gap_ce"] = d["bridge_minus_bridge_mismatched"]
        m = s["means"]
        den = m["oracle"]["acc"] - m["blank"]["acc"]
        s["oracle_share_acc_blank_corrected"] = None if den == 0 else (m["bridge"]["acc"] - m["blank"]["acc"]) / den
        summ[task] = s
    kv = summ["kv"]
    if not selfcheck["pass"]:
        reading = "INVALID: the cache self-check failed (transformers cache API differs from this script's assumption)"
    elif not kv["gate_pass"]:
        reading = "gate failed: the target's own SSM state carries little recall here; no SSM-state bridge can work at this setting"
    elif kv["bridge_beats_all_three"]:
        reading = "bridge beats blank, a different prompt's state, and the small model alone: same-family states map across"
    elif kv["doc_specific_gap_ce"][2] < 0:
        reading = "document-specific gap present (bridge beats its mismatched control) but loses to the small model alone"
    else:
        reading = "no document-specific gap: the ridge-mapped state carries no prompt-specific content (as in the toy)"
    out = {"pair": name, "models": ids, "device": a.device, "dtype": a.dtype, "config": vars(a), "reading": reading,
           "self_check": selfcheck, "summary": summ, "layers": {"small": Ls, "big": Lt, "pairing": pair, "state_dims": dims},
           "ridge": {str(j): v for j, v in info.items()}, "prompts": P["meta"], "recall_by_gap_full_prefill": gap_diag,
           "n": {"cal_samples": n_cal, "test_kv": len(P["kv"]["test"]), "test_nat": len(P["nat"]["test"])},
           "per_prompt": res, "runtime_s": round(time.time() - t0, 1),
           "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "cpu",
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if a.device.startswith("cuda") else None,
           "versions": {"torch": torch.__version__, "transformers": __import__("transformers").__version__}}
    json.dump(out, open(os.path.join(a.out_dir, "results.json"), "w"), indent=1)
    try:  # close the memmap first: Windows cannot delete an open mapped file
        pred._mmap.close()
        del pred
        os.remove(os.path.join(a.out_dir, "pred_states.npy"))
    except Exception as e:
        log(f"could not remove the temporary pred_states.npy ({e!r}); delete it by hand")
    short = {t: {"means": {k: {m: round(v, 3) for m, v in d.items()} for k, d in summ[t]["means"].items()},
                 "gate_pass": summ[t]["gate_pass"], "bridge_beats_all_three": summ[t]["bridge_beats_all_three"],
                 "doc_gap_ce": [round(x, 3) for x in summ[t]["doc_specific_gap_ce"]],
                 "oracle_share": summ[t]["oracle_share_acc_blank_corrected"]} for t in summ}
    log("RESULT_SUMMARY " + json.dumps({"pair": name, "reading": reading, "self_check_pass": selfcheck["pass"],
                                        "runtime_min": round((time.time() - t0) / 60, 1), "tasks": short}))
    log("wrote", os.path.join(a.out_dir, "results.json"))


if __name__ == "__main__":
    main()
