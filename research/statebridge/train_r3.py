"""StateBridge round 3: recall-curriculum training of both toy models (training change only).

usage: python3 train_r3.py [--minutes 150] [--ckpt-min 15] [--tag _r3] [--init _r2]

One controller process spawns two workers (src: 1 thread, tgt: 3 threads) that train in
parallel, each warm-started from ckpt_{name}{init}.pt. Every --ckpt-min minutes the controller
pauses both workers, they save ckpt_{name}{tag}.pt (+ a copy ckpt_{name}{tag}_g{k}.pt), and the
controller runs gate_r3.py k (the registered gate from statebridge2.py on the target, plus the same
procedure on the source; calibration docs only) with all 4 cores, then resumes them.
Stops when both models pass, when the wall-clock budget is used up, or when the stop file
STOP{tag} appears in this directory (or on SIGTERM). Status: round3_status.json (tag _r3).

Training rows (1024 bytes + 1, batch 8):
- LM_FRAC of rows: plain stdlib windows (train split); half of them with random segment-local
  attention (segments 32-256 B).
- the rest: packed recall episodes (bindings `cfg_abc = "q7rt"`, stdlib code gap, queries
  `assert cfg_abc == "q7rt"`), the first one at byte 0. For 90% of recall rows each episode gets a
  "handoff": an attention-segment boundary d bytes before its first query, d in [0, gap] (half the
  time d = min(gap, 32), the gate's window). Attention after the boundary cannot see the bindings,
  so the value bytes can only come through the SSM state, exactly as in the gate's oracle arm.
  10% of recall rows use ordinary full attention.
- loss = mean CE over all bytes + VALW * mean CE over recalled value bytes.

Curriculum (per model, independently): STAGES below. A recall row uses the current stage with
p = 0.75, otherwise a uniformly chosen earlier stage (rehearsal: its bindings/gap ranges, but the
current stage's value alphabet, so the small early alphabet is not rehearsed once left). Every PROBE_EVERY steps the model
is scored on a fixed probe of the current stage (training split, handoff at min(gap, 32)); when
value-byte argmax accuracy >= the stage threshold it moves to the next stage. The last stage is the
registered test distribution (6 bindings, 762-byte code gap, values >= 762 bytes back) mixed 50/50
with 4-6 bindings and 64-800-byte gaps; its probe is exactly 6 bindings, gap 762, handoff 32.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

T = 1024
BS = 8
LM_FRAC = 0.2
FULLATTN_FRAC = 0.1
REHEARSAL = 0.25
VALW = 4.0
LR = {"src": 2e-3, "tgt": 1.5e-3}
WARM = 30
DECAY_FROM = 0.75  # fraction of the budget after which lr decays (cosine) to 0.2x
THREADS = {"src": 1, "tgt": 3}
SEEDS = {"src": 30, "tgt": 31}
PROBE_EVERY = 50
PROBE_ROWS = 8
LM_PROBE_EVERY = 250
LEAD_MAX = 32
# nb: bindings range, gap: code-gap range (bytes), alph: value alphabet size (prefix of recall.ALPH),
# thr: probe value-byte accuracy needed to advance (None = final stage)
STAGES = [
    dict(nb=(1, 1), gap=(0, 16), alph=8, thr=0.8),
    dict(nb=(1, 2), gap=(0, 32), alph=8, thr=0.8),
    dict(nb=(1, 2), gap=(0, 32), alph=36, thr=0.75),
    dict(nb=(1, 3), gap=(0, 64), alph=36, thr=0.7),
    dict(nb=(2, 4), gap=(0, 128), alph=36, thr=0.65),
    dict(nb=(2, 6), gap=(16, 256), alph=36, thr=0.6),
    dict(nb=(3, 6), gap=(32, 512), alph=36, thr=0.6),
    dict(nb=(4, 6), gap=(64, 800), alph=36, thr=None),
]
TEST_NB, TEST_GAP, GATE_W = 6, 762, 32


# ------------------------------------------------------------------ data
def episode(rng, stream, nbind, gap, alph):
    from recall import ALPH, LET, VLEN, _filler
    names = []
    while len(names) < nbind:
        n = b"cfg_" + bytes(LET[k] for k in rng.integers(0, 26, 3))
        if n not in names:
            names.append(n)
    vals = [bytes(ALPH[k] for k in rng.integers(0, alph, VLEN)) for _ in range(nbind)]
    bind = b"".join(n + b' = "' + v + b'"\n' for n, v in zip(names, vals))
    fill = _filler(rng, stream, gap)
    q, mask = [], []
    for i in rng.permutation(nbind):
        pre = b"assert " + names[i] + b' == "'
        q += [pre, vals[i], b'"\n']
        mask += [0] * len(pre) + [1] * VLEN + [0, 0]
    body = bind + fill
    doc = body + b"".join(q)
    m = np.zeros(len(doc), dtype=np.uint8)
    m[len(body):] = mask
    return doc, m, len(body)


def sample_nb_gap(rng, st, final, probe):
    if final and probe:
        return TEST_NB, TEST_GAP
    if final and rng.random() < 0.5:
        return TEST_NB, int(rng.integers(700, 801))
    return int(rng.integers(st["nb"][0], st["nb"][1] + 1)), int(rng.integers(st["gap"][0], st["gap"][1] + 1))


def recall_row(rng, stream, si, handoff=True, probe=False, alph=None):
    """One row of n = T+1 bytes of packed episodes of stage si (value alphabet overridden by alph if
    given). Returns bytes, value mask, and attention-segment ids for the T input positions."""
    from recall import _filler
    n = T + 1
    st, final = STAGES[si], si == len(STAGES) - 1
    parts, masks, bounds, pos = [], [], [], 0
    while True:
        lead = 0 if pos == 0 else int(rng.integers(0, LEAD_MAX + 1))
        nb, gap = sample_nb_gap(rng, st, final, probe)
        d, m, qs = episode(rng, stream, nb, gap, alph or st["alph"])
        if pos + lead + len(d) > n:
            break
        if lead:
            parts.append(_filler(rng, stream, lead))
            masks.append(np.zeros(lead, np.uint8))
        if handoff:
            dd = min(gap, GATE_W) if (probe or rng.random() < 0.5) else int(rng.integers(0, gap + 1))
            bounds.append(pos + lead + qs - dd)
        parts.append(d)
        masks.append(m)
        pos += lead + len(d)
        if probe and final:
            break
    if pos < n:
        parts.append(_filler(rng, stream, n - pos))
        masks.append(np.zeros(n - pos, np.uint8))
    tok = np.frombuffer(b"".join(parts), np.uint8)
    msk = np.concatenate(masks)
    assert len(tok) == n and len(msk) == n
    seg = np.searchsorted(np.array(bounds, dtype=np.int64), np.arange(T), side="right")
    return tok, msk, seg


# ------------------------------------------------------------------ worker
def worker(name, tag, init, t0, budget_s, sh):
    import torch
    import torch.nn.functional as F
    from data import train_stream
    from model import CONFIGS, HybridLM
    from recall import streams

    torch.set_num_threads(THREADS[name])
    torch.manual_seed(SEEDS[name])
    rng = np.random.default_rng(SEEDS[name])
    os.chdir(HERE)
    S = streams()
    rstream, cstream = S["train"], S["calib"]
    lm = train_stream()
    model = HybridLM(**CONFIGS[name])
    model.load_state_dict(torch.load(f"ckpt_{name}{init}.pt")["state"])
    nparam = sum(p.numel() for p in model.parameters())
    decay = [p for _, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for _, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=LR[name], betas=(0.9, 0.95))
    done_v = sh[f"done_{name}"]

    def to_t(rows):
        tok = torch.from_numpy(np.stack([r[0] for r in rows]).astype(np.int64))
        msk = torch.from_numpy(np.stack([r[1][1:] for r in rows]).astype(bool))
        seg = torch.from_numpy(np.stack([r[2] for r in rows]).astype(np.int64))
        return tok, msk, seg

    def lm_row():
        i = int(rng.integers(0, len(lm) - T - 1))
        tok = lm[i:i + T + 1]
        if rng.random() < 0.5:
            b = np.cumsum(rng.integers(32, 257, T // 32 + 1))
            seg = np.searchsorted(b, np.arange(T), side="right")
        else:
            seg = np.zeros(T, np.int64)
        return tok, np.zeros(T + 1, np.uint8), seg

    def probe_set(si):
        prng = np.random.default_rng(10_000 + si)
        return to_t([recall_row(prng, rstream, si, handoff=True, probe=True) for _ in range(PROBE_ROWS)])

    lmp = np.random.default_rng(99)
    lm_probe = torch.from_numpy(np.stack([cstream[i:i + T + 1] for i in lmp.integers(0, len(cstream) - T - 1, 8)]).astype(np.int64))

    @torch.no_grad()
    def run_probe(pb):
        tok, msk, seg = pb
        lo, _, _ = model(tok[:, :-1], seg=seg)
        tg = tok[:, 1:]
        ce = F.cross_entropy(lo.reshape(-1, 256), tg.reshape(-1), reduction="none").view(tg.shape)
        acc = (lo.argmax(-1) == tg)[msk].float().mean().item()
        return acc, ce[msk].mean().item()

    @torch.no_grad()
    def run_lm_probe():
        lo, _, _ = model(lm_probe[:, :-1])
        return F.cross_entropy(lo.reshape(-1, 256), lm_probe[:, 1:].reshape(-1)).item()

    stage, pb = 0, probe_set(0)
    it, ema, vema, lmema = 0, None, None, None
    log, hist = [], []
    lm0 = run_lm_probe()
    a0, c0 = run_probe(pb)
    hist.append({"step": 0, "t": round(time.time() - t0, 1), "stage": 0, "probe_acc": round(a0, 4),
                 "probe_ce": round(c0, 4), "event": "start", "lm_probe_calib": round(lm0, 4)})
    print(f"{name} start: stage 0 probe acc {a0:.3f} ce {c0:.3f}; lm probe (calib) {lm0:.4f}; params {nparam}", flush=True)

    def save(k):
        torch.save({"cfg": CONFIGS[name], "state": model.state_dict()}, f"ckpt_{name}{tag}.pt")
        shutil.copy(f"ckpt_{name}{tag}.pt", f"ckpt_{name}{tag}_g{k}.pt")
        info = {"name": name + tag, "seed": SEEDS[name], "params": nparam, "steps": it, "tokens": it * BS * T,
                "minutes": round((time.time() - t0) / 60, 2), "threads": THREADS[name],
                "final_train_loss_ema": ema, "final_value_byte_loss_ema": vema, "final_lm_row_loss_ema": lmema,
                "options": {"script": "train_r3.py", "init": f"ckpt_{name}{init}.pt", "T": T, "BS": BS, "LM_FRAC": LM_FRAC,
                            "FULLATTN_FRAC": FULLATTN_FRAC, "REHEARSAL": REHEARSAL, "VALW": VALW, "LR": LR[name],
                            "WARM": WARM, "DECAY_FROM": DECAY_FROM, "budget_min": budget_s / 60, "STAGES": STAGES,
                            "PROBE_EVERY": PROBE_EVERY, "PROBE_ROWS": PROBE_ROWS},
                "curriculum": {"stage": stage, "n_stages": len(STAGES), "history": hist},
                "log": log}
        json.dump(info, open(f"train_{name}{tag}.json", "w"))

    while True:
        req = sh["req"].value
        if req > done_v.value:
            save(req)
            done_v.value = req
            print(f"{name} saved checkpoint g{req} at step {it} (stage {stage})", flush=True)
            while sh["resume"].value < req and not sh["quit"].value:
                time.sleep(0.5)
        if sh["quit"].value:
            return
        el = time.time() - t0
        frac = min(1.0, el / budget_s)
        sched = 1.0 if frac < DECAY_FROM else 0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * (frac - DECAY_FROM) / (1 - DECAY_FROM)))
        lr = LR[name] * min(1.0, (it + 1) / WARM) * sched
        for g in opt.param_groups:
            g["lr"] = lr
        rows, is_lm = [], []
        for _ in range(BS):
            if rng.random() < LM_FRAC:
                rows.append(lm_row())
                is_lm.append(True)
            else:
                si = stage if (stage == 0 or rng.random() >= REHEARSAL) else int(rng.integers(0, stage))
                rows.append(recall_row(rng, rstream, si, handoff=rng.random() >= FULLATTN_FRAC, alph=STAGES[stage]["alph"]))
                is_lm.append(False)
        tok, msk, seg = to_t(rows)
        opt.zero_grad(set_to_none=True)
        lo, _, _ = model(tok[:, :-1], seg=seg)
        ce = F.cross_entropy(lo.reshape(-1, 256), tok[:, 1:].reshape(-1), reduction="none").view(BS, T)
        loss = ce.mean()
        vloss = ce[msk].mean() if msk.any() else None
        tot = loss + VALW * vloss if vloss is not None else loss
        tot.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if vloss is not None:
            vema = vloss.item() if vema is None else 0.95 * vema + 0.05 * vloss.item()
        lmi = torch.tensor(is_lm)
        if lmi.any():
            l = ce[lmi].mean().item()
            lmema = l if lmema is None else 0.95 * lmema + 0.05 * l
        it += 1
        if it % 25 == 0:
            log.append((it, round(el, 1), stage, round(ema, 4), None if vema is None else round(vema, 4),
                        None if lmema is None else round(lmema, 4)))
            print(f"{name} step {it} {el:.0f}s stage {stage} loss {ema:.4f} value-byte loss {vema:.4f} "
                  f"lm-row loss {lmema if lmema is None else round(lmema, 4)} lr {lr:.2e}", flush=True)
        if it % PROBE_EVERY == 0:
            acc, pce = run_probe(pb)
            ev = {"step": it, "t": round(time.time() - t0, 1), "stage": stage, "probe_acc": round(acc, 4), "probe_ce": round(pce, 4)}
            if it % LM_PROBE_EVERY == 0:
                ev["lm_probe_calib"] = round(run_lm_probe(), 4)
            msg = f"{name} probe step {it} stage {stage}: value acc {acc:.3f} ce {pce:.3f}" + \
                  (f" lm probe (calib) {ev['lm_probe_calib']:.4f}" if "lm_probe_calib" in ev else "")
            thr = STAGES[stage]["thr"]
            if thr is not None and acc >= thr:
                stage += 1
                pb = probe_set(stage)
                ev["event"] = f"advance to stage {stage}"
                msg += f" -> ADVANCE to stage {stage} {STAGES[stage]}"
            hist.append(ev)
            print(msg, flush=True)


# ------------------------------------------------------------------ controller
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=150.0)
    ap.add_argument("--ckpt-min", type=float, default=15.0)
    ap.add_argument("--tag", default="_r3")
    ap.add_argument("--init", default="_r2")
    ap.add_argument("--require-recall", type=float, default=None,
                    help="also require full-prefill value CE below this (nats/byte) for both models before stopping")
    a = ap.parse_args()
    os.chdir(HERE)
    tag = a.tag
    status_path = os.path.join(HERE, "round3_status.json" if tag == "_r3" else f"round3_status{tag}.json")
    stop_path = os.path.join(HERE, f"STOP{tag}")
    gate_log = os.path.join(HERE, f"gate{tag}_log.jsonl")
    budget_s = a.minutes * 60
    t0 = time.time()
    ctx = mp.get_context("spawn")
    sh = {k: ctx.Value("i", 0) for k in ["req", "resume", "quit", "done_src", "done_tgt"]}
    procs = {n: ctx.Process(target=worker, args=(n, tag, a.init, t0, budget_s, sh), name=f"train_r3_{n}") for n in ["src", "tgt"]}
    for p in procs.values():
        p.start()
    sigterm = {"hit": False}
    signal.signal(signal.SIGTERM, lambda *_: sigterm.__setitem__("hit", True))
    out_dir = "/mnt/project-files/nebius/statebridge/round3"
    test_cmd = (f"cd {HERE} && mkdir -p {out_dir} && SB_SUFFIX={tag} SB_OUT={out_dir} SB_THREADS=4 "
                f"python3 statebridge2.py run")
    gates = []
    status = {"state": "running", "stop_reason": None, "tag": tag, "controller_pid": os.getpid(),
              "worker_pids": {n: p.pid for n, p in procs.items()},
              "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "budget_min": a.minutes,
              "checkpoint_every_min": a.ckpt_min, "stop_file": stop_path,
              "log": os.path.join(HERE, "train_r3.log" if tag == "_r3" else f"train{tag}.log"),
              "gate_log": gate_log, "gate_attempts_json": os.path.join(HERE, f"gate{tag}_attempts.json"),
              "latest_gate": None, "gate_history": [],
              "checkpoints": {n: os.path.join(HERE, f"ckpt_{n}{tag}.pt") for n in ["src", "tgt"]},
              "checkpoint_copies": {}, "curriculum_stage": {},
              "test_command": test_cmd,
              "test_command_note": ("Registered test evaluation (statebridge2.py run, protocol unchanged) on "
                                    f"ckpt_{{src,tgt}}{tag}.pt. Valid only when state == 'passed'. Writes "
                                    f"{out_dir}/round2-results.json (filename fixed by statebridge2.py; SB_OUT keeps "
                                    "it away from the round-2 file). Not run by the training job.")}

    def write_status():
        status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        status["elapsed_min"] = round((time.time() - t0) / 60, 1)
        for n in ["src", "tgt"]:
            try:
                c = json.load(open(f"train_{n}{tag}.json"))["curriculum"]
                status["curriculum_stage"][n] = {"stage": c["stage"], "of": c["n_stages"] - 1,
                                                 "last_probe": c["history"][-1]}
            except Exception:
                pass
        tmp = status_path + ".tmp"
        json.dump(status, open(tmp, "w"), indent=1)
        os.replace(tmp, status_path)

    def alive():
        return all(p.is_alive() for p in procs.values())

    def checkpoint_and_gate(k):
        sh["req"].value = k
        while not all(sh[f"done_{n}"].value >= k for n in procs):
            if not alive():
                raise RuntimeError("a worker died")
            time.sleep(0.5)
        print(f"controller: checkpoint g{k} saved at {(time.time()-t0)/60:.1f} min; running gate", flush=True)
        env = dict(os.environ, SB_SUFFIX=tag, SB_THREADS="4")
        r = subprocess.run([sys.executable, "gate_r3.py", str(k)], cwd=HERE, env=env, capture_output=True, text=True)
        print(r.stdout + r.stderr, flush=True)
        if r.returncode != 0:
            raise RuntimeError(f"gate_r3.py failed (exit {r.returncode})")
        line = json.loads(open(gate_log).read().strip().splitlines()[-1])
        assert line["attempt"] == k
        status["latest_gate"] = line
        status["gate_history"].append({"attempt": k, "elapsed_min": round((time.time() - t0) / 60, 1),
                                       "tgt_pass": line["tgt"]["gate_pass"], "src_pass": line["src"]["gate_pass"],
                                       "tgt_w32_diffs": [line["tgt"]["w32_diff_oracle_minus_blank"][0],
                                                         line["tgt"]["w32_diff_oracle_minus_mismatched_own"][0]],
                                       "src_w32_diffs": [line["src"]["w32_diff_oracle_minus_blank"][0],
                                                         line["src"]["w32_diff_oracle_minus_mismatched_own"][0]]})
        status["checkpoint_copies"][f"g{k}"] = {n: os.path.join(HERE, f"ckpt_{n}{tag}_g{k}.pt") for n in procs}
        if a.require_recall is not None:
            # stricter stop (added 12:15Z after gate g1 of the first full run): the registered gate can pass
            # on overconfident blank/mismatched arms while full prefill is still at chance, so also require
            # real recall with full prefill before stopping. Evaluation protocol unchanged.
            return line["both_pass"] and all(line[n]["full_prefill_value_ce"] < a.require_recall for n in ("src", "tgt"))
        return line["both_pass"]

    k = 0
    next_ck = t0 + a.ckpt_min * 60
    write_status()
    try:
        while True:
            time.sleep(2)
            now = time.time()
            if not alive():
                raise RuntimeError("a worker died")
            stop = os.path.exists(stop_path) or sigterm["hit"]
            over = now - t0 >= budget_s
            if stop or over or now >= next_ck:
                k += 1
                passed = checkpoint_and_gate(k)
                if passed:
                    status.update(state="passed", stop_reason="both models passed the gate")
                    break
                if stop or over:
                    status.update(state="failed", stop_reason="stopped by stop file / SIGTERM" if stop else "budget used up")
                    break
                write_status()
                sh["resume"].value = k
                next_ck = time.time() + a.ckpt_min * 60
    except Exception as e:  # noqa: BLE001
        status.update(state="failed", stop_reason=f"error: {e!r}")
    sh["quit"].value = 1
    sh["resume"].value = max(k, sh["resume"].value)
    for p in procs.values():
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_status()
    print(f"controller: done, state={status['state']} ({status['stop_reason']})", flush=True)


if __name__ == "__main__":
    main()
