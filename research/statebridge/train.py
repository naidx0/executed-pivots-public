"""Train one toy hybrid byte LM for a fixed wall-clock budget.
usage: python3 train.py {src|tgt} SEED MINUTES THREADS

Round 1: plain stdlib byte LM from scratch (defaults below).
Round 2 (env options): warm start from a checkpoint and train on a mix of stdlib windows and the
associative-recall task in recall.py.
  SB_INIT=ckpt_src.pt   warm start          SB_TAG=_r2       checkpoint/json suffix
  SB_T=1024             sequence length     SB_BS=8          batch size
  SB_MIX=0.5            fraction of rows that are recall-task rows
  SB_SEGP=0.75          fraction of rows trained with segment-local attention (segments of
                        SB_SEGMIN..SB_SEGMAX bytes); SSM state still carries across segments.
                        This mimics the StateBridge handoff (empty KV, carried SSM state).
  SB_VALW=1.0           extra loss term: weight x mean CE on recalled value bytes
  SB_MAXGAP=800         max code gap between bindings and queries in training episodes
  SB_LOGGAP=1           gap log-uniform in [4, SB_MAXGAP] instead of uniform
  SB_LR=1.5e-3  SB_WARM=20
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import train_stream
from model import CONFIGS, HybridLM

name, seed, minutes, threads = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
env = os.environ.get
INIT, TAG = env("SB_INIT"), env("SB_TAG", "")
BS, T = int(env("SB_BS", "8")), int(env("SB_T", "2048"))
MIX, SEGP, VALW = float(env("SB_MIX", "0")), float(env("SB_SEGP", "0")), float(env("SB_VALW", "0"))
SEGMIN, SEGMAX, MAXGAP = int(env("SB_SEGMIN", "32")), int(env("SB_SEGMAX", "256")), int(env("SB_MAXGAP", "800"))
LR, WARM = float(env("SB_LR", "2e-3")), int(env("SB_WARM", "100"))
torch.set_num_threads(threads)
torch.manual_seed(seed)
rng = np.random.default_rng(seed + 1000 * bool(INIT))

if MIX > 0:
    from recall import streams, train_row
    rstream = streams()["train"]
data = torch.from_numpy(train_stream().astype(np.int64))
model = HybridLM(**CONFIGS[name])
if INIT:
    model.load_state_dict(torch.load(INIT)["state"])
nparam = sum(p.numel() for p in model.parameters())
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=LR, betas=(0.9, 0.95))


def batch():
    toks, masks, segs = [], [], []
    for _ in range(BS):
        if rng.random() < MIX:
            r, m = train_row(rng, rstream, T + 1, MAXGAP, env("SB_LOGGAP") == "1")
            toks.append(torch.from_numpy(r.astype(np.int64)))
            masks.append(torch.from_numpy(m[1:].astype(bool)))
        else:
            i = int(rng.integers(0, len(data) - T - 1))
            toks.append(data[i:i + T + 1])
            masks.append(torch.zeros(T, dtype=torch.bool))
        if rng.random() < SEGP:
            b = np.cumsum(rng.integers(SEGMIN, SEGMAX + 1, T // SEGMIN + 1))
            segs.append(torch.from_numpy(np.searchsorted(b, np.arange(T), side="right")))
        else:
            segs.append(torch.zeros(T, dtype=torch.long))
    use_seg = SEGP > 0
    return torch.stack(toks), torch.stack(masks), (torch.stack(segs) if use_seg else None)


def step(tok, vmask, seg):
    lo, _, _ = model(tok[:, :-1], seg=seg)
    ce = F.cross_entropy(lo.reshape(-1, 256), tok[:, 1:].reshape(-1), reduction="none")
    loss = ce.mean()
    vloss = ce[vmask.reshape(-1)].mean() if vmask.any() else None
    tot = loss + VALW * vloss if (VALW > 0 and vloss is not None) else loss
    tot.backward()
    return loss, vloss


budget = minutes * 60
t0 = time.time()
it, log, ema, vema = 0, [], None, None
while True:
    el = time.time() - t0
    if el > budget:
        break
    frac = el / budget
    lr = LR * min(1.0, (it + 1) / WARM) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
    for g in opt.param_groups:
        g["lr"] = lr
    tok, vmask, seg = batch()
    opt.zero_grad(set_to_none=True)
    loss, vloss = step(tok, vmask, seg)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
    if vloss is not None:
        vema = vloss.item() if vema is None else 0.95 * vema + 0.05 * vloss.item()
    if it % 25 == 0:
        log.append((it, round(el, 1), round(ema, 4), None if vema is None else round(vema, 4)))
        print(name, it, f"{el:.0f}s", f"loss {ema:.4f}", f"value-byte loss {vema}", f"lr {lr:.2e}", flush=True)
    it += 1

torch.save({"cfg": CONFIGS[name], "state": model.state_dict()}, f"ckpt_{name}{TAG}.pt")
opts = {k: v for k, v in os.environ.items() if k.startswith("SB_")}
json.dump({"name": name + TAG, "seed": seed, "params": nparam, "steps": it, "tokens": it * BS * T,
           "minutes": minutes, "threads": threads, "final_train_loss_ema": ema,
           "final_value_byte_loss_ema": vema, "options": opts, "log": log},
          open(f"train_{name}{TAG}.json", "w"))
print("done", name, it, "steps", it * BS * T, "tokens", "params", nparam)
