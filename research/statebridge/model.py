"""Toy hybrid byte LM: Mamba-2-style selective SSM layers (diagonal, input-dependent scalar decay
per head, chunked SSD scan) plus a few NoPE causal attention(+MLP) layers. Plain PyTorch, CPU.

State per SSM layer: h (B, H, P, N) and the depthwise-conv tail (B, conv_dim, d_conv-1).
State per attention layer: (K, V) of everything it has seen.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def ssd(xdt, la, Bm, Cm, h0, L=64, snap=False):
    """Chunked selective scan.
    xdt: (B,T,H,P) input already scaled by dt; la: (B,T,H) log-decay (<=0);
    Bm, Cm: (B,T,N); h0: (B,H,P,N) or None.
    h_t = exp(la_t) h_{t-1} + xdt_t (outer) B_t ;  y_t = h_t C_t
    Returns y (B,T,H,P), final state, and (if snap) states after every L-th position.
    """
    Bsz, T, H, P = xdt.shape
    N = Bm.shape[-1]
    pad = (-T) % L
    if pad:  # zero input + zero log-decay leaves the state unchanged
        xdt = F.pad(xdt, (0, 0, 0, 0, 0, pad))
        la = F.pad(la, (0, 0, 0, pad))
        Bm = F.pad(Bm, (0, 0, 0, pad))
        Cm = F.pad(Cm, (0, 0, 0, pad))
    nc = (T + pad) // L
    # head-major layout so the heavy products are plain batched matmuls
    xc = xdt.view(Bsz, nc, L, H, P).permute(0, 1, 3, 2, 4)  # (B,nc,H,L,P)
    cs = torch.cumsum(la.view(Bsz, nc, L, H), dim=2).permute(0, 1, 3, 2)  # (B,nc,H,L)
    Bc = Bm.view(Bsz, nc, L, N)
    Cc = Cm.view(Bsz, nc, L, N)
    seg = cs[..., :, None] - cs[..., None, :]  # (B,nc,H,L,L)
    mask = torch.ones(L, L, dtype=torch.bool, device=xdt.device).tril()
    CB = (Cc @ Bc.transpose(-1, -2))[:, :, None]  # (B,nc,1,L,L)
    W = torch.exp(seg.masked_fill(~mask, float("-inf"))) * CB
    y = W @ xc  # (B,nc,H,L,P)
    dte = torch.exp(cs[..., -1:] - cs)  # (B,nc,H,L)
    S_local = (xc * dte[..., None]).transpose(-1, -2) @ Bc[:, :, None]  # (B,nc,H,P,N)
    cdec = torch.exp(cs[..., -1])  # (B,nc,H)
    h = h0 if h0 is not None else xdt.new_zeros(Bsz, H, P, N)
    hs_in = []
    for c in range(nc):
        hs_in.append(h)
        h = h * cdec[:, c, :, None, None] + S_local[:, c]
    h_in = torch.stack(hs_in, 1)  # state entering each chunk (B,nc,H,P,N)
    y = y + (Cc[:, :, None] @ h_in.transpose(-1, -2)) * torch.exp(cs)[..., None]
    y = y.permute(0, 1, 3, 2, 4).reshape(Bsz, nc * L, H, P)[:, :T]
    snaps = None
    if snap:  # states after positions L, 2L, ..., (only full chunks inside T)
        snaps = torch.cat([h_in[:, 1:], h[:, None]], 1)[:, : T // L]
    return y, h, snaps


class Mamba2(nn.Module):
    def __init__(self, d, expand=2, headdim=16, d_state=16, d_conv=4, chunk=64):
        super().__init__()
        self.di = expand * d
        self.P, self.N, self.K, self.L = headdim, d_state, d_conv, chunk
        self.H = self.di // headdim
        self.in_proj = nn.Linear(d, 2 * self.di + 2 * d_state + self.H, bias=False)
        self.cdim = self.di + 2 * d_state
        self.conv = nn.Conv1d(self.cdim, self.cdim, d_conv, groups=self.cdim, bias=True)
        dt = torch.exp(torch.rand(self.H) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(self.H).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(self.H))
        self.norm = RMSNorm(self.di)
        self.out_proj = nn.Linear(self.di, d, bias=False)

    def forward(self, u, state=None, snap=False):
        Bsz, T, _ = u.shape
        z, xBC, dt = torch.split(self.in_proj(u), [self.di, self.cdim, self.H], dim=-1)
        xBC = xBC.transpose(1, 2)
        if state is not None and state[1] is not None:
            prev = state[1]
        else:
            prev = xBC.new_zeros(Bsz, self.cdim, self.K - 1)
        xpad = torch.cat([prev, xBC], 2)
        conv_tail = xpad[:, :, -(self.K - 1):]
        # depthwise causal conv as shifted multiply-adds (much faster backward on CPU than conv1d)
        w = self.conv.weight[:, 0, :]  # (C, K)
        acc = self.conv.bias[None, :, None] + xpad[:, :, 0:T] * w[None, :, 0:1]
        for j in range(1, self.K):
            acc = acc + xpad[:, :, j:j + T] * w[None, :, j:j + 1]
        xBC = F.silu(acc).transpose(1, 2)
        x, Bm, Cm = torch.split(xBC, [self.di, self.N, self.N], dim=-1)
        x = x.reshape(Bsz, T, self.H, self.P)
        dt = F.softplus(dt + self.dt_bias)
        la = dt * (-torch.exp(self.A_log))
        h0 = state[0] if state is not None else None
        y, h, snaps = ssd(x * dt[..., None], la, Bm, Cm, h0, self.L, snap)
        y = y + x * self.D[:, None]
        y = self.norm(y.reshape(Bsz, T, self.di) * F.silu(z))
        return self.out_proj(y), (h, conv_tail), snaps


class Attn(nn.Module):
    def __init__(self, d, headdim=32):
        super().__init__()
        self.h = d // headdim
        self.hd = headdim
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, cache=None, seg=None):
        Bsz, T, d = x.shape
        q, k, v = self.qkv(x).view(Bsz, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        if cache is not None and cache[0] is not None:
            k = torch.cat([cache[0], k], 2)
            v = torch.cat([cache[1], v], 2)
        S = k.shape[2]
        if seg is not None and S == T:  # training only: attention confined to segments (B,T) ids
            m = torch.ones(T, T, dtype=torch.bool).tril()[None] & (seg[:, :, None] == seg[:, None, :])
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=m[:, None])
        elif S == T:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            past = S - T
            m = torch.arange(S)[None, :] <= (torch.arange(T)[:, None] + past)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return self.o(o.transpose(1, 2).reshape(Bsz, T, d)), (k, v)


class HybridLM(nn.Module):
    def __init__(self, d=128, pattern="MMAMM", headdim=16, d_state=16, attn_headdim=32, chunk=64):
        super().__init__()
        self.pattern = pattern
        self.emb = nn.Embedding(256, d)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.mlps = nn.ModuleDict()
        for i, c in enumerate(pattern):
            self.norms.append(RMSNorm(d))
            if c == "M":
                self.layers.append(Mamba2(d, headdim=headdim, d_state=d_state, chunk=chunk))
            else:
                self.layers.append(Attn(d, attn_headdim))
                self.mlps[str(i)] = nn.Sequential(RMSNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.fnorm = RMSNorm(d)
        self.head = nn.Linear(d, 256, bias=False)
        self.ssm_idx = [i for i, c in enumerate(pattern) if c == "M"]

    def forward(self, tok, states=None, snap=False, seg=None):
        """tok (B,T) long. states: list per layer (None = blank). Returns logits, new states,
        and (if snap) {layer_idx: (B, T//chunk, H, P, N)} SSM states after every chunk.
        seg (B,T) optional segment ids: attention only within a segment (SSM state still carries)."""
        x = self.emb(tok)
        new, snaps = [], {}
        for i, (c, lyr, nrm) in enumerate(zip(self.pattern, self.layers, self.norms)):
            st = states[i] if states is not None else None
            if c == "M":
                y, s, sn = lyr(nrm(x), st, snap)
                if snap:
                    snaps[i] = sn
            else:
                y, s = lyr(nrm(x), st, seg)
            x = x + y
            if c == "A":
                x = x + self.mlps[str(i)](x)
            new.append(s)
        return self.head(self.fnorm(x)), new, snaps


CONFIGS = {
    # "Nano": narrower, shallower
    "src": dict(d=128, pattern="MMAMM", headdim=32, d_state=16, attn_headdim=32, chunk=32),
    # "Super": wider, deeper, two attention layers
    "tgt": dict(d=192, pattern="MMMAMMMMAM", headdim=32, d_state=16, attn_headdim=32, chunk=32),
}
