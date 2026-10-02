"""Stand-in for the parts of transformers that run_real_pair.py uses (code test only).
Mimics Mamba2: out.cache_params.ssm_states / conv_states stacked tensors (L, B, ...); prefill when
cache_position is None or cache_position[0] == 0; one-token recurrent decode from the cache otherwise."""
import itertools
import re
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "stand-in-0.1"
_SYL = ["ba", "ce", "di", "fo", "gu", "ha", "ke", "li", "mo", "nu", "pa", "re", "si", "to", "vu", "wa", "ye", "zo"]
_WORDS = ["".join(p) for p in itertools.product(_SYL, repeat=2)] + ["".join(p) for p in itertools.product(_SYL[:8], repeat=3)]


class _Tok:
    def __init__(self):
        self.vocab = {}
        for c in [chr(i) for i in range(32, 127)] + ["\n"]:
            self.vocab["Ġ" if c == " " else ("Ċ" if c == "\n" else c)] = len(self.vocab)
        for w in _WORDS + ["these", "codes", "the", "and", "def", "return", "self"]:
            self.vocab.setdefault("Ġ" + w, len(self.vocab))

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_string(self, toks):
        return "".join(t.replace("Ġ", " ").replace("Ċ", "\n") for t in toks)

    def encode(self, text, add_special_tokens=True):
        out = []
        for m in re.finditer(r" [a-z]+|.|\n", text, re.S):
            s = m.group()
            if len(s) > 1 and "Ġ" + s[1:] in self.vocab:
                out.append(self.vocab["Ġ" + s[1:]])
                continue
            for ch in s:
                out.append(self.vocab.get("Ġ" if ch == " " else ("Ċ" if ch == "\n" else ch), self.vocab["?"]))
        return out


class AutoTokenizer:
    @staticmethod
    def from_pretrained(name):
        return _Tok()


class _Cache:
    def __init__(self, L, B, H, P, N, C, K, dtype):
        self.ssm_states = torch.zeros(L, B, H, P, N, dtype=dtype)
        self.conv_states = torch.zeros(L, B, C, K, dtype=dtype)


class _Layer(nn.Module):
    def __init__(self, d, H, P, N, K):
        super().__init__()
        self.H, self.P, self.N = H, P, N
        self.inp = nn.Linear(d, H * P + 2 * N + H)
        self.w = nn.Parameter(torch.randn(H * P, K) * 0.5)
        self.out = nn.Linear(H * P, d)

    def step(self, u, h, conv):
        B = u.shape[0]
        x, Bm, Cm, dt = torch.split(self.inp(u), [self.H * self.P, self.N, self.N, self.H], -1)
        conv = torch.cat([conv[:, :, 1:], x[:, :, None]], -1)
        xc = F.silu((conv * self.w).sum(-1))
        a = torch.exp(-0.02 * F.softplus(dt))
        h = a[:, :, None, None] * h + xc.view(B, self.H, self.P)[..., None] * Bm[:, None, None, :]
        y = (h * Cm[:, None, None, :]).sum(-1).reshape(B, -1)
        return self.out(y), h, conv


class _Model(nn.Module):
    def __init__(self, V, d, L, H, P, N, K=4):
        super().__init__()
        self.dims = (L, H, P, N, H * P, K)
        self.emb = nn.Embedding(V, d)
        self.layers = nn.ModuleList([_Layer(d, H, P, N, K) for _ in range(L)])
        self.head = nn.Linear(d, V)
        self.config = SimpleNamespace(num_hidden_layers=L)

    def forward(self, input_ids, cache_params=None, use_cache=None, cache_position=None, **kw):
        B, T = input_ids.shape
        L, H, P, N, C, K = self.dims
        decode = cache_params is not None and cache_position is not None and int(cache_position[0]) > 0
        if decode:
            assert T == 1, "decode path takes one token"
            cache = cache_params
        else:
            cache = _Cache(L, B, H, P, N, C, K, self.emb.weight.dtype)
        hs = [cache.ssm_states[i].clone() for i in range(L)]
        cs = [cache.conv_states[i].clone() for i in range(L)]
        x = self.emb(input_ids)
        logits = []
        for t in range(T):
            u = x[:, t]
            for i, lyr in enumerate(self.layers):
                o, hs[i], cs[i] = lyr.step(F.layer_norm(u, u.shape[-1:]), hs[i], cs[i])
                u = u + o
            logits.append(self.head(u))
        for i in range(L):
            cache.ssm_states[i].copy_(hs[i])
            cache.conv_states[i].copy_(cs[i])
        keep = use_cache or cache_params is not None
        return SimpleNamespace(logits=torch.stack(logits, 1), cache_params=cache if keep else None)


class AutoModelForCausalLM:
    @staticmethod
    def from_pretrained(name, dtype=None, torch_dtype=None):
        import os
        if os.environ.get("FAKE_NO_MAMBA2") and name.startswith("AntonV/"):
            raise OSError(f"{name} is not a valid model identifier (stand-in)")
        torch.manual_seed(len(name))
        V = len(_Tok().vocab)
        m = _Model(V, 32, 3, 4, 8, 4) if "130m" in name else _Model(V, 48, 5, 6, 8, 4)
        return m.to(dtype or torch_dtype or torch.float32)
