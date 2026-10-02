"""Round 2 synthetic long-range task: associative recall inside code-like text.

An episode is a block of bindings  `cfg_abc = "q7rt"\\n`  (unique 3-letter names, 4-char values
from [a-z0-9]), then `gap` bytes of real stdlib code from the requested split, then queries
`assert cfg_abc == "q7rt"\\n` in random order. Only the 4 value bytes of each query are "recalled
value bytes"; they can be predicted only from the binding block, `gap` bytes back.
"""
import numpy as np

from data import split

LET = b"abcdefghijklmnopqrstuvwxyz"
ALPH = b"abcdefghijklmnopqrstuvwxyz0123456789"
VLEN = 4
BIND_LEN = 17   # len(b'cfg_abc = "wxyz"\n')
QUERY_LEN = 25  # len(b'assert cfg_abc == "wxyz"\n')
VAL_OFF = 19    # offset of the first value byte inside a query line


def streams():
    tr, ca, te = split()
    mk = lambda s: np.frombuffer(b"\n".join(b for _, b in s).replace(b"\x00", b"\n"), dtype=np.uint8).copy()
    return {"train": mk(tr), "calib": mk(ca), "test": mk(te)}


def _filler(rng, stream, n):
    if n <= 0:
        return b""
    i = int(rng.integers(0, len(stream) - n))
    return stream[i:i + n - 1].tobytes() + b"\n"


def episode(rng, stream, nbind, gap, nq=None):
    names = []
    while len(names) < nbind:
        n = b"cfg_" + bytes(LET[k] for k in rng.integers(0, 26, 3))
        if n not in names:
            names.append(n)
    vals = [bytes(ALPH[k] for k in rng.integers(0, 36, VLEN)) for _ in range(nbind)]
    bind = b"".join(n + b' = "' + v + b'"\n' for n, v in zip(names, vals))
    fill = _filler(rng, stream, gap)
    q, mask = [], []
    for i in rng.permutation(nbind)[: nq or nbind]:
        pre = b"assert " + names[i] + b' == "'
        q += [pre, vals[i], b'"\n']
        mask += [0] * len(pre) + [1] * VLEN + [0, 0]
    body = bind + fill
    doc = body + b"".join(q)
    m = np.zeros(len(doc), dtype=np.uint8)
    m[len(body):] = mask
    return doc, m


def train_row(rng, stream, n, maxgap, loggap=False):
    """Pack episodes (random #bindings 1..8, gap 0..maxgap, short code lead-ins) into n bytes.
    loggap: gap log-uniform in [4, maxgap] instead of uniform (more short-range episodes)."""
    parts, masks, tot = [], [], 0
    while tot < n:
        lead = _filler(rng, stream, int(rng.integers(0, 128)))
        gap = int(np.exp(rng.uniform(np.log(4), np.log(maxgap)))) if loggap else int(rng.integers(0, maxgap))
        d, m = episode(rng, stream, int(rng.integers(1, 9)), gap)
        parts += [lead, d]
        masks += [np.zeros(len(lead), np.uint8), m]
        tot += len(lead) + len(d)
    return np.frombuffer(b"".join(parts), np.uint8)[:n].copy(), np.concatenate(masks)[:n]


def eval_docs(stream, n, T, seed, nbind=6):
    """n docs of exactly T + nbind*QUERY_LEN bytes: bindings at [0, nbind*BIND_LEN), code filler up
    to T, queries from T on. Returns list of bytes, and value-byte positions relative to T."""
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n):
        d, m = episode(rng, stream, nbind, T - nbind * BIND_LEN)
        assert len(d) == T + nbind * QUERY_LEN and m[:T].sum() == 0
        docs.append(d)
    pos = np.array([q * QUERY_LEN + VAL_OFF + k for q in range(nbind) for k in range(VLEN)])
    assert m[T + pos].all() and m.sum() == len(pos)
    return docs, pos


if __name__ == "__main__":
    s = streams()
    rng = np.random.default_rng(0)
    r, m = train_row(rng, s["train"], 600, 300)
    print(r.tobytes().decode("latin1")[:600])
    print(m.sum())
    d, pos = eval_docs(s["test"], 2, 864, 0)
    print(d[0][:120], d[0][864:])
