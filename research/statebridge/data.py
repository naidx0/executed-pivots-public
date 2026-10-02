"""Offline byte-level corpus: CPython 3.11 standard library .py files.

Deterministic split by sha1 of the relative path (files never straddle splits):
  test   : bucket < 12  and size >= MIN_TEST  (held out from training and calibration)
  calib  : 12 <= bucket < 32 and size >= MIN_CALIB (held out from training)
  train  : everything else
Only /usr/lib/python3.11 is used (other python versions on disk contain near-copies of the
same files and would leak test text into training).
"""
import hashlib
import os

import numpy as np

ROOT = "/usr/lib/python3.11"
MIN_TEST = 8192 + 512
MIN_CALIB = 2048


def _files():
    out = []
    for d, _, files in os.walk(ROOT):
        if "site-packages" in d or "dist-packages" in d or "__pycache__" in d:
            continue
        for f in sorted(files):
            if f.endswith(".py"):
                p = os.path.join(d, f)
                out.append((os.path.relpath(p, ROOT), p))
    return sorted(out)


def _bucket(rel):
    return int(hashlib.sha1(rel.encode()).hexdigest(), 16) % 100


def split():
    train, calib, test = [], [], []
    seen = {}
    for rel, p in _files():
        b = open(p, "rb").read()
        if len(b) == 0:
            continue
        h = hashlib.sha1(b).hexdigest()
        if h in seen:  # drop exact duplicates entirely
            continue
        seen[h] = rel
        k = _bucket(rel)
        if k < 12 and len(b) >= MIN_TEST:
            test.append((rel, b))
        elif 12 <= k < 32 and len(b) >= MIN_CALIB:
            calib.append((rel, b))
        else:
            train.append((rel, b))
    return train, calib, test


def train_stream():
    train, _, _ = split()
    # files separated by a NUL byte
    buf = b"\x00".join(b for _, b in train)
    return np.frombuffer(buf, dtype=np.uint8).copy()


if __name__ == "__main__":
    tr, ca, te = split()
    for name, s in [("train", tr), ("calib", ca), ("test", te)]:
        print(name, len(s), sum(len(b) for _, b in s))
