mkdir -p /app/textstats/src/textstats /data
cat > /app/textstats/pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "textstats"
version = "1.2.0"
description = "Line, word and top-word statistics for plain-text files"
readme = "README.md"
requires-python = ">=3.12"
dependencies = []

[project.scripts]
textstats = "textstats.cli:main"

[tool.setuptools]
packages = ["textstats"]
EOF
cat > /app/textstats/README.md <<'EOF'
# textstats

Line, word and top-word statistics for plain-text files.

    textstats FILE [--top N]

Stop words (the, and, of, ...) are listed in `src/textstats/stopwords.txt`, which ships inside the package
and is read at run time with importlib.resources.

Install into the tools venv (the build box has no network, setuptools comes from the system site-packages):

    /app/venv/bin/pip install --no-index --no-build-isolation /app/textstats
EOF
cat > /app/textstats/src/textstats/__init__.py <<'EOF'
"""textstats: line, word and top-word statistics for plain-text files."""
__version__ = "1.2.0"
EOF
cat > /app/textstats/src/textstats/core.py <<'EOF'
import re
from collections import Counter
from importlib import resources

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def stopwords() -> set[str]:
    text = resources.files("textstats").joinpath("stopwords.txt").read_text(encoding="utf-8")
    return {w.strip() for w in text.split() if w.strip()}


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def top_words(text: str, n: int = 5) -> list[tuple[str, int]]:
    sw = stopwords()
    counts = Counter(w for w in words(text) if w not in sw)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def summary(text: str, n: int = 5) -> dict:
    return {"lines": len(text.splitlines()), "words": len(words(text)), "top": top_words(text, n)}
EOF
cat > /app/textstats/src/textstats/cli.py <<'EOF'
import argparse
import os

from .core import summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="textstats")
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)
    with open(a.path, encoding="utf-8") as fh:
        s = summary(fh.read(), a.top)
    print(f"file={os.path.basename(a.path)}")
    print(f"lines={s['lines']}")
    print(f"words={s['words']}")
    print("top:")
    for w, c in s["top"]:
        print(f"{w} {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
EOF
cat > /app/textstats/src/textstats/stopwords.txt <<'EOF'
a an and are as at be but by for from had has have he her his i in is it its
of on or our she so that the their them they this to was we were which with you
EOF
cat > /data/sample.txt <<'EOF'
The lighthouse keeper climbed the stairs at dusk, as he had done for thirty years.
The lamp was cleaned, the wick was trimmed, and the lens was polished until it shone.
Ships passed the lighthouse in the night; the keeper counted the ships and wrote them down.
In the morning the keeper slept, and the lighthouse stood quiet above the grey water.
Storms came in winter. The keeper's log recorded every storm, every ship, every lamp change.
EOF
python3 -m venv --system-site-packages /app/venv
