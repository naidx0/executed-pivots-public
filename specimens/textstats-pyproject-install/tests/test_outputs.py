"""Outcome checks for textstats-pyproject-install: the venv has a complete, non-editable install that works
from anywhere, the sources are untouched, and a fresh wheel from /app/textstats is complete."""
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter

ok = True
VENV = "/app/venv"
PROJ = "/app/textstats"
# sha256 of the pristine sources (the task forbids changing them)
PRISTINE = {
    "src/textstats/__init__.py": "60b6a5780ba726f7e734fe371ffca9500649b5a411377eec2a744a09c0589e52",
    "src/textstats/core.py": "5ac44ab7d9ed5becca561ffc5681a9efe0eca6647e5a1eee43f58ed258def9d1",
    "src/textstats/cli.py": "08c531d93f9f558dfd8e601dd98abb847b34ac8982151a9a8009dbd68d9987e9",
    "src/textstats/stopwords.txt": "4927e39357b1ae55f814e35043f225acf77948565be4ea6a77aeebf8e2fc6470",
}
STOP = set("""a an and are as at be but by for from had has have he her his i in is it its
of on or our she so that the their them they this to was we were which with you""".split())
HIDDEN = """Rivers carry silt to the delta, and the delta grows.
Farmers plant rice in the delta; rice needs the river's water.
When the river floods, the farmers move their rice seedlings uphill.
The delta, the river and the rice: three words that every farmer knows.
"""


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


def expected(text, n):
    ws = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    c = Counter(w for w in ws if w not in STOP)
    top = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return ["file=notes.txt", f"lines={len(text.splitlines())}", f"words={len(ws)}", "top:"] + [f"{w} {k}" for w, k in top]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


for rel, h in PRISTINE.items():
    p = os.path.join(PROJ, rel)
    check(f"source unchanged: {rel}", os.path.isfile(p) and sha(p) == h)

env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8"}
work = "/tmp/verify-textstats"
os.makedirs(work, exist_ok=True)
open(os.path.join(work, "notes.txt"), "w").write(HIDDEN)

exe = os.path.join(VENV, "bin", "textstats")
check("console script /app/venv/bin/textstats exists", os.path.isfile(exe))
if os.path.isfile(exe):
    r = subprocess.run([exe, "notes.txt", "--top", "4"], cwd=work, env=env, capture_output=True, text=True, timeout=60)
    got = r.stdout.strip().splitlines()
    check(f"textstats on hidden input exits 0 (rc={r.returncode}, stderr={r.stderr.strip()[-200:]!r})", r.returncode == 0)
    check(f"textstats output on hidden input: want {expected(HIDDEN, 4)} got {got}", got == expected(HIDDEN, 4))

r = subprocess.run([os.path.join(VENV, "bin", "python"), "-c",
                    "import textstats, importlib.resources as r; print(textstats.__file__);"
                    " print(r.files('textstats').joinpath('stopwords.txt').is_file())"],
                   cwd=work, env=env, capture_output=True, text=True, timeout=60)
lines = r.stdout.split()
check(f"package imports from the venv's site-packages (non-editable): {lines[:1]}",
      r.returncode == 0 and lines and lines[0].startswith(VENV + "/lib/"))
check("installed package contains stopwords.txt", r.returncode == 0 and lines[-1:] == ["True"])

# a fresh wheel from a clean copy of the project must be complete and installable on this Python
src = "/tmp/verify-textstats-src"
shutil.rmtree(src, ignore_errors=True)
shutil.copytree(PROJ, src, ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"))
wh = "/tmp/verify-textstats-wheel"
os.makedirs(wh, exist_ok=True)
r = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-index", "--no-build-isolation", "--no-deps", "-q",
                    "-w", wh, src], env=env, capture_output=True, text=True, timeout=240)
wheels = glob.glob(os.path.join(wh, "textstats-*.whl"))
check(f"fresh wheel builds (rc={r.returncode}) {r.stderr.strip()[-300:]!r}", r.returncode == 0 and len(wheels) == 1)
if len(wheels) == 1:
    z = zipfile.ZipFile(wheels[0])
    names = z.namelist()
    for m in ("textstats/__init__.py", "textstats/core.py", "textstats/cli.py", "textstats/stopwords.txt"):
        check(f"wheel contains {m}", m in names)
    meta = next((n for n in names if n.endswith(".dist-info/METADATA")), None)
    eps = next((n for n in names if n.endswith(".dist-info/entry_points.txt")), None)
    ep_text = z.read(eps).decode() if eps else ""
    check("wheel declares console script textstats = textstats.cli:main",
          re.search(r"^textstats\s*=\s*textstats\.cli:main\s*$", ep_text, re.M) is not None)
    md = z.read(meta).decode() if meta else ""
    m = re.search(r"^Requires-Python:\s*(.+)$", md, re.M)
    from pip._vendor.packaging.specifiers import SpecifierSet
    pyver = "%d.%d.%d" % sys.version_info[:3]
    check(f"Requires-Python ({m.group(1).strip() if m else None}) admits this Python {pyver}",
          m is None or SpecifierSet(m.group(1).strip()).contains(pyver))
    check("wheel is version 1.2.0 of textstats", re.search(r"^Name: textstats$", md, re.M) is not None
          and re.search(r"^Version: 1\.2\.0$", md, re.M) is not None)

print("RESULT", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
