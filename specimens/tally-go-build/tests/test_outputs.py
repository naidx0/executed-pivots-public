"""Outcome checks for tally-go-build: the test file is untouched; a clean copy builds, passes make test and links the
version from VERSION (also after VERSION changes); the built and the installed tally give the right counts and
tie order on a hidden input, from the PATH in /."""
import hashlib
import os
import re
import shutil
import subprocess
import sys
from collections import Counter

ok = True
PROJ = "/app/tally"
TEST_SHA = "5b788cb9f9cfad20807f90c8ba3eb058e09f4b6315bcd60170eaf1bcdd8fb1d4"
ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8"}
HIDDEN = """Owls hoot at night; owls sleep by day.
Night owls and day larks don't agree: larks sing, owls hoot.
'Night' is when the owl's day starts, and larks' day ends.
"""


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


def run(cmd, cwd="/", timeout=240):
    try:
        return subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return subprocess.CompletedProcess(cmd, 127, "", repr(e))


def expected(text, n):
    words = [w.strip("'") for w in re.findall(r"[a-z']+", text.lower())]
    c = Counter(w for w in words if w)
    top = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [f"words={sum(c.values())} distinct={len(c)}"] + [f"{w} {k}" for w, k in top]


def clean_copy(dst, version=None):
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(PROJ, dst, symlinks=True, ignore=shutil.ignore_patterns("bin"))
    if version:
        open(os.path.join(dst, "VERSION"), "w").write(version + "\n")
    return dst


work = "/tmp/verify-tally"
os.makedirs(work, exist_ok=True)
data = os.path.join(work, "notes.txt")
open(data, "w").write(HIDDEN)

tf = os.path.join(PROJ, "internal/agg/agg_test.go")
check("internal/agg/agg_test.go unchanged",
      os.path.isfile(tf) and hashlib.sha256(open(tf, "rb").read()).hexdigest() == TEST_SHA)

src = clean_copy("/tmp/verify-tally-src")
r = run(["make", "build"], cwd=src)
exe = os.path.join(src, "bin", "tally")
check(f"make build on a clean copy builds bin/tally (rc={r.returncode}) {r.stderr.strip()[-300:]!r}",
      r.returncode == 0 and os.path.isfile(exe))
r = run(["make", "test"], cwd=src)
check(f"make test passes on a clean copy (rc={r.returncode}) {(r.stdout + r.stderr).strip()[-300:]!r}",
      r.returncode == 0)
r = run([exe, "--version"], cwd=work)
check(f"clean build: --version prints 'tally 1.4.0' (got {r.stdout.strip()!r})", r.stdout.strip() == "tally 1.4.0")
want = expected(HIDDEN, 4)
r = run([exe, "--top", "4", data], cwd=work)
got = r.stdout.strip().splitlines()
check(f"clean build on the hidden input: want {want} got {got}", r.returncode == 0 and got == want)

src2 = clean_copy("/tmp/verify-tally-src2", version="2.0.1")
r = run(["make", "build"], cwd=src2)
r2 = run([os.path.join(src2, "bin", "tally"), "--version"], cwd=work)
check(f"the version comes from VERSION at build time: VERSION=2.0.1 gives {r2.stdout.strip()!r} (build rc="
      f"{r.returncode})", r.returncode == 0 and r2.stdout.strip() == "tally 2.0.1")

want3 = expected(HIDDEN, 3)
r = run([os.path.join(PROJ, "bin", "tally"), "--top", "3", data], cwd=work)
got = r.stdout.strip().splitlines()
check(f"/app/tally/bin/tally on the hidden input: want {want3} got {got}", r.returncode == 0 and got == want3)

check("tally is installed as /usr/local/bin/tally", os.path.isfile("/usr/local/bin/tally"))
r = run(["bash", "-c", "exec tally --version"], cwd="/")
check(f"tally on the PATH: --version prints 'tally 1.4.0' (got {r.stdout.strip()!r})", r.stdout.strip() == "tally 1.4.0")
want5 = expected(HIDDEN, 5)
r = run(["bash", "-c", 'exec tally --top 5 "$1"', "tally", data], cwd="/")
got = r.stdout.strip().splitlines()
check(f"tally on the PATH, from /, on the hidden input: want {want5} got {got} {r.stderr.strip()[-200:]!r}",
      r.returncode == 0 and got == want5)

print("RESULT", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
