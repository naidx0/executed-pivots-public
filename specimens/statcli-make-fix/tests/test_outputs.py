"""Outcome checks for statcli-make-fix: clean build, make test, untouched cases, hidden inputs."""
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile

ok = True
# Build a fresh copy without build/ and bin/: this checks a from-scratch build and avoids deleting
# directories that live in lower overlay layers (rm -rf of those fails with EIO on the overlay backend).
ROOT = os.path.join(tempfile.mkdtemp(prefix="statcli-verify-"), "statcli")
shutil.copytree("/app/statcli", ROOT, ignore=shutil.ignore_patterns("build", "bin", "*.o"))


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


def sh(cmd, stdin=None):
    return subprocess.run(cmd, shell=True, cwd=ROOT, input=stdin, capture_output=True, text=True, timeout=180)


orig = "/tests/cases_orig"
for name in sorted(os.listdir(orig)):
    p = os.path.join("/app/statcli", "tests", "cases", name)
    check(f"case {name} unmodified", os.path.exists(p) and filecmp.cmp(p, os.path.join(orig, name), shallow=False))

r = sh("make")
print(r.stdout[-800:], r.stderr[-800:])
check("from-scratch `make` succeeds", r.returncode == 0 and os.access(os.path.join(ROOT, "bin", "statcli"), os.X_OK))
r = sh("make test")
print(r.stdout[-800:], r.stderr[-400:])
check("`make test` passes", r.returncode == 0)


def expected(xs):
    n = len(xs)
    m = sum(xs) / n
    s = sorted(xs)
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    sd = (sum((x - m) ** 2 for x in xs) / n) ** 0.5
    return n, m, med, sd


hidden = [
    [5.5, 5.1, 5.9, 5.2, 5.3],
    [0.25, 0.75],
    [-1.5, -1.2, -1.9, -1.1],
    [7.0],
    [((i * 37) % 101) / 10.0 for i in range(1, 102)],
    [1e3, 999.9, 1000.4, 999.5, 1000.2, 1000.1],
]
for xs in hidden:
    if not os.access(os.path.join(ROOT, "bin", "statcli"), os.X_OK):
        check("binary exists for hidden cases", False)
        break
    r = sh("./bin/statcli", " ".join(repr(x) for x in xs) + "\n")
    try:
        got = dict(line.split("=", 1) for line in r.stdout.split())
        n, m, med, sd = expected(xs)
        good = (int(got["count"]) == n and abs(float(got["mean"]) - m) < 0.0011
                and abs(float(got["median"]) - med) < 0.0011 and abs(float(got["stddev"]) - sd) < 0.0011
                and all(len(got[k].split(".")[-1]) == 3 for k in ("mean", "median", "stddev")))
    except (ValueError, KeyError):
        good = False
    check(f"hidden input of {len(xs)} values -> {r.stdout.split()}", good)

sys.exit(0 if ok else 1)
