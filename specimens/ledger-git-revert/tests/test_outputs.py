"""Outcome checks for ledger-git-revert."""
import os
import subprocess
import sys
import tempfile

REPO = "/app/ledger"
ORIGINAL = {  # fixed author/committer dates in setup.sh make these hashes identical on every build
    "d12b1ce8617b14e6f02b7c05a078733e240c6af9": "Initial ledger with tax helpers",
    "84bda7f5f3cc27e82d43618012ed0614e14b3794": "Add money formatting helper",
    "cdf209be747ebb3cade45c0e584d443306264301": "Speed up tax computation",
    "a72c823d70e27b4447a2864a1f308ba7318ff11a": "Add changelog for 1.1",
    "c8c1125e57654548f411bc90d6ec7e4bcb0a5d24": "Add zero-rated tax category",
}
ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


def git(*args):
    r = subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout.strip()


rc, head = git("symbolic-ref", "--short", "HEAD")
check(f"main checked out (HEAD={head})", rc == 0 and head == "main")
rc, main = git("rev-parse", "--verify", "main^{commit}")
check("main exists", rc == 0)
for sha, subject in ORIGINAL.items():
    rc, _ = git("merge-base", "--is-ancestor", sha, "main")
    check(f"original commit kept in history: {subject}", rc == 0)
rc, n = git("rev-list", "--count", "main")
check(f"a new commit was made on main (count={n})", rc == 0 and n.isdigit() and int(n) > len(ORIGINAL))
rc, st = git("status", "--porcelain")
check(f"working tree clean ({st!r})", rc == 0 and st == "")
rc, rel = git("rev-parse", "--verify", "refs/heads/release/1.1^{commit}")
check("release/1.1 points at main", rc == 0 and rel == main)

# test the committed tree of main (not whatever is lying around in the working tree)
with tempfile.TemporaryDirectory() as td:
    subprocess.run(f"git -C {REPO} archive main | tar -x -C {td}", shell=True, check=False)
    r = subprocess.run([sys.executable, "-m", "unittest", "-v"], cwd=td, capture_output=True, text=True, timeout=120)
    print(r.stderr[-1200:])
    check("committed test suite passes", r.returncode == 0 and "Ran 6 tests" in r.stderr)
    check("later commits kept (changelog, fmt)", os.path.exists(os.path.join(td, "CHANGELOG.md"))
          and os.path.exists(os.path.join(td, "ledger", "fmt.py")))
    r = subprocess.run([sys.executable, "-c",
                        "from ledger.tax import tax_for, gross\n"
                        "assert tax_for(10.10, 'reduced') == 0.51\n"
                        "assert tax_for(0.99) == 0.2\n"
                        "assert tax_for(50.0, 'zero') == 0.0\n"
                        "assert gross(19.99) == 23.99\n"
                        "print('hidden ok')"], cwd=td, capture_output=True, text=True, timeout=60)
    print(r.stdout, r.stderr[-600:])
    check("hidden tax checks", r.returncode == 0)

sys.exit(0 if ok else 1)
