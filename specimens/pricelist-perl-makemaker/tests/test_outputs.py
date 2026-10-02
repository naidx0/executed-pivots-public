"""Outcome checks for pricelist-perl-makemaker: the test files are untouched; a clean copy builds with
`perl Makefile.PL && make`, passes `make test` and puts the script in blib; the installed pricelist-report
(/usr/local/bin, the module from the site library) prints the README's report for a hidden price list, from /, with
the installed module at version 0.03."""
import hashlib
import os
import shutil
import subprocess
import sys

ok = True
PROJ = "/app/Acme-Pricelist"
ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C"}
HIDDEN = """sku,qty,unit_price
# spring list
Q1,1,9.95
R22,4,24.99
S3,10,10.00
T4,1,100

U5,3,33.33
V6,20,0.05
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


def expected(text):
    items = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("sku,"):
            continue
        sku, qty, price = s.split(",")
        items.append((sku, int(qty), float(price), int(qty) * float(price)))
    items.sort(key=lambda x: (-x[3], x[0]))
    total = sum(x[3] for x in items)
    out = ["%-8s %4d x %8.2f = %9.2f" % x for x in items]
    out += ["TOTAL %.2f" % total, "AVG %.2f" % (total / len(items))]
    return "\n".join(out) + "\n"


for ln in open("/tests/manifest.sha256").read().split("\n"):
    if ln.strip():
        h, p = ln.split("  ", 1)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else None
        check(f"{p} unchanged", got == h)

work = "/tmp/verify-pricelist"
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
data = f"{work}/spring.csv"
open(data, "w").write(HIDDEN)
want = expected(HIDDEN)

src = f"{work}/src"
shutil.copytree(PROJ, src, symlinks=True, ignore=shutil.ignore_patterns(
    "blib", "Makefile", "Makefile.old", "pm_to_blib", "MYMETA.*", "*.tar.gz"))
r = run(["bash", "-c", "perl Makefile.PL && make"], cwd=src)
check(f"clean copy: perl Makefile.PL && make (rc={r.returncode}) {(r.stdout + r.stderr).strip()[-300:]!r}",
      r.returncode == 0 and os.path.isfile(f"{src}/blib/script/pricelist-report"))
r = run(["make", "test"], cwd=src)
check(f"clean copy: make test passes (rc={r.returncode}) {(r.stdout + r.stderr).strip()[-300:]!r}",
      r.returncode == 0 and "Result: PASS" in r.stdout)
r = run(["perl", "-Mblib", "blib/script/pricelist-report", data], cwd=src)
check(f"clean copy's blib script on the hidden list: want {want!r} got {r.stdout!r} {r.stderr[-200:]!r}",
      r.returncode == 0 and r.stdout == want)

check("pricelist-report is installed in /usr/local/bin", os.path.isfile("/usr/local/bin/pricelist-report"))
r = run(["bash", "-c", 'exec pricelist-report "$1"', "x", data], cwd="/")
check(f"installed pricelist-report from / on the hidden list: got {r.stdout!r} {r.stderr[-200:]!r}",
      r.returncode == 0 and r.stdout == want)
r = run(["perl", "-MAcme::Pricelist", "-e", "print $INC{'Acme/Pricelist.pm'}, ' ', $Acme::Pricelist::VERSION"], cwd="/")
check(f"the installed module is loaded from the site library at 0.03 ({r.stdout!r} {r.stderr[-150:]!r})",
      r.returncode == 0 and r.stdout.startswith("/usr/local/share/perl/") and r.stdout.endswith(" 0.03"))

print("RESULT", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
