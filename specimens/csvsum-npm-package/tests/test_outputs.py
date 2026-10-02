"""Outcome checks for csvsum-npm-package: the tests are untouched and pass on a clean copy, the package loads with
require, a tarball packed from a clean copy installs on its own and gives a working csvsum, and the global csvsum on
the PATH works from / on a hidden input."""
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile

ok = True
PROJ = "/app/csvsum"
TEST_SHA = "8f22c8d261bd1be035756e9529c9f955ba1c6748538e4e4e9fb631f545b9647b"
ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8"}
HIDDEN = "id,amount,fee,note\n1, 10.25,0.30,a\n2,-3.5,,b\n3,,1.20,c\n\n4,100,0.05,d\n5,7.8,2,e\n"
WANT = {"amount": "count=4 total=114.55 mean=28.64 min=-3.50 max=100.00",
        "fee": "count=4 total=3.55 mean=0.89 min=0.05 max=2.00"}


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


def run(cmd, cwd="/", timeout=120):
    try:
        return subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return subprocess.CompletedProcess(cmd, 127, "", repr(e))


def clean_copy(dst):
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(PROJ, dst, symlinks=True, ignore=shutil.ignore_patterns("node_modules", "*.tgz"))
    return dst


work = "/tmp/verify-csvsum"
os.makedirs(work, exist_ok=True)
data = os.path.join(work, "orders.csv")
open(data, "w").write(HIDDEN)

tf = os.path.join(PROJ, "test/index.test.js")
check("test/index.test.js unchanged", os.path.isfile(tf) and hashlib.sha256(open(tf, "rb").read()).hexdigest() == TEST_SHA)

src = clean_copy("/tmp/verify-csvsum-src")
r = run(["npm", "test"], cwd=src, timeout=180)
check(f"npm test passes on a clean copy (rc={r.returncode}) {r.stdout.strip()[-300:]!r}", r.returncode == 0)

js = ("const m = require('/app/csvsum'); const fs = require('fs');"
      "const t = fs.readFileSync(process.argv[1], 'utf8');"
      "console.log(typeof m.summarize, typeof m.format);"
      "console.log(m.format(m.summarize(t))); console.log(m.format(m.summarize(t, 'fee')));")
r = run(["node", "-e", js, data], cwd=work)
got = r.stdout.strip().splitlines()
check(f"require('/app/csvsum') exports summarize and format and sums the hidden file: got {got} "
      f"{r.stderr.strip()[-200:]!r}", got == ["function function", WANT["amount"], WANT["fee"]])

packdir = "/tmp/verify-csvsum-pack"
shutil.rmtree(packdir, ignore_errors=True)
os.makedirs(packdir)
r = run(["npm", "pack", "--pack-destination", packdir, src], cwd=packdir, timeout=180)
tgz = glob.glob(os.path.join(packdir, "*.tgz"))
check(f"npm pack of a clean copy gives one tarball (rc={r.returncode}) {r.stderr.strip()[-200:]!r}",
      r.returncode == 0 and len(tgz) == 1)
names = tarfile.open(tgz[0]).getnames() if len(tgz) == 1 else []
check(f"tarball ships bin/csvsum.js and src/index.js: {sorted(names)}",
      "package/bin/csvsum.js" in names and "package/src/index.js" in names)

prefix = "/tmp/verify-csvsum-prefix"
shutil.rmtree(prefix, ignore_errors=True)
if len(tgz) == 1:
    r = run(["npm", "install", "-g", "--prefix", prefix, tgz[0]], cwd=packdir, timeout=180)
    check(f"the tarball installs into a fresh prefix (rc={r.returncode}) {r.stderr.strip()[-200:]!r}", r.returncode == 0)
else:
    check("the tarball installs into a fresh prefix (no tarball)", False)
exe = os.path.join(prefix, "bin", "csvsum")
for col, argv in (("amount", [data]), ("fee", [data, "fee"])):
    r = run([exe, *argv], cwd=work)
    check(f"packed csvsum {' '.join(argv[1:]) or '(default column)'}: want {WANT[col]!r} got {r.stdout.strip()!r} "
          f"{r.stderr.strip()[-200:]!r}", r.returncode == 0 and r.stdout.strip() == WANT[col])

g = os.path.join("/usr/local/lib/node_modules/csvsum/package.json")
check("csvsum is installed globally (/usr/local/lib/node_modules/csvsum)", os.path.isfile(g))
for col, argv in (("amount", [data]), ("fee", [data, "fee"])):
    r = run(["bash", "-c", 'exec csvsum "$@"', "csvsum", *argv], cwd="/")
    check(f"global csvsum on the PATH, from /, {' '.join(argv[1:]) or '(default column)'}: want {WANT[col]!r} "
          f"got {r.stdout.strip()!r} {r.stderr.strip()[-200:]!r}", r.returncode == 0 and r.stdout.strip() == WANT[col])
r = run(["bash", "-c", 'exec csvsum "$@"', "csvsum", data, "price"], cwd="/")
check(f"global csvsum rejects an unknown column (rc={r.returncode}, stderr={r.stderr.strip()[-120:]!r})",
      r.returncode != 0 and "no column price" in r.stderr)

print("RESULT", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
