"""Outcome checks for reportgen-pip-constraints: the code and the wheelhouse are untouched; the pip command from the
task succeeds in a fresh venv with the fixed requirements.txt and constraints.txt and gives the right versions and
a working tool; /app/venv holds the same versions with no broken requirement; the report and the lock file are
right; the launcher works from / on a hidden input."""
import hashlib
import os
import shutil
import subprocess
import sys
from collections import defaultdict

ok = True
PROJ = "/app/reportgen"
WH = "/opt/wheelhouse"
ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8"}
WANT = {"datesy": "1.3.2", "decimalx": "2.1.0", "moneyfmt": "1.0.0", "tinyfmt": "1.2.0"}
ORDERS = """date,customer,amount_cents
2024-01-03,acme,125000
2024-01-17,globex,4999
2024-01-29,initech,98001
2024-02-02,acme,250050
2024-02-14,umbrella,1999
2024-03-01,globex,1000000
2024-03-09,acme,75
2024-03-30,initech,33333
"""  # /data/orders.csv as shipped
HIDDEN = """date,customer,amount_cents
2023-11-30,acme,99
2023-12-01,globex,123456789
2023-12-31,acme,1
2024-02-29,initech,100000
2024-02-01,umbrella,5
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


def sha(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError:
        return None


def expected(text):
    months = defaultdict(lambda: [0, 0])
    lines = text.strip().splitlines()[1:]
    for ln in lines:
        d, _c, amt = ln.split(",")
        m = months[d[:7]]
        m[0] += 1
        m[1] += int(amt)
    rows = [[k, str(n), f"${c // 100:,}.{c % 100:02d}"] for k, (n, c) in sorted(months.items())]
    hdr = ["month", "orders", "revenue"]
    w = [max(len(hdr[i]), *(len(r[i]) for r in rows)) for i in range(3)]
    line = lambda cells: "  ".join(c.ljust(x) for c, x in zip(cells, w)).rstrip()
    return "\n".join([line(hdr), line(["-" * x for x in w])] + [line(r) for r in rows]) + "\n"


def freeze(py):
    r = run([py, "-m", "pip", "freeze", "--all"])
    got = {}
    for ln in r.stdout.splitlines():
        if "==" in ln:
            n, v = ln.split("==", 1)
            got[n.strip().lower()] = v.strip()
    return {k: v for k, v in got.items() if k not in ("pip", "setuptools", "wheel")}


# 1. the code and the wheelhouse are as shipped
manifest = open("/tests/manifest.sha256").read().split("\n") if os.path.exists("/tests/manifest.sha256") else []
bad = []
for ln in manifest:
    if ln.strip():
        h, p = ln.split("  ", 1)
        if sha(p) != h:
            bad.append(p)
check(f"reportgen code and wheelhouse unchanged ({len(manifest) - 1} files; changed: {bad})", manifest and not bad)
check("no extra files in the wheelhouse", sorted(os.listdir(WH)) == sorted(
    os.path.basename(ln.split("  ", 1)[1]) for ln in manifest if ln.strip() and ln.split("  ", 1)[1].startswith(WH)))

# 2. the task's pip command in a fresh venv
fresh = "/tmp/verify-reportgen-venv"
shutil.rmtree(fresh, ignore_errors=True)
r = run([sys.executable, "-m", "venv", fresh])
r = run([f"{fresh}/bin/pip", "install", "--no-index", "--find-links", WH, "-r", "requirements.txt", "-c",
         "constraints.txt"], cwd=PROJ)
check(f"pip install -r requirements.txt -c constraints.txt succeeds in a fresh venv (rc={r.returncode}) "
      f"{(r.stdout + r.stderr).strip()[-300:]!r}", r.returncode == 0)
got = freeze(f"{fresh}/bin/python")
check(f"fresh venv versions: want {WANT} got {got}", got == WANT)

hidden = "/tmp/verify-reportgen-orders.csv"
open(hidden, "w").write(HIDDEN)
want_hidden = expected(HIDDEN)
code = "import sys; sys.path.insert(0, '/app/reportgen'); from reportgen import main; sys.exit(main(sys.argv[1:]))"
r = run([f"{fresh}/bin/python", "-c", code, hidden])
check(f"fresh venv runs reportgen on the hidden input: want {want_hidden!r} got {r.stdout!r} {r.stderr[-200:]!r}",
      r.returncode == 0 and r.stdout == want_hidden)

# 3. /app/venv
got = freeze("/app/venv/bin/python")
check(f"/app/venv versions: want {WANT} got {got}", got == WANT)
r = run(["/app/venv/bin/python", "-m", "pip", "check"])
check(f"/app/venv: pip check (rc={r.returncode}) {r.stdout.strip()[-200:]!r}", r.returncode == 0)
r = run(["bash", "-c", 'exec /app/reportgen/bin/reportgen "$1"', "x", hidden], cwd="/")
check(f"bin/reportgen from / on the hidden input: got {r.stdout!r} {r.stderr[-200:]!r}",
      r.returncode == 0 and r.stdout == want_hidden)

# 4. outputs
want = expected(ORDERS)
got = open("/app/out/monthly.txt").read() if os.path.isfile("/app/out/monthly.txt") else None
check(f"/app/out/monthly.txt: want {want!r} got {got!r}", got == want)
lock = f"{PROJ}/requirements.lock"
lines = []
if os.path.isfile(lock):
    lines = [ln.strip() for ln in open(lock) if ln.strip() and not ln.lstrip().startswith("#")]
pins = {}
for ln in lines:
    n, _, v = ln.partition("==")
    pins[n.strip().lower().replace("_", "-")] = v.strip()
check(f"requirements.lock pins exactly the installed versions: want {WANT} got {pins}",
      pins == WANT and len(lines) == len(WANT))

print("RESULT", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
