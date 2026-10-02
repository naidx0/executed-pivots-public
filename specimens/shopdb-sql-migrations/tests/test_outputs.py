"""Outcome checks for shopdb-sql-migrations: the applied migrations 001 and 002 are unedited; shop.db records every
migration file exactly once, keeps its customers and orders, holds integer cents in orders.total_cents and has the
customer_totals view; migrate.py builds the same data from an empty database and is a no-op on an up-to-date one;
out/top_customers.csv is right."""
import csv
import hashlib
import io
import os
import shutil
import sqlite3
import subprocess
import sys

ok = True
PROJ = "/app/shopdb"
ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8"}
CUSTOMERS = [(1, "Ada", "ada@example.com"), (2, "Brook", "brook@example.com"), (3, "Chen", "chen@example.com"),
             (4, "Dana", "dana@example.com"), (5, "Eli", None)]
ORDERS = [(1, 1, 19.99, "2024-05-01"), (2, 2, 0.29, "2024-05-02"), (3, 1, 1.15, "2024-05-03"),
          (4, 3, 4.35, "2024-05-03"), (5, 4, 250.0, "2024-05-04"), (6, 2, 12.5, "2024-05-06"),
          (7, 3, 99.95, "2024-05-07"), (8, 1, 0.1, "2024-05-08"), (9, 4, 7.7, "2024-05-09"),
          (10, 2, 33.33, "2024-05-10"), (11, 3, 1.01, "2024-05-11"), (12, 1, 64.99, "2024-05-12")]
CENTS = {oid: int(round(total * 100)) for oid, _c, total, _d in ORDERS}


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


def run(cmd, cwd="/", extra=None):
    try:
        return subprocess.run(cmd, cwd=cwd, env={**ENV, **(extra or {})}, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        return subprocess.CompletedProcess(cmd, 127, "", repr(e))


def expected_totals():
    per = {cid: [name, 0, 0] for cid, name, _e in CUSTOMERS}
    for oid, cid, _t, _d in ORDERS:
        per[cid][1] += 1
        per[cid][2] += CENTS[oid]
    return sorted(((cid, *v) for cid, v in per.items()), key=lambda r: (-r[3], r[0]))


def db_checks(path, label):
    """The data checks every up-to-date database must pass (the deployed one and one built from scratch)."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
        custs = conn.execute("SELECT id, name, email FROM customers ORDER BY id").fetchall()
        orders = conn.execute("SELECT id, customer_id, total, placed_on FROM orders ORDER BY id").fetchall()
        cents = conn.execute("SELECT id, total_cents, typeof(total_cents) FROM orders ORDER BY id").fetchall()
        view = conn.execute("SELECT customer_id, name, orders, total_cents FROM customer_totals").fetchall()
    except sqlite3.Error as e:
        check(f"{label}: readable with the expected tables and view ({e})", False)
        return
    files = sorted(int(n.split("_")[0]) for n in os.listdir(f"{PROJ}/migrations") if n.endswith(".sql"))
    check(f"{label}: schema_migrations records every migration file once: files {files} recorded {versions}",
          versions == files and len(set(files)) == len(files) and files[:4] == [1, 2, 3, 4])
    check(f"{label}: customers kept ({len(custs)} rows)", custs == CUSTOMERS)
    check(f"{label}: orders kept ({len(orders)} rows)", orders == ORDERS)
    bad = [(i, c, t) for i, c, t in cents if t != "integer" or c != CENTS[i]]
    check(f"{label}: orders.total_cents is integer cents on every row (wrong: {bad[:4]})", not bad and len(cents) == 12)
    want = expected_totals()
    check(f"{label}: customer_totals view: want {want} got {view}", view == want)


# 1. applied migrations are not edited
shipped = {}
for ln in open("/tests/frozen.sha256").read().split("\n"):
    if ln.strip():
        h, n = ln.split("  ", 1)
        shipped[n] = h
for n, h in shipped.items():
    p = f"{PROJ}/migrations/{n}"
    got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else None
    check(f"applied migration {n} unedited", got == h)

# 2. the deployed database
db_checks(f"{PROJ}/shop.db", "shop.db")

# 3. migrate.py on an up-to-date copy is a no-op, and from an empty database builds the same data
work = "/tmp/verify-shopdb"
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
shutil.copy2(f"{PROJ}/shop.db", f"{work}/copy.db")
before = open(f"{work}/copy.db", "rb").read()
r = run([sys.executable, f"{PROJ}/migrate.py"], extra={"SHOPDB": f"{work}/copy.db"})
check(f"migrate.py on an up-to-date copy: rc {r.returncode}, nothing applied ({r.stdout.strip()[-120:]!r})",
      r.returncode == 0 and "applied" not in r.stdout and open(f"{work}/copy.db", "rb").read() == before)
r = run([sys.executable, f"{PROJ}/migrate.py"], cwd="/tmp", extra={"SHOPDB": f"{work}/fresh.db"})
check(f"migrate.py builds an empty database from scratch: rc {r.returncode} {(r.stdout + r.stderr).strip()[-300:]!r}",
      r.returncode == 0)
db_checks(f"{work}/fresh.db", "fresh database")

# 4. the report
want = io.StringIO()
w = csv.writer(want, lineterminator="\n")
w.writerow(["customer", "orders", "total"])
for _cid, name, n, c in expected_totals():
    w.writerow([name, n, f"{c // 100}.{c % 100:02d}"])
p = f"{PROJ}/out/top_customers.csv"
got = open(p).read() if os.path.isfile(p) else None
check(f"out/top_customers.csv: want {want.getvalue()!r} got {got!r}", got == want.getvalue())

print("RESULT", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
