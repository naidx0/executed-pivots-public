"""Outcome checks for sqlite-email-dedupe: rebuild the pristine DB from the seed, compute the expected merge,
and compare with /app/shop/shop.db. Also checks that migration 002 was applied through migrate.py."""
import hashlib
import os
import sqlite3
import subprocess
import sys

sys.path.insert(0, "/tests")
import seed  # noqa: E402

ok = True
DB = "/app/shop/shop.db"
MIG_002 = "dca0fa84345ea3710c0e42cd7f4eecb76644211f1a4e5f7fc54dd80366d353fd"
MIGRATE = "4f8097688b4b8b8d389747d828c6209e2b3bb9f34e32ceddc0d2db4a1e0866d7"


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else None


check("migrations/002_unique_email.sql unchanged", sha("/app/shop/migrations/002_unique_email.sql") == MIG_002)
check("migrate.py unchanged", sha("/app/shop/migrate.py") == MIGRATE)

# expected state
keep = {}
for cid, email, name in sorted(seed.CUSTOMERS):
    keep.setdefault(email.strip().lower(), (cid, name))
canon = {cid: keep[email.strip().lower()][0] for cid, email, _ in seed.CUSTOMERS}
want_customers = sorted((cid, em, name) for em, (cid, name) in keep.items())
want_orders = sorted((oid, canon[cid], cents, at) for oid, cid, cents, at in seed.orders())

if not os.path.isfile(DB):
    check("shop.db exists", False)
    sys.exit(1)
db = sqlite3.connect(DB)
vers = sorted(v for (v,) in db.execute("SELECT version FROM schema_migrations"))
check(f"schema_migrations records versions 1 and 2 (got {vers})", vers == [1, 2])
idx = db.execute("SELECT name FROM pragma_index_list('customers') WHERE name = 'ux_customers_email' AND \"unique\" = 1").fetchall()
sql = (db.execute("SELECT sql FROM sqlite_master WHERE name = 'ux_customers_email'").fetchone() or [""])[0] or ""
check(f"unique index ux_customers_email on lower(trim(email)) ({sql!r})",
      len(idx) == 1 and sql.replace(" ", "").endswith("customers(lower(trim(email)))"))
got_c = sorted(db.execute("SELECT id, email, name FROM customers").fetchall())
check(f"customers: {len(want_customers)} rows, lowest id kept per normalized email, names kept (got {len(got_c)})",
      got_c == want_customers)
bad = [e for (_, e, _) in got_c if e != e.strip().lower()]
check(f"all emails normalized (bad: {bad[:3]})", not bad)
got_o = sorted(db.execute("SELECT id, customer_id, total_cents, placed_at FROM orders").fetchall())
check(f"orders: all {len(want_orders)} present (got {len(got_o)})", len(got_o) == len(want_orders))
check("orders: amounts and dates unchanged", [(o[0], o[2], o[3]) for o in got_o] == [(o[0], o[2], o[3]) for o in want_orders])
wrong = [o for o, w in zip(got_o, want_orders) if o[1] != w[1]]
check(f"orders moved to the kept customer (wrong: {wrong[:3]})", got_o == want_orders)
check("PRAGMA foreign_key_check is empty", db.execute("PRAGMA foreign_key_check").fetchall() == [])
check("PRAGMA integrity_check is ok", db.execute("PRAGMA integrity_check").fetchone() == ("ok",))
db.close()
r = subprocess.run([sys.executable, "/app/shop/migrate.py"], cwd="/", capture_output=True, text=True, timeout=60)
check(f"migrate.py has nothing pending (rc={r.returncode}, out={r.stdout.strip()!r})",
      r.returncode == 0 and "nothing to migrate" in r.stdout)

print("RESULT", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
