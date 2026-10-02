mkdir -p /app/shop/migrations
cat > /tmp/seed.py <<'SEEDEOF'
"""Deterministic seed for sqlite-email-dedupe (a copy lives in setup.sh; the verifier rebuilds the pristine DB)."""
import sqlite3
import sys

CUSTOMERS = [
    (1, "ana.lopez@example.com", "Ana Lopez"), (2, "bob@shop.io", "Bob Stone"), (3, "Carol.Diaz@Example.com", "Carol Diaz"),
    (4, "dmitri@mail.net", "Dmitri Volkov"), (5, "erin@example.org", "Erin Walsh"), (6, "fatima@shop.io", "Fatima Noor"),
    (7, "BOB@shop.io", "Bob Stone"), (8, "gus@mail.net", "Gus Ferreira"), (9, "hana@example.com", "Hana Kim"),
    (10, " ana.lopez@example.com", "Ana López"), (11, "ivan@example.org", "Ivan Petrov"), (12, "Jade@Shop.io ", "Jade Moreau"),
    (13, "kofi@mail.net", "Kofi Mensah"), (14, "lena@example.com", "Lena Braun"), (15, "Erin@Example.org", "Erin Walsh"),
    (16, "mo@shop.io", "Mo Haddad"), (17, "nia@example.com", "Nia Clarke"), (18, "Ana.Lopez@example.com", "A. Lopez"),
    (19, "omar@mail.net", "Omar Aziz"), (20, "pia@example.org", "Pia Lund"), (21, "jade@shop.io", "Jade Moreau"),
    (22, "quinn@example.com", "Quinn Hale"), (23, "raj@shop.io", "Raj Iyer"), (24, "HANA@EXAMPLE.COM", "Hana Kim"),
    (25, "sol@mail.net", "Sol Reyes"), (26, "Tom@Mail.net", "Tom Berg"),
]


def orders():
    state = 20260924
    out = []
    for oid in range(1, 61):
        state = (state * 1103515245 + 12345) % 2**31
        cid = 1 + (state >> 8) % len(CUSTOMERS)
        state = (state * 1103515245 + 12345) % 2**31
        cents = 500 + (state >> 8) % 20000
        out.append((oid, cid, cents, f"2026-0{1 + oid % 8}-{1 + oid % 27:02d}"))
    return out


def build(path):
    db = sqlite3.connect(path)
    db.executescript("""
    CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL);
    CREATE TABLE customers (id INTEGER PRIMARY KEY, email TEXT NOT NULL, name TEXT NOT NULL);
    CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
                         total_cents INTEGER NOT NULL, placed_at TEXT NOT NULL);
    CREATE INDEX ix_orders_customer ON orders(customer_id);
    """)
    db.execute("INSERT INTO schema_migrations VALUES (1, '001_init', '2026-01-02T09:00:00Z')")
    db.executemany("INSERT INTO customers VALUES (?, ?, ?)", CUSTOMERS)
    db.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", orders())
    db.commit()
    db.close()


if __name__ == "__main__":
    build(sys.argv[1])
SEEDEOF
python3 /tmp/seed.py /app/shop/shop.db && rm /tmp/seed.py
cat > /app/shop/migrations/001_init.sql <<'EOF'
-- 001: initial schema
CREATE TABLE customers (id INTEGER PRIMARY KEY, email TEXT NOT NULL, name TEXT NOT NULL);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
                     total_cents INTEGER NOT NULL, placed_at TEXT NOT NULL);
CREATE INDEX ix_orders_customer ON orders(customer_id);
EOF
cat > /app/shop/migrations/002_unique_email.sql <<'EOF'
-- 002: one customer account per email address.
-- Emails are stored normalized (trimmed, lower case) and the index enforces uniqueness.
CREATE UNIQUE INDEX ux_customers_email ON customers(lower(trim(email)));
EOF
cat > /app/shop/migrate.py <<'EOF'
#!/usr/bin/env python3
"""Apply pending migrations from migrations/NNN_name.sql to shop.db, each in its own transaction."""
import datetime
import pathlib
import sqlite3
import sys

HERE = pathlib.Path(__file__).resolve().parent
DB = HERE / "shop.db"


def main() -> int:
    db = sqlite3.connect(DB, isolation_level=None)
    done = {v for (v,) in db.execute("SELECT version FROM schema_migrations")}
    pending = sorted(p for p in (HERE / "migrations").glob("[0-9][0-9][0-9]_*.sql") if int(p.name[:3]) not in done)
    if not pending:
        print("nothing to migrate; at version", max(done))
        return 0
    for p in pending:
        version = int(p.name[:3])
        print(f"applying {p.name} ...", flush=True)
        try:
            db.execute("BEGIN")
            for stmt in [s for s in p.read_text().split(";") if s.strip() and not all(
                    l.strip().startswith("--") or not l.strip() for l in s.splitlines())]:
                db.execute(stmt)
            db.execute("INSERT INTO schema_migrations VALUES (?, ?, ?)",
                       (version, p.stem, datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
            db.execute("COMMIT")
        except sqlite3.Error as e:
            db.execute("ROLLBACK")
            print(f"FAILED {p.name}: {e}", file=sys.stderr)
            return 1
        print(f"applied {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
EOF
cat > /app/shop/README.md <<'EOF'
# shop database

- `shop.db`: SQLite copy of the production shop database (customers, orders, schema_migrations)
- `migrations/NNN_name.sql`: schema migrations, applied in order by `python3 migrate.py`
- `migrate.py` records every applied migration in `schema_migrations`

The sqlite3 command-line shell is not installed on this host; use Python's sqlite3 module.
EOF
