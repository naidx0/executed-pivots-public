mkdir -p /app/shopdb/migrations
cat > /app/shopdb/migrate.py <<'EOF'
"""Apply the pending SQL migrations in migrations/ to the shop database, in order, each exactly once.

    python3 migrate.py            apply everything pending
    python3 migrate.py --status   show applied and pending migrations

A migration is migrations/NNN_name.sql; NNN is its version. Each file runs in one transaction together with its
row in schema_migrations, so a failing migration leaves nothing behind. The database is $SHOPDB, or shop.db next
to this file.
"""
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("SHOPDB", os.path.join(HERE, "shop.db"))
MIGRATIONS = os.path.join(HERE, "migrations")


def applied(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    return {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}


def pending(conn):
    done = applied(conn)
    out = []
    for name in sorted(os.listdir(MIGRATIONS)):
        if not name.endswith(".sql"):
            continue
        version = name.split("_")[0]
        if version not in done:
            out.append((version, name))
    return out


def statements(sql):
    buf = ""
    for line in sql.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            yield buf.strip()
            buf = ""
    if buf.strip():
        yield buf.strip()


def main():
    conn = sqlite3.connect(DB, isolation_level=None)
    if "--status" in sys.argv:
        print("applied:", sorted(applied(conn)))
        print("pending:", [name for _, name in pending(conn)])
        return 0
    for version, name in pending(conn):
        with open(os.path.join(MIGRATIONS, name)) as fh:
            sql = fh.read()
        try:
            conn.execute("BEGIN")
            for stmt in statements(sql):
                conn.execute(stmt)
            conn.execute("INSERT INTO schema_migrations (version, name) VALUES (?, ?)", (version, name))
            conn.execute("COMMIT")
        except sqlite3.Error as e:
            conn.execute("ROLLBACK")
            print(f"migration {name} failed: {e}", file=sys.stderr)
            return 1
        print(f"applied {name}")
    print("up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
EOF
cat > /app/shopdb/report.py <<'EOF'
"""Write out/top_customers.csv (customer, orders, total in dollars) from the customer_totals view."""
import csv
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("SHOPDB", os.path.join(HERE, "shop.db"))

rows = sqlite3.connect(DB).execute("SELECT name, orders, total_cents FROM customer_totals").fetchall()
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
path = os.path.join(HERE, "out", "top_customers.csv")
with open(path, "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["customer", "orders", "total"])
    for name, orders, cents in rows:
        w.writerow([name, orders, f"{cents // 100}.{cents % 100:02d}"])
print(open(path).read(), end="")
EOF
cat > /app/shopdb/migrations/001_init.sql <<'EOF'
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    total REAL NOT NULL,
    placed_on TEXT NOT NULL
);
EOF
cat > /app/shopdb/migrations/002_seed.sql <<'EOF'
INSERT INTO customers (id, name, email) VALUES
    (1, 'Ada', 'ada@example.com'),
    (2, 'Brook', 'brook@example.com'),
    (3, 'Chen', 'chen@example.com'),
    (4, 'Dana', 'dana@example.com'),
    (5, 'Eli', NULL);
INSERT INTO orders (id, customer_id, total, placed_on) VALUES
    (1, 1, 19.99, '2024-05-01'),
    (2, 2, 0.29, '2024-05-02'),
    (3, 1, 1.15, '2024-05-03'),
    (4, 3, 4.35, '2024-05-03'),
    (5, 4, 250.0, '2024-05-04'),
    (6, 2, 12.5, '2024-05-06'),
    (7, 3, 99.95, '2024-05-07'),
    (8, 1, 0.1, '2024-05-08'),
    (9, 4, 7.7, '2024-05-09'),
    (10, 2, 33.33, '2024-05-10'),
    (11, 3, 1.01, '2024-05-11'),
    (12, 1, 64.99, '2024-05-12');
EOF
cat > /app/shopdb/migrations/003_orders_total_cents.sql <<'EOF'
-- store order totals as integer cents
ALTER TABLE orders ADD COLUMN total_cents INTEGER NOT NULL;
UPDATE orders SET total_cents = total * 100;
EOF
cat > /app/shopdb/migrations/004_customer_totals_view.sql <<'EOF'
-- per-customer order count and spend, biggest spenders first
CREATE VIEW customer_totals AS
SELECT c.id AS customer_id, c.name AS name, COUNT(o.id) AS orders, COALESCE(SUM(o.total_cents), 0) AS total_cents
FROM customers c LEFT JOIN orders o ON o.customer_id = c.id
GROUP BY c.id
ORDER BY total_cents DESC, c.id;
EOF
# the database as deployed: 001 and 002 applied (by the previous, working release of the runner)
python3 - <<'EOF'
import sqlite3
conn = sqlite3.connect("/app/shopdb/shop.db")
conn.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL)")
for v, name in [(1, "001_init.sql"), (2, "002_seed.sql")]:
    conn.executescript(open(f"/app/shopdb/migrations/{name}").read())
    conn.execute("INSERT INTO schema_migrations VALUES (?, ?)", (v, name))
conn.commit()
EOF
