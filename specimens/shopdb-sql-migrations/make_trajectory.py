"""Writes trajectory.json for shopdb-sql-migrations (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


def q(sql, db="shop.db"):
    return (f"python3 -c \"import sqlite3; [print(r) for r in sqlite3.connect('{db}').execute("
            f"\\\"{sql}\\\")]\"\n")


FIX = """cat > migrations/005_orders_total_cents_integer.sql <<'EOF'
-- 003 stored total * 100 as REAL for totals like 19.99 (1998.9999999999998): store whole cents
UPDATE orders SET total_cents = CAST(ROUND(total * 100) AS INTEGER);
EOF
"""

traj = [
    turn("I'm at /app. The shop database is at schema version 2 and the pending migrations have to be applied with "
         "migrate.py. First the layout and what the runner thinks is pending.",
         "cd into /app/shopdb, list it and the migrations, and run migrate.py --status (read-only).",
         [("cd /app/shopdb\n", 0.1), ("ls -la . migrations\n", 0.1), ("python3 migrate.py --status\n", 0.5)]),
    turn("There are four migrations (001_init, 002_seed, 003_orders_total_cents, 004_customer_totals_view) and "
         "shop.db. --status says applied: [1, 2] but lists all four files as pending, including 001 and 002, which "
         "are already applied.",
         "Read how the runner decides what is pending, and the two migrations that should run.",
         [("sed -n '/^def applied/,/^def statements/p' migrate.py\n", 0.1),
          ("cat migrations/003_orders_total_cents.sql migrations/004_customer_totals_view.sql\n", 0.1)]),
    turn("pending() compares `version = name.split(\"_\")[0]`, the string '001', with the applied versions, which "
         "are integers (1, 2), so nothing ever matches and 001 would run again. 003 adds orders.total_cents as "
         "INTEGER NOT NULL and fills it with total * 100; 004 creates the customer_totals view on total_cents.",
         "Back up the database, then run migrate.py once to confirm how it fails.",
         [("cp shop.db /tmp/shop.db.bak\n", 0.1), ("python3 migrate.py; echo rc=$?\n", 0.5)]),
    turn("As expected: `migration 001_init.sql failed: table customers already exists`, rc=1. The runner rolled it "
         "back, so the database is unchanged.",
         "Compare versions as integers, check --status again, and run the migrations.",
         [("sed -i 's/        version = name.split(\"_\")\\[0\\]/        version = int(name.split(\"_\")[0])/' "
           "migrate.py && grep -n 'version = ' migrate.py\n", 0.1),
          ("python3 migrate.py --status\n", 0.5), ("python3 migrate.py; echo rc=$?\n", 0.5)]),
    turn("--status now lists only 003 and 004 as pending. 003 fails: `Cannot add a NOT NULL column with default value "
         "NULL`, rc=1 (rolled back). SQLite cannot add a NOT NULL column to a table with rows unless it has a "
         "default; 003 has not been applied yet, so it can still be fixed.",
         "Give the new column DEFAULT 0 in 003, run the migrations, then run the report.",
         [("sed -i 's/ADD COLUMN total_cents INTEGER NOT NULL;/ADD COLUMN total_cents INTEGER NOT NULL DEFAULT 0;/' "
           "migrations/003_orders_total_cents.sql && head -n 3 migrations/003_orders_total_cents.sql\n", 0.1),
          ("python3 migrate.py; echo rc=$?\n", 0.5), ("python3 report.py\n", 0.5)]),
    turn("003 and 004 applied (up to date, rc=0), but report.py fails: `ValueError: Unknown format code 'd' for "
         "object of type 'float'` on the cents. total * 100 is a float, and SQLite keeps it REAL in an INTEGER column "
         "when it is not a whole number.",
         "Look at total, total_cents and typeof(total_cents) in orders.",
         [(q("SELECT id, total, total_cents, typeof(total_cents) FROM orders ORDER BY id LIMIT 5"), 0.5),
          (q("SELECT typeof(total_cents), count(*) FROM orders GROUP BY 1"), 0.5)]),
    turn("Order 1 has total 19.99 and total_cents 1998.9999999999998 (real), order 2 0.29 -> 28.999999999999996; "
         "5 rows are real and 7 integer. 003 is applied and recorded now, so it must not be edited: the fix goes in "
         "a new migration.",
         "Add 005 that rounds total * 100 to whole integer cents, run migrate.py, and recheck the types.",
         [(FIX, 0.1), ("python3 migrate.py; echo rc=$?\n", 0.5),
          (q("SELECT typeof(total_cents), count(*) FROM orders GROUP BY 1"), 0.5),
          (q("SELECT id, total, total_cents FROM orders ORDER BY id LIMIT 4"), 0.5)]),
    turn("005 applied (rc=0): all 12 rows are integer now, 19.99 -> 1999, 0.29 -> 29, 1.15 -> 115, 4.35 -> 435.",
         "Run the report and show the migration history.",
         [("python3 report.py\n", 0.5), (q("SELECT * FROM schema_migrations ORDER BY version"), 0.5)]),
    turn("out/top_customers.csv is written: Dana 2 257.70, Chen 3 105.31, Ada 4 86.23, Brook 3 46.12, Eli 0 0.00. "
         "schema_migrations has 1 to 5, once each.",
         "Check that migrate.py also builds a database from scratch now: run it on an empty file in /tmp and compare "
         "the cents and the view.",
         [("SHOPDB=/tmp/fresh.db python3 migrate.py; echo rc=$?\n", 1.0),
          (q("SELECT * FROM customer_totals", "/tmp/fresh.db"), 0.5),
          (q("SELECT typeof(total_cents), count(*) FROM orders GROUP BY 1", "/tmp/fresh.db"), 0.5)]),
    turn("From an empty database the runner applies 001 to 005 (rc=0), and the fresh copy has 12 integer cents and "
         "the same customer_totals rows. The runner is fixed, the migrations are applied once each, the cents are "
         "integers and the report is written.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
