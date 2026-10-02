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
