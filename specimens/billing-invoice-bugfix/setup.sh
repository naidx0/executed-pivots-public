mkdir -p /app/billing /app/tests
cat > /app/README.md <<'EOF'
# billing

Small helpers used by the invoicing service: line totals with a bulk
discount, invoice totals, and pagination of invoice lines for the web UI.

Run the tests from this directory:

    python3 -m unittest discover -s tests -v
EOF
cat > /app/billing/__init__.py <<'EOF'
"""Billing helpers."""
EOF
cat > /app/billing/invoice.py <<'EOF'
"""Invoice helpers for the billing service."""
from decimal import Decimal, ROUND_HALF_UP

BULK_THRESHOLD = 10
BULK_RATE = Decimal("0.10")


def money(x):
    """Round to cents, half up."""
    return Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def bulk_discount(qty, unit_price):
    """Discount for one line: 10% of the subtotal for bulk quantities."""
    subtotal = Decimal(unit_price) * qty
    if qty > BULK_THRESHOLD:
        return money(subtotal * BULK_RATE)
    return Decimal("0.00")


def line_total(qty, unit_price):
    subtotal = Decimal(unit_price) * qty
    return money(subtotal - bulk_discount(qty, unit_price))


def invoice_total(lines):
    """lines: iterable of (qty, unit_price) pairs."""
    return money(sum((line_total(q, p) for q, p in lines), Decimal("0")))


def paginate(items, page, per_page=20):
    """Return the items shown on page `page`."""
    if page < 1:
        raise ValueError("page must be >= 1")
    start = page * per_page
    return items[start:start + per_page]


def page_count(n_items, per_page=20):
    return (n_items + per_page - 1) // per_page
EOF
cat > /app/tests/__init__.py <<'EOF'
EOF
cat > /app/tests/test_invoice.py <<'EOF'
import unittest
from decimal import Decimal

from billing.invoice import bulk_discount, invoice_total, line_total, page_count, paginate


class TestDiscount(unittest.TestCase):
    def test_no_discount_below_threshold(self):
        self.assertEqual(bulk_discount(9, "2.50"), Decimal("0.00"))

    def test_discount_at_threshold(self):
        self.assertEqual(bulk_discount(10, "2.50"), Decimal("2.50"))

    def test_line_total_with_discount(self):
        self.assertEqual(line_total(12, "1.99"), Decimal("21.49"))

    def test_invoice_total(self):
        self.assertEqual(invoice_total([(3, "4.00"), (10, "2.50")]), Decimal("34.50"))


class TestPagination(unittest.TestCase):
    items = list(range(1, 46))

    def test_first_page(self):
        self.assertEqual(paginate(self.items, 1, 20), list(range(1, 21)))

    def test_last_partial_page(self):
        self.assertEqual(paginate(self.items, 3, 20), [41, 42, 43, 44, 45])

    def test_page_count(self):
        self.assertEqual(page_count(45, 20), 3)


if __name__ == "__main__":
    unittest.main()
EOF
