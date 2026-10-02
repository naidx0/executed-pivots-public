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
