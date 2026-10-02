"""Outcome checks: the visible tests (pristine copy) pass, hidden edge cases pass, tests untouched."""
import filecmp
import subprocess
import sys
from decimal import Decimal

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


check("visible tests unmodified", filecmp.cmp("/app/tests/test_invoice.py", "/tests/visible_test_invoice.py", shallow=False))

# run the pristine visible tests against the library in /app
r = subprocess.run([sys.executable, "-m", "unittest", "-v", "visible_test_invoice"],
                   cwd="/tests", env={"PYTHONPATH": "/app", "PATH": "/usr/bin:/bin"},
                   capture_output=True, text=True, timeout=120)
print(r.stderr[-1500:])
check("visible suite passes", r.returncode == 0)

sys.path.insert(0, "/app")
try:
    from billing.invoice import bulk_discount, invoice_total, line_total, page_count, paginate
except Exception as e:  # noqa: BLE001
    check(f"import billing.invoice ({e})", False)
    print("reward 0")
    sys.exit(1)

items = list(range(100, 157))  # 57 items
try:
    check("hidden: page 1 of 10", paginate(items, 1, 10) == list(range(100, 110)))
    check("hidden: page 2 of 10", paginate(items, 2, 10) == list(range(110, 120)))
    check("hidden: last page", paginate(items, 6, 10) == list(range(150, 157)))
    check("hidden: past the end", paginate(items, 7, 10) == [])
    check("hidden: default per_page", paginate(items, 1) == list(range(100, 120)))
    try:
        paginate(items, 0, 10)
        check("hidden: page 0 rejected", False)
    except ValueError:
        check("hidden: page 0 rejected", True)
    check("hidden: page_count", page_count(57, 10) == 6 and page_count(0, 10) == 0)
    check("hidden: discount at 9", bulk_discount(9, "10.00") == Decimal("0.00"))
    check("hidden: discount at 10", bulk_discount(10, "10.00") == Decimal("10.00"))
    check("hidden: discount at 11", bulk_discount(11, "3.33") == Decimal("3.66"))
    check("hidden: line total at 10", line_total(10, "0.99") == Decimal("8.91"))
    check("hidden: invoice total", invoice_total([(1, "5.00"), (10, "1.00"), (20, "0.50")]) == Decimal("23.00"))
except Exception as e:  # noqa: BLE001
    check(f"hidden checks raised {e!r}", False)

sys.exit(0 if ok else 1)
