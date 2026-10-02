# Private git identity for this world (the host's ~/.gitconfig may require commit signing).
cat > /root/.gitconfig <<'EOF'
[user]
	name = Priya Raman
	email = priya@ledger.example
[init]
	defaultBranch = main
[commit]
	gpgsign = false
[advice]
	detachedHead = false
EOF
export GIT_AUTHOR_NAME="Priya Raman" GIT_AUTHOR_EMAIL=priya@ledger.example
export GIT_COMMITTER_NAME="Priya Raman" GIT_COMMITTER_EMAIL=priya@ledger.example
commit() {  # commit <iso date> <message>: fixed dates keep the hashes identical on every build
  git add -A
  GIT_AUTHOR_DATE="$1" GIT_COMMITTER_DATE="$1" git commit -q -m "$2"
}
mkdir -p /app/ledger/ledger /app/ledger/tests
cd /app/ledger
git init -q -b main

cat > ledger/__init__.py <<'EOF'
"""Ledger: tax helpers."""
EOF
cat > ledger/tax.py <<'EOF'
"""Tax calculation."""

RATES = {
    "standard": 0.20,
    "reduced": 0.05,
}


def tax_for(amount, category="standard"):
    """Tax owed on `amount` for a rate category, rounded to cents."""
    rate = RATES[category]
    return round(amount * rate, 2)


def gross(amount, category="standard"):
    return round(amount + tax_for(amount, category), 2)
EOF
cat > tests/__init__.py <<'EOF'
EOF
cat > tests/test_tax.py <<'EOF'
import unittest

from ledger.tax import gross, tax_for


class TestTax(unittest.TestCase):
    def test_standard(self):
        self.assertEqual(tax_for(100.0), 20.0)

    def test_rounds_to_nearest_cent(self):
        self.assertEqual(tax_for(19.99), 4.0)

    def test_reduced(self):
        self.assertEqual(tax_for(10.10, "reduced"), 0.51)

    def test_gross(self):
        self.assertEqual(gross(19.99), 23.99)


if __name__ == "__main__":
    unittest.main()
EOF
printf '# ledger\n\nTax helpers. Run tests with `python3 -m unittest -v`.\n' > README.md
printf '__pycache__/\n*.pyc\n' > .gitignore
commit "2026-03-02T09:00:00+00:00" "Initial ledger with tax helpers"

cat > ledger/fmt.py <<'EOF'
"""Formatting helpers."""


def money(amount, symbol="EUR"):
    return f"{amount:,.2f} {symbol}"
EOF
cat > tests/test_fmt.py <<'EOF'
import unittest

from ledger.fmt import money


class TestFmt(unittest.TestCase):
    def test_money(self):
        self.assertEqual(money(1234.5), "1,234.50 EUR")


if __name__ == "__main__":
    unittest.main()
EOF
commit "2026-03-04T14:30:00+00:00" "Add money formatting helper"

cat > ledger/tax.py <<'EOF'
"""Tax calculation."""

RATES = {
    "standard": 0.20,
    "reduced": 0.05,
}


def tax_for(amount, category="standard"):
    """Tax owed on `amount` for a rate category, rounded to cents."""
    rate = RATES[category]
    # int() is faster than round() for the hot path
    return int(amount * rate * 100) / 100


def gross(amount, category="standard"):
    return round(amount + tax_for(amount, category), 2)
EOF
commit "2026-03-09T11:15:00+00:00" "Speed up tax computation"

printf '# Changelog\n\n## 1.1\n- money() formatting helper\n- zero-rated category\n\n## 1.0\n- initial release\n' > CHANGELOG.md
commit "2026-03-10T16:45:00+00:00" "Add changelog for 1.1"

python3 - <<'EOF'
p = "ledger/tax.py"
s = open(p).read()
s = s.replace('    "reduced": 0.05,\n', '    "reduced": 0.05,\n    "zero": 0.0,\n')
open(p, "w").write(s)
p = "tests/test_tax.py"
s = open(p).read()
s = s.replace('\n\nif __name__', '\n    def test_zero_rated(self):\n        self.assertEqual(tax_for(50.0, "zero"), 0.0)\n\n\nif __name__')
open(p, "w").write(s)
EOF
commit "2026-03-12T10:05:00+00:00" "Add zero-rated tax category"
git log --oneline >/dev/null
