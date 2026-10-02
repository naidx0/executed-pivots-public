mkdir -p /app/reportgen/reportgen /app/reportgen/bin /opt/wheelhouse /data
# the offline wheelhouse: small pure-Python wheels, written byte for byte (fixed zip timestamps)
python3 - <<'EOF'
import base64, hashlib, zipfile

PKGS = {
    ("tinyfmt", "1.2.0"): ({}, ">=3.8", '''"""tinyfmt 1.x: plain-text tables."""
__version__ = "1.2.0"


def table(rows, headers):
    """Left-aligned columns, two spaces apart, the header underlined with dashes."""
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()
    return "\\n".join([line([str(h) for h in headers]), line(["-" * w for w in widths])] + [line(r) for r in rows])
'''),
    ("tinyfmt", "2.0.0"): ({}, ">=3.8", '''"""tinyfmt 2.x: plain-text tables. 2.0 replaced table() with render()."""
__version__ = "2.0.0"


def render(rows, headers, style="plain"):
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = " | " if style == "grid" else "  "
    line = lambda cells: sep.join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()
    return "\\n".join([line([str(h) for h in headers])] + [line(r) for r in rows])
'''),
    ("decimalx", "2.0.0"): ({}, ">=3.8", '''"""decimalx 2.0: integer helpers."""
__version__ = "2.0.0"


def pad(n, width):
    return str(n).rjust(width, "0")
'''),
    ("decimalx", "2.1.0"): ({}, ">=3.8", '''"""decimalx 2.1: integer helpers; group() added in 2.1."""
__version__ = "2.1.0"


def pad(n, width):
    return str(n).rjust(width, "0")


def group(n):
    """1234567 -> '1,234,567'."""
    return f"{n:,}"
'''),
    ("decimalx", "3.0.0"): ({}, ">=3.8", '''"""decimalx 3.0: group() was renamed thousands()."""
__version__ = "3.0.0"


def thousands(n, sep=","):
    return f"{n:,}".replace(",", sep)
'''),
    ("moneyfmt", "0.9.0"): ({"decimalx": ">=2.0"}, ">=3.8", '''"""moneyfmt 0.9: money formatting."""
__version__ = "0.9.0"


def cents(n):
    sign = "-" if n < 0 else ""
    n = abs(int(n))
    return f"{sign}${n // 100}.{n % 100:02d}"
'''),
    ("moneyfmt", "1.0.0"): ({"decimalx": "<3,>=2.1"}, ">=3.8", '''"""moneyfmt 1.0: money formatting with thousands separators."""
import decimalx

__version__ = "1.0.0"


def cents(n):
    """12345678 -> '$123,456.78'."""
    sign = "-" if n < 0 else ""
    n = abs(int(n))
    return f"{sign}${decimalx.group(n // 100)}.{n % 100:02d}"
'''),
    ("datesy", "1.3.2"): ({}, ">=3.8", '''"""datesy: date keys."""
import datetime

__version__ = "1.3.2"


def month_key(s):
    """'2024-03-15' -> '2024-03' (the date must be valid)."""
    return datetime.date.fromisoformat(s.strip()).strftime("%Y-%m")
'''),
    ("datesy", "1.4.0"): ({}, ">=3.12", '''"""datesy: date keys (1.4 needs Python 3.12)."""
import datetime
from typing import override  # noqa: F401  (3.12+)

__version__ = "1.4.0"


def month_key(s):
    return datetime.date.fromisoformat(s.strip()).strftime("%Y-%m")
'''),
}


def digest(b):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(b).digest()).rstrip(b"=").decode()


for (name, ver), (deps, py, src) in PKGS.items():
    di = f"{name}-{ver}.dist-info"
    meta = f"Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\nSummary: {name} (in-house wheelhouse)\nRequires-Python: {py}\n"
    meta += "".join(f"Requires-Dist: {d}{spec}\n" for d, spec in deps.items())
    files = {f"{name}/__init__.py": src.encode(), f"{di}/METADATA": meta.encode(),
             f"{di}/WHEEL": b"Wheel-Version: 1.0\nGenerator: wheelhouse-build\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
             f"{di}/top_level.txt": f"{name}\n".encode()}
    record = "".join(f"{p},{digest(b)},{len(b)}\n" for p, b in files.items()) + f"{di}/RECORD,,\n"
    files[f"{di}/RECORD"] = record.encode()
    with zipfile.ZipFile(f"/opt/wheelhouse/{name}-{ver}-py3-none-any.whl", "w", zipfile.ZIP_DEFLATED) as z:
        for p, b in files.items():
            z.writestr(zipfile.ZipInfo(p, (1980, 1, 1, 0, 0, 0)), b)
EOF
cat > /app/reportgen/reportgen/__init__.py <<'EOF'
"""reportgen: monthly order summaries from an orders CSV (date, customer, amount_cents)."""
import argparse
import csv
import sys

import datesy
import moneyfmt
import tinyfmt

HEADERS = ["month", "orders", "revenue"]


def summarize(path):
    months = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            m = months.setdefault(datesy.month_key(row["date"]), [0, 0])
            m[0] += 1
            m[1] += int(row["amount_cents"])
    return [[k, str(n), moneyfmt.cents(c)] for k, (n, c) in sorted(months.items())]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="reportgen", description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("-o", "--output", help="write the report here instead of stdout")
    a = ap.parse_args(argv)
    text = tinyfmt.table(summarize(a.csv), HEADERS) + "\n"
    if a.output:
        with open(a.output, "w") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0
EOF
cat > /app/reportgen/bin/reportgen <<'EOF'
#!/app/venv/bin/python
"""Launcher: runs reportgen from /app/reportgen with the /app/venv interpreter."""
import sys

sys.path.insert(0, "/app/reportgen")
from reportgen import main  # noqa: E402

sys.exit(main())
EOF
chmod +x /app/reportgen/bin/reportgen
cat > /app/reportgen/requirements.txt <<'EOF'
# reportgen runtime dependencies, installed from the local wheelhouse
tinyfmt>=1.0
moneyfmt==1.0.0
datesy==1.4.0
EOF
cat > /app/reportgen/constraints.txt <<'EOF'
# org-wide version pins (pip -c)
decimalx==2.0.0
EOF
cat > /app/reportgen/README.md <<'EOF'
# reportgen

Monthly order summary: `bin/reportgen ORDERS.csv [-o OUT]` prints one row per month (orders, revenue).
The CSV has the columns date (YYYY-MM-DD), customer and amount_cents.

Runtime: Python 3.11 in /app/venv. Dependencies (see requirements.txt): tinyfmt 1.x (`tinyfmt.table`),
moneyfmt 1.0 (thousands separators) and datesy. The machine is offline; wheels live in /opt/wheelhouse.

    /app/venv/bin/pip install --no-index --find-links /opt/wheelhouse -r requirements.txt -c constraints.txt
EOF
cat > /data/orders.csv <<'EOF'
date,customer,amount_cents
2024-01-03,acme,125000
2024-01-17,globex,4999
2024-01-29,initech,98001
2024-02-02,acme,250050
2024-02-14,umbrella,1999
2024-03-01,globex,1000000
2024-03-09,acme,75
2024-03-30,initech,33333
EOF
python3 -m venv /app/venv
