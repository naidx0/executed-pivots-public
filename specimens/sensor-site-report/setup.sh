mkdir -p /app/pipeline /data/readings
cat > /app/pipeline/loader.py <<'EOF'
"""Load raw sensor dumps."""
import glob
import json
import os


def load_records(data_dir):
    """Return the parsed JSON records in data_dir, in filename order."""
    records = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(path) as fh:
            records.append(json.load(fh))
    return records
EOF
cat > /app/pipeline/report.py <<'EOF'
"""Site-level summary report."""
import csv

FIELDS = ["site", "sensors", "readings", "mean_value", "max_value"]


def build_site_summary(records):
    """Aggregate sensor records into one row per site.

    TODO: implement (see the task description for the rules).
    """
    raise NotImplementedError("build_site_summary is not implemented yet")


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for r in rows:
            w.writerow([r["site"], r["sensors"], r["readings"],
                        f"{r['mean_value']:.2f}", f"{r['max_value']:.2f}"])
EOF
cat > /app/pipeline/run.py <<'EOF'
"""Entry point: python3 run.py"""
import os

from loader import load_records
from report import build_site_summary, write_csv

DATA_DIR = os.environ.get("SENSOR_DATA", "/data/readings")
OUT_DIR = os.environ.get("REPORT_DIR", "out")


def main():
    records = load_records(DATA_DIR)
    rows = build_site_summary(records)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "site_summary.csv")
    write_csv(rows, path)
    print(f"wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
EOF
python3 - <<'EOF'
import json
sites = ["north", "south", "east", "west"]
offline = {4, 11}
for i in range(1, 15):
    rec = {"sensor_id": f"s-{i:03d}", "site": sites[(i * 3) % 4],
           "status": "offline" if i in offline else "ok", "readings": []}
    for j in range(6):
        v = None if (i * 5 + j) % 7 == 0 else round(18 + ((i * 7 + j * 13) % 50) / 5, 1)
        if i in offline and j == 2:
            v = 99.9  # stuck sensor
        rec["readings"].append({"t": j * 600, "value": v})
    with open(f"/data/readings/sensor_{i:03d}.json", "w") as fh:
        json.dump(rec, fh, indent=2)
        fh.write("\n")
EOF
