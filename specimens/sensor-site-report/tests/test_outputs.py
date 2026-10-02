"""Outcome checks for sensor-site-report: the CSV at the stated path and the function itself."""
import csv
import glob
import json
import re
import sys

ok = True
OUT = "/app/reports/site_summary.csv"


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


def expected(records):
    acc = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        a = acc.setdefault(r["site"], {"sensors": 0, "vals": []})
        a["sensors"] += 1
        a["vals"] += [x["value"] for x in r["readings"] if x["value"] is not None]
    return [{"site": s, "sensors": a["sensors"], "readings": len(a["vals"]),
             "mean_value": sum(a["vals"]) / len(a["vals"]), "max_value": max(a["vals"])}
            for s, a in sorted(acc.items())]


records = []
for p in sorted(glob.glob("/data/readings/*.json")):
    with open(p) as fh:
        records.append(json.load(fh))
exp = expected(records)

# 1) the report file
try:
    with open(OUT, newline="") as fh:
        text = fh.read()
    rows = list(csv.reader(text.splitlines()))
    check("header", rows and rows[0] == ["site", "sensors", "readings", "mean_value", "max_value"])
    body = [r for r in rows[1:] if r]
    check(f"row count {len(body)} == {len(exp)}", len(body) == len(exp))
    for got, e in zip(body, exp):
        good = (len(got) == 5 and got[0] == e["site"] and got[1] == str(e["sensors"]) and got[2] == str(e["readings"])
                and all(re.fullmatch(r"-?\d+\.\d\d", x) for x in got[3:5])
                and abs(float(got[3]) - e["mean_value"]) <= 0.0051 and abs(float(got[4]) - e["max_value"]) <= 0.0051)
        check(f"row {e['site']}: {got}", good)
except FileNotFoundError:
    check(f"{OUT} exists", False)

# 2) the function on a hidden input
sys.path.insert(0, "/app/pipeline")
try:
    from report import build_site_summary
    hidden = [
        {"sensor_id": "a", "site": "zeta", "status": "ok", "readings": [{"t": 0, "value": 1.0}, {"t": 1, "value": None}, {"t": 2, "value": 4.0}]},
        {"sensor_id": "b", "site": "alpha", "status": "ok", "readings": [{"t": 0, "value": 2.5}]},
        {"sensor_id": "c", "site": "zeta", "status": "maintenance", "readings": [{"t": 0, "value": 100.0}]},
        {"sensor_id": "d", "site": "zeta", "status": "ok", "readings": [{"t": 0, "value": 7.0}]},
        {"sensor_id": "e", "site": "gamma", "status": "offline", "readings": [{"t": 0, "value": 9.0}]},
    ]
    got = build_site_summary(hidden)
    want = expected(hidden)
    check("function returns a list", isinstance(got, list) and len(got) == len(want))
    for g, e in zip(got, want):
        check(f"function row {e['site']}",
              g["site"] == e["site"] and int(g["sensors"]) == e["sensors"] and int(g["readings"]) == e["readings"]
              and abs(float(g["mean_value"]) - e["mean_value"]) < 1e-6 and abs(float(g["max_value"]) - e["max_value"]) < 1e-6)
except Exception as e:  # noqa: BLE001
    check(f"build_site_summary on hidden input raised {e!r}", False)

sys.exit(0 if ok else 1)
