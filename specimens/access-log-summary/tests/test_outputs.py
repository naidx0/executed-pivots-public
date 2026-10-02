"""Outcome checks for access-log-summary: recompute both reports from the logs and compare."""
import re
import sys
from collections import Counter

ok = True
LINE = re.compile(r'^(\S+) \S+ \S+ \[[^\]]*\] "(\S+) (\S+) [^"]*" (\d{3}) ')


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


reqs = []
for f in ("/var/log/webapp/access.log.1", "/var/log/webapp/access.log"):
    for ln in open(f):
        m = LINE.match(ln)
        if not m:
            continue
        ip, _method, target, status = m.groups()
        path = target.split("?", 1)[0]
        if path == "/healthz":
            continue
        reqs.append((ip, path, int(status)))

by_ip = Counter(ip for ip, _, _ in reqs)
classes = Counter(s // 100 for _, _, s in reqs)
top = sorted(by_ip.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
want = [f"total_requests={len(reqs)}", f"unique_ips={len(by_ip)}"]
want += [f"status_{c}xx={classes.get(c, 0)}" for c in (2, 3, 4, 5)]
want += ["top_ips:"] + [f"{ip} {n}" for ip, n in top]

try:
    got = [ln.rstrip() for ln in open("/app/reports/traffic_summary.txt").read().splitlines() if ln.strip()]
    got = [re.sub(r"\s+", " ", ln).strip() for ln in got]
    for i, w in enumerate(want):
        g = got[i] if i < len(got) else None
        check(f"summary line {i}: want {w!r} got {g!r}", g == w)
    check("summary has no extra lines", len(got) == len(want))
except OSError as e:
    check(f"traffic_summary.txt readable ({e})", False)

err = Counter(p for _, p, s in reqs if 500 <= s <= 599)
want_csv = ["path,count"] + [f"{p},{n}" for p, n in sorted(err.items(), key=lambda kv: (-kv[1], kv[0]))]
try:
    got_csv = [ln.strip().replace(", ", ",") for ln in open("/app/reports/errors_by_path.csv").read().splitlines() if ln.strip()]
    check(f"errors_by_path.csv\n want {want_csv}\n got  {got_csv}", got_csv == want_csv)
except OSError as e:
    check(f"errors_by_path.csv readable ({e})", False)

sys.exit(0 if ok else 1)
