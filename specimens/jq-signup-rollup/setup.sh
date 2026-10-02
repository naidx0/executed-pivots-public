mkdir -p /data/events /app/etl
cat > /tmp/gen.py <<'GENEOF'
"""Deterministic event generator for jq-signup-rollup (setup.sh has a copy; the verifier uses it for hidden data)."""
import gzip
import json
import os

PLANS = ["free", "pro", "team", "enterprise"]
TYPES = ["signup", "login", "login", "upgrade", "signup", "login"]


def make(seed, days, per_day, dup_every):
    state = seed

    def rnd(n):
        nonlocal state
        state = (state * 1103515245 + 12345) % 2**31
        return (state >> 8) % n

    out = {}
    eid = seed % 9000 + 1000
    prev = []
    for d, day in enumerate(days):
        v2 = d > 0  # the first day is still on schema v1
        evs = []
        for i in range(per_day):
            eid += 1
            uid = 100 + rnd(40)
            plan = PLANS[min(rnd(10), 9) // 3] if rnd(5) else "enterprise"
            typ = TYPES[rnd(len(TYPES))]
            ts = f"{day}T{8 + i // 6:02d}:{(i * 7) % 60:02d}:00Z"
            if v2:
                e = {"v": 2, "event_id": f"ev-{eid}", "type": typ, "ts": ts, "user": {"id": uid}, "account": {"plan": plan}}
            else:
                e = {"event_id": f"ev-{eid}", "type": typ, "ts": ts, "user_id": uid, "plan": plan}
            evs.append(e)
            if i % dup_every == dup_every - 1 and prev:  # a retried delivery of an earlier event
                evs.append(prev[rnd(len(prev))])
        prev = [e for e in evs if e["type"] == "signup"][-5:] or prev
        out[day] = evs
    return out


def write(dirpath, seed, days, per_day, dup_every):
    os.makedirs(dirpath, exist_ok=True)
    evs = make(seed, days, per_day, dup_every)
    for k, day in enumerate(days):
        body = "".join(json.dumps(e, separators=(",", ":")) + "\n" for e in evs[day])
        name = os.path.join(dirpath, f"events-{day}.jsonl")
        if k == 0 and len(days) > 2:  # the oldest rotated file is compressed
            with gzip.GzipFile(name + ".gz", "wb", mtime=0) as fh:
                fh.write(body.encode())
        else:
            open(name, "w").write(body)
    return evs


def expected(evs):
    seen, n, users = set(), {}, {}
    for day in evs:
        for e in evs[day]:
            if e["type"] != "signup" or e["event_id"] in seen:
                continue
            seen.add(e["event_id"])
            plan = e["account"]["plan"] if "account" in e else e["plan"]
            uid = e["user"]["id"] if "user" in e else e["user_id"]
            n[plan] = n.get(plan, 0) + 1
            users.setdefault(plan, set()).add(uid)
    rows = sorted(n, key=lambda p: (-n[p], p))
    return ["plan,signups,unique_users"] + [f"{p},{n[p]},{len(users[p])}" for p in rows]


if __name__ == "__main__":
    write("/data/events", 424242, ["2026-03-09", "2026-03-10", "2026-03-11"], 42, 9)
GENEOF
python3 /tmp/gen.py && rm /tmp/gen.py
cat > /app/etl/rollup.sh <<'EOF'
#!/bin/bash
# Daily signup rollup for the growth dashboard.
# Usage: rollup.sh [EVENTS_DIR] [OUT_CSV]
set -euo pipefail
dir=${1:-/data/events}
out=${2:-/app/out/signups_by_plan.csv}
mkdir -p "$(dirname "$out")"
{
  echo "plan,signups,unique_users"
  cat "$dir"/*.jsonl \
    | jq -r 'select(.type == "signup") | [.plan, .user_id] | @tsv' \
    | awk -F'\t' '{ n[$1]++; if (!seen[$1 FS $2]++) u[$1]++ } END { for (p in n) print p "," n[p] "," u[p] }' \
    | sort -t, -k2,2nr
} > "$out"
EOF
chmod +x /app/etl/rollup.sh
cat > /app/etl/README.md <<'EOF'
# etl

`rollup.sh [EVENTS_DIR] [OUT_CSV]` turns the product event stream into `signups_by_plan.csv` for the growth
dashboard. Defaults: /data/events and /app/out/signups_by_plan.csv.

Event files are JSON lines, one file per day (`events-YYYY-MM-DD.jsonl`). Logrotate compresses older days to
`events-YYYY-MM-DD.jsonl.gz`. The producer delivers at least once: a retried delivery repeats an event with the
same `event_id`.

Schema v1 (no "v" field): {"event_id", "type", "ts", "user_id", "plan"}
Schema v2 ("v": 2):       {"v", "event_id", "type", "ts", "user": {"id"}, "account": {"plan"}}
EOF
