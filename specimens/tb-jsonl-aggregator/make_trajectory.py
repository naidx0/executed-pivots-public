"""Writes trajectory.json for tb-jsonl-aggregator (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path

PROFILE = r"""python3 - <<'EOF'
import glob, json
from collections import Counter
keys, types, n = Counter(), Counter(), 0
for p in sorted(glob.glob('/app/records_*.jsonl')):
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line); n += 1
        keys[tuple(sorted(r))] += 1
        types.update(f"{k}:{type(v).__name__}" for k, v in r.items())
print(n, 'records'); print(keys); print(sorted(types.items()))
EOF
"""
AGG = r"""cat > /app/aggregate.py <<'EOF'
import glob
import json
from collections import Counter, defaultdict

amount = defaultdict(float)
items = defaultdict(int)
tags = Counter()

for path in sorted(glob.glob("/app/records_*.jsonl")):
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            user = rec["user"]
            amount[user] += float(rec.get("amount", 0))
            items[user] += int(rec.get("items", 0))
            tags.update(rec.get("tags", []))

# ties broken by name so the output is deterministic
top_users = sorted(amount, key=lambda u: (-amount[u], u))[:5]
top_tags = sorted(tags, key=lambda t: (-tags[t], t))[:5]
out = {
    "top_5_users_by_amount": {u: {"total_amount": round(amount[u], 2), "total_items": items[u]} for u in top_users},
    "top_5_tags_by_count": {t: {"count": tags[t]} for t in top_tags},
}
with open("/app/aggregates.json", "w") as f:
    json.dump(out, f, indent=2)
print(len(amount), "users,", len(tags), "tags")
for u in sorted(amount, key=lambda u: (-amount[u], u))[:7]:
    print("user", u, round(amount[u], 2), items[u])
for t in sorted(tags, key=lambda t: (-tags[t], t))[:7]:
    print("tag", t, tags[t])
EOF
"""


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. I need to aggregate all /app/records_*.jsonl files into /app/aggregates.json: the top 5 users "
         "by summed amount (with their total items) and the top 5 tags by count. First I'll see the files and the "
         "record shape.",
         "List /app, count the lines per file, and show a few records.",
         [("ls -la /app\n", 0.1), ("wc -l /app/records_*.jsonl\n", 1.0), ("head -n 3 /app/records_1.jsonl\n", 0.1)]),
    turn("There are five files, records_1.jsonl to records_5.jsonl, with 200000 lines each (1,000,000 records). "
         "Each record looks like {\"user\": ..., \"amount\": <float>, \"items\": <int>, \"tags\": [...]}, and tags "
         "can be an empty list. Before aggregating I want to be sure every record has the same keys and types.",
         "Profile all records with a short Python pass: key sets and value types.",
         [(PROFILE, 15.0)]),
    turn("All 1,000,000 records have exactly the keys amount, items, tags and user; amount is always a float, "
         "items an int, tags a list and user a str. Nothing needs cleaning.",
         "Write /app/aggregate.py: sum amount and items per user, count tags across all tag lists, take the top 5 of "
         "each with ties broken by name, round amounts to 2 decimals, and print the top 7 of each so I can see the "
         "margin at 5th place.",
         [(AGG, 0.2)]),
    turn("aggregate.py is written.", "Run it.", [("cd /app\n", 0.1), ("python3 aggregate.py\n", 15.0)]),
    turn("It saw 500 users and 74 tags. The top users by amount are user211 (2131992.47, 149035 items), "
         "user220_dev, chloe_test, user260_123 and user271 (2034668.79); the top tags are lime (28874), type_a, "
         "reporting, frontend and approved (28665). Neither list has a tie at 5th place: user494_99 "
         "(2024564.25) and the tag web (28657) come next.",
         "Check that aggregates.json is valid JSON with the required structure and types.",
         [("python3 -m json.tool /app/aggregates.json\n", 0.5)]),
    turn("aggregates.json parses and has exactly the two required keys. top_5_users_by_amount holds 5 users with "
         "total_amount rounded to 2 decimals and integer total_items; top_5_tags_by_count holds 5 tags with integer "
         "counts.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
