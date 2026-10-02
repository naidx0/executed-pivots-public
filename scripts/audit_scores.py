"""Turn an audit's rows.jsonl into per-pivot scores that arloop.py can compare.

python audit_scores.py out/audit/rows.jsonl --reward X > new.json
python arloop.py compare --base base.json --new new.json --paired

Each run is one pivot (task, turn). score = working student steps (E = 1) the reward paid there.
wrong = steps the reward paid where E = 0. The keep rule needs the gain above 2 SE and total wrong == 0.
"""
import argparse, json, collections

ap = argparse.ArgumentParser()
ap.add_argument("rows")
ap.add_argument("--reward", default="X", choices=["X", "J"])
a = ap.parse_args()
paid = collections.Counter(); wrong = collections.Counter(); keys = set()
for line in open(a.rows):
    r = json.loads(line)
    k = (r["task"], r["turn"]); keys.add(k)
    if r[a.reward] == 1 and r["E"] == 1 and r["action"].startswith("student"):
        paid[k] += 1
    if r[a.reward] == 1 and r["E"] == 0:
        wrong[k] += 1
runs = [{"seed": f"{t}:{n}", "score": paid[(t, n)], "wrong": wrong[(t, n)]} for t, n in sorted(keys)]
print(json.dumps({"reward": a.reward, "total_paid": sum(paid.values()), "total_wrong": sum(wrong.values()), "runs": runs}))
