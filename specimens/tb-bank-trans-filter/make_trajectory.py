"""Writes trajectory.json for tb-bank-trans-filter (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path

FILTER = r"""cat > /app/filter_nwc.py <<'EOF'
import csv
import json
import re

SRC = "/app/data/bank_transactions.csv"
OUT = "/app/data/output.json"

# the company's names, typos included: "North West Capital ..." and its short form "NWCG ..."
NAME = re.compile(r"north\s*west\s*cap|nwcg", re.I)

with open(SRC, newline="") as f:
    rows = list(csv.DictReader(f))

accounts = {r["Account Number"] for r in rows if NAME.search(r["Company Name"])}
keep = [r for r in rows if NAME.search(r["Company Name"]) or r["Account Number"] in accounts]
keep.sort(key=lambda r: r["Date"])  # stable: same-day rows keep their file order

with open(OUT, "w") as f:
    json.dump(keep, f, indent=2)
print(len(keep), "rows; accounts:", sorted(accounts))
EOF
"""
CHECK = ("python3 -c \"import json; d = json.load(open('/app/data/output.json')); "
         "print(type(d).__name__, len(d)); print(list(d[0])); "
         "[print(r['Date'], r['Account Number'], r['Amount'], r['Company Name']) for r in d]\"\n")


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. I need every transaction of North West Capital from /app/data/bank_transactions.csv, matched by "
         "name or account number despite typos, written as a date-sorted JSON array to /app/data/output.json. First "
         "I'll look at the CSV.",
         "Show the header and first rows, and count the lines.",
         [("head -n 5 /app/data/bank_transactions.csv\n", 0.1), ("wc -l /app/data/bank_transactions.csv\n", 0.1)]),
    turn("The columns are Company Name, Account Number, Date, Amount, Transaction Type, Description; 500 transactions "
         "plus the header, with amounts like -$211608.00 and no quoted fields in these rows.",
         "cd to /app/data and search for the company by the words of its name and a likely abbreviation (NWCG).",
         [("cd /app/data\n", 0.1), ("grep -n -i -E 'north|west|nwcg' bank_transactions.csv\n", 0.1)]),
    turn("The company shows up as `North West Capital Group Incorporated`, a typo `North West Capital Group "
         "Incoporated`, and the short form `NWCG Inc. Pty. Ltd.`. Its main account is BUS-30001, but single rows "
         "use BUS-30g01, BUS-30999 and BUS-35001, which look like typos of it. Northern Light Energy and Westwind "
         "Trading Co only share a word and are different companies. I need to check whether any other company name uses these accounts.",
         "List every row whose account looks like BUS-30001 or its variants, and every company name that mentions "
         "Capital, to rule out other lookalikes.",
         [("grep -n -E ',BUS-3[0-9a-z]{4},' bank_transactions.csv\n", 0.1),
          ("cut -d, -f1 bank_transactions.csv | sort | uniq -c | grep -i capital\n", 0.1)]),
    turn("Only the 9 rows already found use a BUS-3xxxx account, so no other company shares them, and the only other "
         "Capital company is Momentum Capital. The rule is: the name matches North West Capital or NWCG (typos "
         "included), or the account is one used by such a row.",
         "Write /app/filter_nwc.py with csv.DictReader so values stay the exact strings, sort by Date, and dump "
         "JSON. Run it.",
         [(FILTER, 0.2), ("python3 /app/filter_nwc.py\n", 0.5)]),
    turn("The filter kept 9 rows, and the accounts seen with the company's names are BUS-30001, BUS-30999, BUS-30g01 "
         "and BUS-35001.",
         "Load output.json back and print its type, keys and rows to check the format and the order.",
         [(CHECK, 0.5)]),
    turn("output.json is a JSON list of 9 objects with exactly the six CSV column names as keys. Every value is the "
         "CSV string (for example -$197887.00), and the dates are ascending from 2024-01-22 to 2024-04-13.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
