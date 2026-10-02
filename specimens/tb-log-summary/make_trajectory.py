"""Writes trajectory.json for tb-log-summary (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path

WRITE = ("{ echo 'severity,count'; for s in ERROR WARNING INFO; do echo \"$s,$(grep -h \"$s\" *.log | wc -l)\"; done; }"
         " > /app/summary.csv\n")


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. I need to count the ERROR, WARNING and INFO lines across every .log file in /app/logs and "
         "write the totals to /app/summary.csv. First I'll see which files are there and how big they are.",
         "List /app/logs and count the lines per file.",
         [("ls -la /app/logs\n", 0.1), ("wc -l /app/logs/*.log\n", 0.1)]),
    turn("There are three logs (app1.log, app2.log, db.log) with 5 lines each, 15 lines in total, so I can read them "
         "all before counting.",
         "Print the three files to see the line format and where the severity appears.",
         [("head -n 20 /app/logs/*.log\n", 0.1)]),
    turn("Every line is `date time LEVEL message`, and the levels used are INFO, WARNING and ERROR. No message text "
         "contains another level's word, so counting lines that contain each word gives the per-level totals.",
         "cd into /app/logs, count per file with grep -c, then the totals across all files.",
         [("cd /app/logs\n", 0.1), ("grep -c ERROR *.log\n", 0.1),
          ("for s in ERROR WARNING INFO; do echo \"$s $(grep -h \"$s\" *.log | wc -l)\"; done\n", 0.1)]),
    turn("The totals are ERROR 4 (2 in app1, 1 in app2, 1 in db), WARNING 3 and INFO 8, which adds up to the 15 lines.",
         "Write /app/summary.csv with the header and one row per severity, from the same grep counts.",
         [(WRITE, 0.1)]),
    turn("The CSV is written. I want to check its exact content and that it has exactly four lines and parses as CSV.",
         "cat the file, count its lines, and parse it with Python's csv module.",
         [("cat /app/summary.csv\n", 0.1), ("wc -l < /app/summary.csv\n", 0.1),
          ("python3 -c \"import csv; print(list(csv.reader(open('/app/summary.csv'))))\"\n", 0.5)]),
    turn("/app/summary.csv has the header severity,count and the rows ERROR,4, WARNING,3 and INFO,8: four lines, and it "
         "parses as valid CSV.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
