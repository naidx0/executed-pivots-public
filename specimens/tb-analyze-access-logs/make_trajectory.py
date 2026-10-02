"""Writes trajectory.json for tb-analyze-access-logs (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path

REPORT = r"""{
  echo "Total requests: $(wc -l < access_log)"
  echo "Unique IP addresses: $(awk '{print $1}' access_log | sort -u | wc -l)"
  echo "Top 3 URLs:"
  awk '{print $7}' access_log | sort | uniq -c | sort -k1,1nr -k2,2 | head -n 3 | awk '{printf "  %s: %d\n", $2, $1}'
  echo "404 errors: $(awk '$9 == "404"' access_log | wc -l)"
} > report.txt
"""


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. I need a summary of /app/access_log in /app/report.txt: total requests, unique IPs, the top 3 "
         "URLs with counts, and the number of 404s. First I'll look at the log format and its size.",
         "Show the first lines of the log and count its lines.",
         [("head -n 5 /app/access_log\n", 0.1), ("wc -l /app/access_log\n", 0.1)]),
    turn("The log is in Common Log Format: `IP - - [time] \"METHOD URL HTTP/1.1\" STATUS BYTES`, 2000 lines. With "
         "whitespace splitting, $1 is the IP, $7 the URL and $9 the status, if every line has the same number of "
         "fields.",
         "cd to /app, check the field count on every line, and look at the status distribution in $9.",
         [("cd /app\n", 0.1), ("awk '{print NF}' access_log | sort | uniq -c\n", 0.2),
          ("awk '{print $9}' access_log | sort | uniq -c\n", 0.2)]),
    turn("All 2000 lines have 10 fields, so the positions hold everywhere. $9 is always a status code; 404 appears 83 "
         "times.",
         "Count the unique IPs and list the most requested URLs, taking 5 to see whether the third place is tied.",
         [("awk '{print $1}' access_log | sort -u | wc -l\n", 0.2),
          ("awk '{print $7}' access_log | sort | uniq -c | sort -nr | head -n 5\n", 0.2)]),
    turn("There are 273 unique IPs. The top URLs are /order-confirmation (54), /product/456 (53) and /about.html (52); "
         "the next one, /news, has 51, so there is no tie for third place.",
         "Write report.txt in the required format, with the URL lines indented by two spaces.",
         [(REPORT, 0.5)]),
    turn("The report is written.", "Print it to check the format and the numbers.",
         [("cat -A report.txt\n", 0.1)]),
    turn("report.txt has `Total requests: 2000`, `Unique IP addresses: 273`, the `Top 3 URLs:` header with "
         "`  /order-confirmation: 54`, `  /product/456: 53` and `  /about.html: 52`, and `404 errors: 83`, with no "
         "trailing spaces.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
