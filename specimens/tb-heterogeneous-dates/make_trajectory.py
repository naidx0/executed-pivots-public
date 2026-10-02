"""Writes trajectory.json for tb-heterogeneous-dates (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path

SCRIPT = r"""cat > /app/process_data.py <<'EOF'
import pandas as pd

high = pd.read_csv("/app/daily_temp_sf_high.csv")
low = pd.read_csv("/app/daily_temp_sf_low.csv")

# high: YYYY-MM-DD; low: MM/DD/YYYY or MM-DD-YYYY with a 06:00:00 time
high["date"] = pd.to_datetime(high["date"], format="%Y-%m-%d")
low["date"] = pd.to_datetime(low["date"].astype(str).str.replace("/", "-"),
                             format="%m-%d-%Y %H:%M:%S").dt.normalize()

merged = high.rename(columns={"temperature": "temp_high"}).merge(
    low.rename(columns={"temperature": "temp_low"}), on="date", how="inner")
merged["difference"] = merged["temp_high"] - merged["temp_low"]
print(merged.sort_values("date").to_string(index=False))

with open("/app/avg_temp.txt", "w") as f:
    f.write(str(merged["difference"].mean()))
EOF
"""


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. I need the average daily high-minus-low temperature from the two San Francisco CSVs, written "
         "as a bare number to /app/avg_temp.txt. First I'll look at both files.",
         "List /app and print both CSVs with cat -A so line endings and odd characters show.",
         [("ls -la /app\n", 0.1), ("cat -A /app/daily_temp_sf_high.csv; echo\n", 0.1),
          ("cat -A /app/daily_temp_sf_low.csv; echo\n", 0.1)]),
    turn("Both files have CRLF line endings (^M$) and no newline after the last row. The high file uses ISO dates "
         "(2025-04-19 .. 2025-04-25). The low file mixes MM/DD/YYYY and MM-DD-YYYY, all with a 06:00:00 time, and its "
         "rows are not in date order. So the two files have to be joined on a normalized date, not by row position.",
         "Check that pandas is available to parse and join the dates.",
         [("python3 -c 'import pandas; print(pandas.__version__)'\n", 2.0)]),
    turn("pandas 2.3.0 is installed.",
         "Write /app/process_data.py: parse the high dates as ISO, turn the low dates' slashes into dashes and parse "
         "them as %m-%d-%Y %H:%M:%S normalized to midnight, inner-join on the date, and write the mean difference.",
         [(SCRIPT, 0.2)]),
    turn("The script is in place.", "Run it and show the result file.",
         [("cd /app\n", 0.1), ("python3 process_data.py\n", 2.0), ("cat avg_temp.txt; echo\n", 0.1)]),
    turn("All 7 days from 2025-04-19 to 2025-04-25 matched, and the differences (16, 11, 22, 18, 5, 3, 5) average "
         "11.428571428571429, which is what avg_temp.txt holds.",
         "Cross-check the mean by hand from the joined differences, independent of pandas.",
         [("python3 -c 'd = [16, 11, 22, 18, 5, 3, 5]; print(sum(d), len(d), sum(d) / len(d))'\n", 0.5)]),
    turn("80 / 7 = 11.428571428571429 agrees with the file, and /app/avg_temp.txt contains only that number.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
