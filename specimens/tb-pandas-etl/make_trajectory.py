"""Writes trajectory.json for tb-pandas-etl (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path

ETL = r"""cat > /app/etl.py <<'EOF'
import pandas as pd

df = pd.read_csv("/app/data.csv", dtype=str)

# address is "street\nCity, Region POSTAL"
df["postal_code"] = df["address"].str.extract(r"(\d+)\s*$", expand=False)
df["city"] = df["address"].str.extract(r"\n\s*([^,\n]+),", expand=False).str.strip()
df["birth_date"] = pd.to_datetime(df["birth_date"], format="%d %b %Y").dt.strftime("%Y-%m-%d")


def team(row):
    initials = row["first_name"][0].upper() + row["last_name"][0].upper()
    inst = "".join(word[0] for word in row["institute"].split())
    return f"{initials}-{row['country']}-{inst}"


df["team_name"] = df.apply(team, axis=1)
df.to_csv("/app/result.csv", index=False)
print(df[["participant_id", "birth_date", "postal_code", "city", "team_name"]].to_string(index=False))
EOF
"""
CHECK = ("python3 -c \"import pandas as pd; df = pd.read_csv('/app/result.csv', dtype=str); "
         "print(list(df.columns)); print(df.shape); print(df[['postal_code', 'city', 'birth_date']].to_dict('list'))\"\n")


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. I need to transform /app/data.csv into /app/result.csv with postal_code, city and team_name "
         "columns added and birth_date as YYYY-MM-DD. First I'll look at the raw file.",
         "Print the CSV and check whether pandas is available.",
         [("cat /app/data.csv\n", 0.1), ("python3 -c 'import pandas; print(pandas.__version__)'\n", 2.0)]),
    turn("The CSV has 5 participants. address is a quoted field that spans two lines: `street\\nCity, Region "
         "POSTAL`, for example `Gg. Monginsidi No. 08` / `Medan, Aceh 80734`. birth_date is like `05 Feb 1991`. "
         "pandas 2.3.0 is installed and handles the embedded newlines.",
         "Load it with pandas as strings and print the address reprs, the dates and the institutes to pin down the "
         "patterns.",
         [("python3 -c \"import pandas as pd; df = pd.read_csv('/app/data.csv', dtype=str); print(df.shape); "
           "[print(repr(a)) for a in df['address']]; print(df[['birth_date', 'country', 'institute']])\"\n", 2.0)]),
    turn("pandas reads 5 rows and 10 columns. Every address ends in a 5-digit postal code (one starts with 0: "
         "09434, so it must stay a string), and the city sits between the newline and the first comma. Countries "
         "can contain spaces (Korea Utara), and the institutes have 3 words each.",
         "Write /app/etl.py: read everything as strings so postal codes keep leading zeros, extract the postal code "
         "and the city with regexes, reformat birth_date from %d %b %Y, and build team_name as FL-Country-I.",
         [(ETL, 0.2)]),
    turn("The script is written.", "Run it.", [("cd /app\n", 0.1), ("python3 etl.py\n", 2.0)]),
    turn("etl.py printed the new columns: birth dates 1991-02-05, 1993-01-11, 1993-07-31, 1991-11-04, 2003-01-22; "
         "postal codes 80734, 09434, 57739, 76156, 70074; cities Medan, Prabumulih, Ambon, Lubuklinggau, Kediri; and "
         "team names like CN-Georgia-UPM and AS-Korea Utara-UD.",
         "Read result.csv back the way it will be checked (dtype=str) and confirm the columns and values.",
         [(CHECK, 2.0)]),
    turn("result.csv has exactly the 13 required columns in order and 5 rows, and read with dtype=str it keeps "
         "09434 with its leading zero, the city names and the ISO birth dates.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
