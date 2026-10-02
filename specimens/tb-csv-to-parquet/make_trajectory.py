"""Writes trajectory.json for tb-csv-to-parquet (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path

CONVERT = r"""cat > /app/convert.py <<'EOF'
import pandas as pd

df = pd.read_csv("/app/data.csv")
df.to_parquet("/app/data.parquet", index=False)
print(df.dtypes.to_dict(), len(df))
EOF
"""
CHECK = ("python3 -c \"import pandas as pd; a = pd.read_csv('/app/data.csv'); b = pd.read_parquet('/app/data.parquet'); "
         "pd.testing.assert_frame_equal(a, b); print('equal'); print(b)\"\n")


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


traj = [
    turn("I'm at /app. I need to convert /app/data.csv into a Parquet file at /app/data.parquet. First I'll look at "
         "the CSV.",
         "List /app and print the CSV.", [("ls -la /app\n", 0.1), ("cat /app/data.csv\n", 0.1)]),
    turn("data.csv has a header name,age,city and 5 rows (John, Alice, Bob, Emma, David); city values contain spaces "
         "(New York, San Francisco) but no quotes, and the file has no final newline. Writing Parquet from Python needs pandas plus a Parquet engine "
         "(pyarrow or fastparquet).",
         "Check which of pandas, pyarrow and fastparquet are installed.",
         [("python3 -c 'import pandas, pyarrow; print(pandas.__version__, pyarrow.__version__)'\n", 2.0),
          ("python3 -c 'import fastparquet' 2>&1 | tail -n 1\n", 1.0)]),
    turn("pandas 2.3.0 and pyarrow 20.0.0 are installed (fastparquet is not, which is fine: pandas uses pyarrow).",
         "Write /app/convert.py: read the CSV with pandas and write it with to_parquet, without the index.",
         [(CONVERT, 0.2)]),
    turn("convert.py is in place.", "Run it from /app and list the output.",
         [("cd /app\n", 0.1), ("python3 convert.py\n", 2.0), ("ls -la data.parquet\n", 0.1)]),
    turn("convert.py read 5 rows with dtypes name object, age int64, city object, and data.parquet now exists.",
         "Read the Parquet file back and compare it to the CSV with pandas' own frame equality check.",
         [(CHECK, 2.0)]),
    turn("assert_frame_equal passed ('equal'): the Parquet file has the same 5 rows, columns and dtypes as the CSV.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
