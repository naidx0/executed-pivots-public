"""Writes trajectory.json for reportgen-pip-constraints (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


PIP = "/app/venv/bin/pip install --no-index --find-links /opt/wheelhouse -r requirements.txt -c constraints.txt"
META = ("/app/venv/bin/python -c \"import sys, zipfile; [print(zipfile.ZipFile(w).read(w.split('/')[-1].rsplit('-', 3)[0]"
        " + '.dist-info/METADATA').decode()) for w in sys.argv[1:]]\" ")

traj = [
    turn("I'm at /app. reportgen's venv at /app/venv has no dependencies installed. I need to read the requirement "
         "and constraint files and see which wheels the offline wheelhouse has.",
         "cd into the project, read requirements.txt, constraints.txt and the README, and list /opt/wheelhouse.",
         [("cd /app/reportgen\n", 0.1), ("cat requirements.txt constraints.txt README.md\n", 0.1),
          ("ls /opt/wheelhouse\n", 0.1)]),
    turn("requirements.txt asks for tinyfmt>=1.0, moneyfmt==1.0.0 and datesy==1.4.0; constraints.txt pins "
         "decimalx==2.0.0. The README says the code needs tinyfmt 1.x (tinyfmt.table). The wheelhouse has tinyfmt 1.2.0 "
         "and 2.0.0, decimalx 2.0.0, 2.1.0 and 3.0.0, moneyfmt 0.9.0 and 1.0.0, and datesy 1.3.2 and 1.4.0.",
         "Run the install command from the task as is to see what fails.",
         [(PIP + " 2>&1 | tail -n 12\n", 15.0)]),
    turn("The resolver fails: moneyfmt 1.0.0 depends on decimalx<3 and >=2.1, but the constraint pins decimalx==2.0.0. "
         "moneyfmt==1.0.0 is the required version (the README wants its thousands separators), so the constraint is "
         "the thing that is wrong.",
         "Confirm the dependency in the moneyfmt 1.0.0 wheel metadata before changing the pin.",
         [(META + "/opt/wheelhouse/moneyfmt-1.0.0-py3-none-any.whl | grep -E '^(Version|Requires)'\n", 1.0)]),
    turn("The moneyfmt 1.0.0 metadata says Requires-Dist: decimalx<3,>=2.1. decimalx 2.1.0 is in the wheelhouse and "
         "satisfies it.",
         "Pin decimalx==2.1.0 in constraints.txt and rerun the install.",
         [("sed -i 's/^decimalx==2.0.0$/decimalx==2.1.0/' constraints.txt\n", 0.1),
          (PIP + " 2>&1 | tail -n 3\n", 15.0)]),
    turn("The decimalx conflict is gone, but now the install stops on datesy: 'Package datesy requires a different "
         "Python: 3.11.16 not in >=3.12'. requirements.txt pins datesy==1.4.0.",
         "Check the Requires-Python of both datesy wheels.",
         [(META + "/opt/wheelhouse/datesy-*.whl | grep -E '^(Version|Requires-Python)'\n", 1.0)]),
    turn("datesy 1.4.0 requires Python >=3.12, and 1.3.2 requires >=3.8. This venv is Python 3.11 and has to stay "
         "on it, so the pin must be 1.3.2.",
         "Pin datesy==1.3.2 in requirements.txt, rerun the install, and try the tool.",
         [("sed -i 's/^datesy==1.4.0$/datesy==1.3.2/' requirements.txt\n", 0.1),
          (PIP + " 2>&1 | tail -n 2\n", 15.0), ("bin/reportgen /data/orders.csv\n", 1.0)]),
    turn("The install succeeds (datesy-1.3.2 decimalx-2.1.0 moneyfmt-1.0.0 tinyfmt-2.0.0), but the tool fails with "
         "AttributeError: module 'tinyfmt' has no attribute 'table'. tinyfmt>=1.0 let pip pick 2.0.0, which dropped "
         "table(); the code needs tinyfmt 1.x.",
         "Cap tinyfmt below 2 in requirements.txt, reinstall (pip downgrades it to 1.2.0), and rerun the tool.",
         [("sed -i 's/^tinyfmt>=1.0$/tinyfmt>=1.0,<2/' requirements.txt\n", 0.1),
          (PIP + " 2>&1 | tail -n 2\n", 15.0), ("bin/reportgen /data/orders.csv\n", 1.0)]),
    turn("tinyfmt is 1.2.0 now and the report prints: 2024-01 3 orders $2,280.00, 2024-02 2 $2,520.49, 2024-03 3 "
         "$10,334.08.",
         "Check the venv has no broken requirement, then write the report to /app/out/monthly.txt.",
         [("/app/venv/bin/pip check\n", 2.0), ("mkdir -p /app/out\n", 0.1),
          ("bin/reportgen /data/orders.csv -o /app/out/monthly.txt && cat /app/out/monthly.txt\n", 1.0)]),
    turn("pip check finds no broken requirements, and /app/out/monthly.txt holds the three-month report.",
         "Record the exact installed versions in requirements.lock with pip freeze.",
         [("/app/venv/bin/pip freeze > requirements.lock\n", 2.0), ("cat requirements.lock\n", 0.1)]),
    turn("requirements.lock pins datesy==1.3.2, decimalx==2.1.0, moneyfmt==1.0.0 and tinyfmt==1.2.0. The task also "
         "wants the fixed files to work in a fresh venv.",
         "Create a scratch venv in /tmp, run the same install there, and compare its freeze with the lock file.",
         [("python3 -m venv /tmp/fresh-venv\n", 10.0),
          ("/tmp/fresh-venv/bin/pip install -q --no-index --find-links /opt/wheelhouse -r requirements.txt -c "
           "constraints.txt && /tmp/fresh-venv/bin/pip freeze | diff - requirements.lock && echo same\n", 20.0)]),
    turn("The fresh venv installs the same four versions (diff is empty, `same`). The requirements and constraints "
         "are fixed, /app/venv is complete, the report and the lock file are written.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
