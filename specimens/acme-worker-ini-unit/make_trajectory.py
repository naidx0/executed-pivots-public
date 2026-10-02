"""Writes trajectory.json for acme-worker-ini-unit (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


UNIT = "/etc/systemd/system/acme-worker.service"
CHECK = "python3 unitcheck.py " + UNIT + "\n"
WCHECK = "python3 worker.py --config /etc/acme/worker.ini --check; echo rc=$?\n"

traj = [
    turn("I'm at /app. The acme worker service has to be repaired: its systemd unit, its INI config, and then one "
         "batch run. There is no systemd here, so first the files and the helper tools.",
         "Read the unit and the config, and list the worker directory and the queue.",
         [("cat " + UNIT + " /etc/acme/worker.ini\n", 0.1), ("ls -la /app/worker /srv/queue/inbox\n", 0.1)]),
    turn("The unit runs `/usr/bin/python /app/worker/worker.py --config /etc/acme/worker.conf --once` as a oneshot in "
         "/app/worker, WantedBy=multi-user.target. The config is /etc/acme/worker.ini (not .conf); it sets batch_size "
         "twice (10, then 25 'raised for the spring backlog'), a dsn with the password s3cr%t, and level = info. "
         "/app/worker has worker.py and unitcheck.py; the inbox holds job-1.job to job-4.job.",
         "Run unitcheck.py on the unit to get its list of problems.",
         [("cd /app/worker\n", 0.1), (CHECK, 1.0)]),
    turn("unitcheck reports: ExecStart executable /usr/bin/python does not exist, config file /etc/acme/worker.conf "
         "does not exist, and enabled: no.",
         "Find the real python3, fix both paths in ExecStart, and rerun unitcheck.",
         [("command -v python3; ls -l /usr/bin/python* 2>&1 | head -n 3\n", 0.1),
          ("sed -i 's#^ExecStart=/usr/bin/python /app/worker/worker.py --config /etc/acme/worker.conf --once$#"
           "ExecStart=/usr/local/bin/python3 /app/worker/worker.py --config /etc/acme/worker.ini --once#' " + UNIT +
           " && grep -n '^ExecStart' " + UNIT + "\n", 0.1), (CHECK, 1.0)]),
    turn("python3 is /usr/local/bin/python3 (there is no /usr/bin/python). With ExecStart fixed, unitcheck runs the "
         "config check, which fails: DuplicateOptionError, option 'batch_size' in section 'queue' already exists "
         "(line 7). The worker's configparser is strict; the intended value is the later 25.",
         "Delete the stale `batch_size = 10` line and run the worker's --check directly.",
         [("sed -i '/^batch_size = 10$/d' /etc/acme/worker.ini && grep -n 'batch_size' /etc/acme/worker.ini\n", 0.1),
          (WCHECK, 1.0)]),
    turn("batch_size is only 25 now, and the next error is an InterpolationSyntaxError: '%' must be followed by '%' "
         "or '(', found '%t@db.internal:5432/acme'. The password s3cr%t has a literal %, which configparser's "
         "interpolation needs escaped as %%.",
         "Escape it as s3cr%%t (the worker then reads s3cr%t) and check again.",
         [("sed -i 's/s3cr%t@/s3cr%%t@/' /etc/acme/worker.ini && grep -n '^dsn' /etc/acme/worker.ini\n", 0.1),
          (WCHECK, 1.0)]),
    turn("The dsn parses now; the check fails on ValueError: log level 'info' is not one of DEBUG, INFO, WARNING, "
         "ERROR. The worker compares the level case-sensitively.",
         "Set level = INFO, check the config, and rerun unitcheck on the unit.",
         [("sed -i 's/^level = info$/level = INFO/' /etc/acme/worker.ini\n", 0.1), (WCHECK, 1.0), (CHECK, 1.0)]),
    turn("worker.py --check prints config OK: batch_size=25 level=INFO db=db.internal:5432/acme, and unitcheck says "
         "unit OK, but enabled: no. `systemctl enable` would create the WantedBy symlink "
         "multi-user.target.wants/acme-worker.service pointing at the unit.",
         "Create that symlink and confirm unitcheck sees the unit as enabled.",
         [("mkdir -p /etc/systemd/system/multi-user.target.wants\n", 0.1),
          ("ln -sf " + UNIT + " /etc/systemd/system/multi-user.target.wants/acme-worker.service\n", 0.1),
          (CHECK, 1.0)]),
    turn("unitcheck: unit OK and enabled: yes. The four queued jobs still have to be processed once.",
         "Run the unit's ExecStart the way systemd would (WorkingDirectory /app/worker, a bare environment), then "
         "look at the queue and the report.",
         [("cd /app/worker && env -i PATH=/usr/bin:/bin /usr/local/bin/python3 /app/worker/worker.py --config "
           "/etc/acme/worker.ini --once\n", 1.0), ("ls /srv/queue/inbox /srv/queue/done\n", 0.1),
          ("cat /srv/queue/report.txt\n", 0.1)]),
    turn("processed 4 jobs (db db.internal:5432/acme): the inbox is empty, done holds job-1.job to job-4.job, and "
         "report.txt has J001 ada 42, J002 brook 5, J003 chen 21, J004 dana 351. The unit is fixed and enabled, the "
         "config keeps batch_size 25, the password and level INFO, and the backlog was processed once.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
