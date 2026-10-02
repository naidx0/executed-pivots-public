"""Outcome checks for acme-worker-ini-unit: worker.py and unitcheck.py are untouched; worker.ini parses under the
worker's own rules with the intended values (batch_size 25, the password s3cr%t, level INFO, paths unchanged); the
unit's ExecStart runs the worker with that config (checked by running it on a hidden queue with 30 jobs); the unit is
enabled like `systemctl enable`; the 4 queued jobs were processed exactly once."""
import configparser
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys

ok = True
UNIT = "/etc/systemd/system/acme-worker.service"
INI = "/etc/acme/worker.ini"
LINK = "/etc/systemd/system/multi-user.target.wants/acme-worker.service"
JOBS = {"job-1.job": ("J001", "ada", [12, 30]), "job-2.job": ("J002", "brook", [5]),
        "job-3.job": ("J003", "chen", [7, 7, 7]), "job-4.job": ("J004", "dana", [100, 250, 1])}


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


def run(cmd, cwd="/"):
    try:
        return subprocess.run(cmd, cwd=cwd, env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return subprocess.CompletedProcess(cmd, 127, "", repr(e))


def parse_unit(path):
    sec, out = None, {}
    for raw in open(path):
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1]
            out.setdefault(sec, {})
        elif "=" in line and sec:
            k, v = line.split("=", 1)
            out[sec].setdefault(k.strip(), []).append(v.strip())
    return out


# 1. code untouched
for ln in open("/tests/manifest.sha256").read().split("\n"):
    if ln.strip():
        h, p = ln.split("  ", 1)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else None
        check(f"{p} unchanged", got == h)

# 2. the config, read the way the worker reads it
cfg = None
try:
    cp = configparser.ConfigParser()
    with open(INI) as fh:
        cp.read_file(fh)
    cfg = {"inbox": cp.get("queue", "inbox"), "done": cp.get("queue", "done"),
           "batch_size": cp.getint("queue", "batch_size"), "dsn": cp.get("db", "dsn"),
           "report": cp.get("output", "report"), "level": cp.get("log", "level")}
except (OSError, configparser.Error, ValueError) as e:
    check(f"worker.ini parses under strict configparser with interpolation ({type(e).__name__}: {e})", False)
want = {"inbox": "/srv/queue/inbox", "done": "/srv/queue/done", "batch_size": 25,
        "dsn": "postgresql://acme:s3cr%t@db.internal:5432/acme", "report": "/srv/queue/report.txt", "level": "INFO"}
if cfg is not None:
    check(f"worker.ini values: want {want} got {cfg}", cfg == want)
r = run(["/usr/local/bin/python3", "/app/worker/worker.py", "--config", INI, "--check"], cwd="/app/worker")
check(f"worker.py --check accepts worker.ini (rc={r.returncode}) {(r.stdout + r.stderr).strip()!r}", r.returncode == 0)

# 3. the unit
unit = parse_unit(UNIT) if os.path.isfile(UNIT) else {}
svc, inst = unit.get("Service", {}), unit.get("Install", {})
execs = svc.get("ExecStart", [])
argv = shlex.split(execs[0]) if len(execs) == 1 else []
check(f"one ExecStart ({execs})", len(execs) == 1)
exe_ok = bool(argv) and os.path.isabs(argv[0]) and os.path.isfile(argv[0]) and os.access(argv[0], os.X_OK)
check(f"ExecStart's executable exists: {argv[:1]}", exe_ok)
check(f"ExecStart runs /app/worker/worker.py --config {INI} --once: {argv}",
      argv[1:] == ["/app/worker/worker.py", "--config", INI, "--once"])
check(f"WorkingDirectory=/app/worker, Type=oneshot, WantedBy=multi-user.target ({svc.get('WorkingDirectory')}, "
      f"{svc.get('Type')}, {inst.get('WantedBy')})",
      svc.get("WorkingDirectory") == ["/app/worker"] and svc.get("Type") == ["oneshot"]
      and inst.get("WantedBy") == ["multi-user.target"])
check(f"enabled: {LINK} is a symlink to the unit",
      os.path.islink(LINK) and os.path.realpath(LINK) == os.path.realpath(UNIT))

# 4. the queued jobs: processed exactly once
inbox = sorted(os.listdir("/srv/queue/inbox")) if os.path.isdir("/srv/queue/inbox") else None
done = sorted(os.listdir("/srv/queue/done")) if os.path.isdir("/srv/queue/done") else []
check(f"inbox is empty ({inbox}) and done holds the 4 jobs ({done})", inbox == [] and done == sorted(JOBS))
good = True
for n, (jid, who, items) in JOBS.items():
    try:
        good = good and json.load(open(f"/srv/queue/done/{n}")) == {"id": jid, "customer": who, "items": items}
    except (OSError, ValueError):
        good = False
check("the done jobs are the queued jobs, unchanged", good)
want_rep = "".join(f"{jid} {who} {sum(items)}\n" for jid, who, items in JOBS.values())
got_rep = open("/srv/queue/report.txt").read() if os.path.isfile("/srv/queue/report.txt") else None
check(f"report.txt: one line per job, once: want {want_rep!r} got {got_rep!r}", got_rep == want_rep)

# 5. hidden: the unit's own ExecStart, on a copy of worker.ini pointed at a queue with 30 jobs
if exe_ok and cfg is not None and len(argv) >= 4 and "--config" in argv:
    base = "/tmp/verify-worker"
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(f"{base}/inbox")
    for i in range(30):
        json.dump({"id": f"H{i:03d}", "customer": f"c{i % 4}", "items": [i, 2 * i]}, open(f"{base}/inbox/{i:03d}.job", "w"))
    raw = configparser.ConfigParser(interpolation=None)
    raw.read(INI)
    raw.set("queue", "inbox", f"{base}/inbox")
    raw.set("queue", "done", f"{base}/done")
    raw.set("output", "report", f"{base}/report.txt")
    with open(f"{base}/worker.ini", "w") as fh:
        raw.write(fh)
    hidden = list(argv)
    hidden[hidden.index("--config") + 1] = f"{base}/worker.ini"
    r = run(hidden, cwd=(svc.get("WorkingDirectory") or ["/"])[0])
    rep = open(f"{base}/report.txt").read() if os.path.isfile(f"{base}/report.txt") else ""
    want_h = "".join(f"H{i:03d} c{i % 4} {3 * i}\n" for i in range(25))
    check(f"ExecStart on a hidden queue of 30 jobs: rc {r.returncode} {(r.stdout + r.stderr).strip()[-200:]!r}, "
          f"25 processed ({len(rep.splitlines())} lines), 5 left ({len(os.listdir(base + '/inbox'))})",
          r.returncode == 0 and rep == want_h and len(os.listdir(f"{base}/inbox")) == 5)
else:
    check("ExecStart can be run on a hidden queue", False)

print("RESULT", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
