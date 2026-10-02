mkdir -p /app/worker /etc/acme /etc/systemd/system /srv/queue/inbox /srv/queue/done
cat > /app/worker/worker.py <<'EOF'
"""acme queue worker.

    worker.py --config FILE --check   validate the config and print a summary
    worker.py --config FILE --once    process up to batch_size jobs from the inbox, then exit

The config is INI (configparser, strict, with %-interpolation):
    [queue]  inbox, done, batch_size (1-100)
    [db]     dsn
    [output] report   (one "ID CUSTOMER TOTAL" line is appended per processed job)
    [log]    level    (DEBUG, INFO, WARNING or ERROR)
"""
import argparse
import configparser
import json
import os
import sys

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def load(path):
    cp = configparser.ConfigParser()
    with open(path) as fh:
        cp.read_file(fh)
    cfg = {"inbox": cp.get("queue", "inbox"), "done": cp.get("queue", "done"),
           "batch_size": cp.getint("queue", "batch_size"), "dsn": cp.get("db", "dsn"),
           "report": cp.get("output", "report"), "level": cp.get("log", "level")}
    if not 1 <= cfg["batch_size"] <= 100:
        raise ValueError(f"batch_size {cfg['batch_size']} is out of range 1-100")
    if cfg["level"] not in LEVELS:
        raise ValueError(f"log level {cfg['level']!r} is not one of {', '.join(LEVELS)}")
    return cfg


def process(cfg):
    jobs = sorted(f for f in os.listdir(cfg["inbox"]) if f.endswith(".job"))[:cfg["batch_size"]]
    os.makedirs(cfg["done"], exist_ok=True)
    lines = []
    for name in jobs:
        with open(os.path.join(cfg["inbox"], name)) as fh:
            job = json.load(fh)
        lines.append(f"{job['id']} {job['customer']} {sum(job['items'])}\n")
        os.replace(os.path.join(cfg["inbox"], name), os.path.join(cfg["done"], name))
    with open(cfg["report"], "a") as fh:
        fh.writelines(lines)
    return len(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="worker.py")
    ap.add_argument("--config", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    try:
        cfg = load(a.config)
    except (OSError, configparser.Error, ValueError) as e:
        print(f"config error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    host = cfg["dsn"].rsplit("@", 1)[-1]
    if a.check:
        print(f"config OK: batch_size={cfg['batch_size']} level={cfg['level']} db={host}")
        return 0
    n = process(cfg)
    print(f"processed {n} jobs (db {host})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
EOF
cat > /app/worker/unitcheck.py <<'EOF'
"""Check a systemd unit the way this container can (there is no systemd here).

    python3 unitcheck.py /etc/systemd/system/NAME.service

Checks: [Service] ExecStart names an existing executable; a --config file it passes exists and passes the
worker's --check (run in WorkingDirectory, as ExecStart with --once replaced by --check); WorkingDirectory exists;
[Install] WantedBy is set. Also reports whether the unit is enabled (the WantedBy .wants/ symlink that
`systemctl enable` creates). Exit 1 if there is a problem.
"""
import os
import shlex
import subprocess
import sys


def parse(path):
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
            out[sec][k.strip()] = v.strip()
    return out


def main(path):
    unit = parse(path)
    svc, inst = unit.get("Service", {}), unit.get("Install", {})
    problems = []
    argv = shlex.split(svc.get("ExecStart", ""))
    wd = svc.get("WorkingDirectory", "/")
    if not argv:
        problems.append("[Service] has no ExecStart")
    elif not (os.path.isabs(argv[0]) and os.path.isfile(argv[0]) and os.access(argv[0], os.X_OK)):
        problems.append(f"ExecStart executable {argv[0]} does not exist or is not executable")
    if not os.path.isdir(wd):
        problems.append(f"WorkingDirectory {wd} does not exist")
    if "--config" in argv[:-1]:
        conf = argv[argv.index("--config") + 1]
        if not os.path.isfile(conf):
            problems.append(f"config file {conf} does not exist")
        elif not problems:
            chk = [a if a != "--once" else "--check" for a in argv]
            r = subprocess.run(chk, cwd=wd, env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
            print(f"config check: {(r.stdout + r.stderr).strip()}")
            if r.returncode != 0:
                problems.append(f"config check failed (rc={r.returncode})")
    wanted = inst.get("WantedBy")
    if not wanted:
        problems.append("[Install] has no WantedBy")
    else:
        link = os.path.join(os.path.dirname(path), f"{wanted}.wants", os.path.basename(path))
        enabled = os.path.islink(link) and os.path.realpath(link) == os.path.realpath(path)
        print(f"enabled: {'yes' if enabled else 'no'} ({link})")
    for p in problems:
        print(f"PROBLEM: {p}")
    print("unit OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
EOF
cat > /etc/acme/worker.ini <<'EOF'
# acme worker settings
[queue]
inbox = /srv/queue/inbox
done = /srv/queue/done
batch_size = 10
# raised for the spring backlog
batch_size = 25

[db]
dsn = postgresql://acme:s3cr%t@db.internal:5432/acme

[output]
report = /srv/queue/report.txt

[log]
level = info
EOF
cat > /etc/systemd/system/acme-worker.service <<'EOF'
[Unit]
Description=acme queue worker (one batch per start)
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/app/worker
ExecStart=/usr/bin/python /app/worker/worker.py --config /etc/acme/worker.conf --once

[Install]
WantedBy=multi-user.target
EOF
for j in 1:ada:"12, 30" 2:brook:"5" 3:chen:"7, 7, 7" 4:dana:"100, 250, 1"; do
    IFS=: read -r id who items <<< "$j"
    printf '{"id": "J%03d", "customer": "%s", "items": [%s]}\n' "$id" "$who" "$items" > "/srv/queue/inbox/job-$id.job"
done
