"""Outcome checks for backup-cron-repair."""
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile

ok = True
DEST = "/var/backups/app"
DATA_FILES = {"./orders.csv", "./customers.csv", "./archive/refunds.csv", "./notes.txt"}


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


def expected_manifest():
    paths = []
    for root, _dirs, files in os.walk("/app/data"):
        for f in files:
            if f.endswith(".csv"):
                paths.append("./" + os.path.relpath(os.path.join(root, f), "/app/data"))
    lines = []
    for p in sorted(paths):
        with open(os.path.join("/app/data", p[2:]), "rb") as fh:
            lines.append(f"{hashlib.sha256(fh.read()).hexdigest()}  {p}")
    return lines


def check_artifacts(tag):
    tgz = os.path.join(DEST, "app-data.tar.gz")
    try:
        with tarfile.open(tgz, "r:gz") as tf:
            members = {"./" + m.name.lstrip("./") if not m.name.startswith("./") else m.name
                       for m in tf.getmembers() if m.isfile()}
        check(f"{tag}: tarball members {sorted(members)}", DATA_FILES <= members and
              not any(m.startswith("./app/") or m.startswith("/") for m in members))
    except (OSError, tarfile.TarError) as e:
        check(f"{tag}: tarball readable ({e})", False)
    try:
        got = [ln.rstrip("\n") for ln in open(os.path.join(DEST, "MANIFEST.txt")) if ln.strip()]
        check(f"{tag}: MANIFEST.txt content", got == expected_manifest())
    except OSError as e:
        check(f"{tag}: MANIFEST.txt readable ({e})", False)


def cron_ok(path):
    try:
        lines = [ln.split() for ln in open(path) if ln.strip() and not ln.lstrip().startswith("#")]
    except OSError:
        return False
    for f in lines:
        if len(f) >= 7 and f[2:5] == ["*", "*", "*"] and f[5] == "root" and "/app/ops/backup.sh" in " ".join(f[6:]):
            try:
                if int(f[0]) == 30 and int(f[1]) == 2:
                    return True
            except ValueError:
                pass
    return False


# data untouched
check("data unchanged", open("/app/data/orders.csv").read().count("\n") == 4 and os.path.exists("/app/data/notes.txt"))

# 1) the artifacts left by the agent's run
check_artifacts("agent run")

# 2) cron snippet fixed and installed
check("backup.cron schedules 02:30 daily as root", cron_ok("/app/ops/backup.cron"))
check("/etc/cron.d/app-backup installed", cron_ok("/etc/cron.d/app-backup"))

# 3) the script works from a clean slate with a minimal environment
shutil.rmtree(DEST, ignore_errors=True)
r = subprocess.run(["/app/ops/backup.sh"], cwd="/", env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/root"},
                   capture_output=True, text=True, timeout=120)
print(r.stdout[-500:], r.stderr[-500:])
check(f"fresh run exits 0 (rc={r.returncode})", r.returncode == 0)
check_artifacts("fresh run")

sys.exit(0 if ok else 1)
