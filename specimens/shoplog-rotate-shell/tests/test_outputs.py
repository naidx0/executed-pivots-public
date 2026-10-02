"""Outcome checks for shoplog-rotate-shell: rotate.conf is unchanged; /srv/shop/logs was rotated exactly once (every
generation holds the right content, the .gz ones are gzip); last_rotation.txt is right; and copies of the fixed
rotate.sh rotate hidden log sets correctly: from / under a minimal env with the conf next to the script (KEEP=3,
names with spaces, two runs), and with ROTATE_CONF pointing elsewhere (KEEP=2)."""
import gzip
import os
import shutil
import subprocess
import sys

ok = True
LOGS = "/srv/shop/logs"
CONF = "# rotate.sh settings (sourced)\nLOG_DIR=/srv/shop/logs\nSTATE_DIR=/srv/shop/state\n" \
       "# generations kept: NAME.log.1 plus NAME.log.2.gz .. NAME.log.KEEP.gz\nKEEP=4\n"
ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root"}


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


def gen(tag, n, off=0):
    return "".join(f"2024-06-{20 - off:02d} 0{i % 10}:00:00 {tag} line {i}\n" for i in range(1, n + 1))


INITIAL = {
    "api.log": gen("api current", 6), "api.log.1": gen("api gen1", 5, 1), "api.log.2.gz": gen("api gen2", 4, 2),
    "api.log.3.gz": gen("api gen3", 3, 3), "api.log.4.gz": gen("api gen4", 2, 4),
    "worker.log": gen("worker current", 7), "worker.log.1": gen("worker gen1", 3, 1),
    "worker.log.2.gz": gen("worker gen2", 2, 2),
    "payment gateway.log": gen("payment current", 5), "payment gateway.log.1": gen("payment gen1", 4, 1),
}


def rotate(state, keep):
    """The README's rotation on {name: content}; returns (new state, report lines)."""
    s = dict(state)
    report = []
    for log in sorted(n for n in s if n.endswith(".log")):
        report.append(f"{log} {len(s[log].encode())}")
        s.pop(f"{log}.{keep}.gz", None)
        for i in range(keep - 1, 1, -1):
            if f"{log}.{i}.gz" in s:
                s[f"{log}.{i + 1}.gz"] = s.pop(f"{log}.{i}.gz")
        if f"{log}.1" in s:
            s[f"{log}.2.gz"] = s.pop(f"{log}.1")
        s[f"{log}.1"] = s[log]
        s[log] = ""
    return s, sorted(report)


def read_dir(d):
    out, bad = {}, []
    for n in sorted(os.listdir(d)):
        raw = open(os.path.join(d, n), "rb").read()
        if n.endswith(".gz"):
            try:
                out[n] = gzip.decompress(raw).decode()
            except (OSError, EOFError):
                out[n] = None
                bad.append(n)
        else:
            out[n] = raw.decode(errors="replace")
    return out, bad


def diff(got, want):
    names = sorted(set(got) | set(want))
    return [n for n in names if got.get(n, "<missing>") != want.get(n, "<missing>")]


def write_dir(d, state):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    for n, c in state.items():
        p = os.path.join(d, n)
        if n.endswith(".gz"):
            with gzip.open(p, "wb") as fh:
                fh.write(c.encode())
        else:
            open(p, "w").write(c)


def run(cmd, cwd, extra=None):
    try:
        return subprocess.run(cmd, cwd=cwd, env={**ENV, **(extra or {})}, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return subprocess.CompletedProcess(cmd, 127, "", repr(e))


# 1. settings unchanged
check("rotate.conf unchanged", os.path.isfile("/app/ops/rotate.conf") and open("/app/ops/rotate.conf").read() == CONF)

# 2. the real logs: rotated exactly once
want, want_report = rotate(INITIAL, 4)
got, bad = read_dir(LOGS) if os.path.isdir(LOGS) else ({}, [])
check(f"{LOGS} holds one rotation of the shipped logs (differs: {diff(got, want)}; not gzip: {bad})",
      got == want and not bad)
rep = "/srv/shop/state/last_rotation.txt"
got_rep = open(rep).read().splitlines() if os.path.isfile(rep) else None
check(f"last_rotation.txt: want {want_report} got {got_rep}", got_rep == want_report)

# 3. hidden: a copy with rotate.conf next to it, from / with a minimal env, KEEP=3, two runs
base = "/tmp/verify-rotate-1"
shutil.rmtree(base, ignore_errors=True)
os.makedirs(f"{base}/ops")
os.makedirs(f"{base}/state")
shutil.copy2("/app/ops/rotate.sh", f"{base}/ops/rotate.sh")
open(f"{base}/ops/rotate.conf", "w").write(f"LOG_DIR={base}/logs\nSTATE_DIR={base}/state\nKEEP=3\n")
state = {"a b.log": "alpha now\n", "a b.log.1": "alpha one\n", "a b.log.2.gz": "alpha two\n",
         "c.log": "gamma now\ngamma more\n", "[x].log": "bracket\n"}
write_dir(f"{base}/logs", state)
okrun = True
for k in range(2):
    r = run(["/bin/bash", f"{base}/ops/rotate.sh"], cwd="/")
    okrun = okrun and r.returncode == 0
    state, report = rotate(state, 3)
    if k == 0:
        with open(f"{base}/logs/a b.log", "a") as fh:
            fh.write("alpha later\n")
        state["a b.log"] = "alpha later\n"
got, bad = read_dir(f"{base}/logs")
check(f"hidden copy (conf next to the script, run twice from /, env -i): rc ok {okrun} "
      f"{(r.stdout + r.stderr).strip()[-200:]!r}; differs {diff(got, state)}; not gzip {bad}",
      okrun and got == state and not bad)
got_rep = open(f"{base}/state/last_rotation.txt").read().splitlines() if os.path.isfile(
    f"{base}/state/last_rotation.txt") else None
check(f"hidden copy report after the second run: want {report} got {got_rep}", got_rep == report)

# 4. hidden: ROTATE_CONF names another file, KEEP=2, run from /tmp
base2 = "/tmp/verify-rotate-2"
shutil.rmtree(base2, ignore_errors=True)
os.makedirs(f"{base2}/state")
open(f"{base2}/other.conf", "w").write(f"LOG_DIR={base2}/logs\nSTATE_DIR={base2}/state\nKEEP=2\n")
state2 = {"db.log": "db now\n", "db.log.1": "db one\n", "db.log.2.gz": "db two\n"}
write_dir(f"{base2}/logs", state2)
r = run(["/bin/bash", "/app/ops/rotate.sh"], cwd="/tmp", extra={"ROTATE_CONF": f"{base2}/other.conf"})
want2, rep2 = rotate(state2, 2)
got2, bad2 = read_dir(f"{base2}/logs")
check(f"ROTATE_CONF with KEEP=2: rc {r.returncode} {(r.stdout + r.stderr).strip()[-200:]!r}; "
      f"differs {diff(got2, want2)}", r.returncode == 0 and got2 == want2 and not bad2)
check("the real logs were not touched by the hidden runs", read_dir(LOGS)[0] == want)

print("RESULT", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
