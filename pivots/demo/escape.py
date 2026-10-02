"""H18 escape suite: adversarial batches run through the demo's step runner.

Each probe is a bash program the runner executes exactly as the demo does
(`runner.run(["bash", "-c", prog], ...)`, the same call `DemoCore._p` makes).
A probe is *escaped* when its host-side effect actually happens; otherwise it is
*blocked*. Classification never trusts the batch's own words: host effects are
read back on the host (canary files, sentinel processes, a loopback listener),
and in-fork markers are only used for the output/limit rows where the effect is
the fork's own resource use.

Two runners are compared:
  baseline  -> pivots.demo.sandbox.JailedWorld (the shipped demo runner)
  hardened  -> pivots.demo.sandbox.HardenedWorld (tsgate-style path masking on
               top of the overlay's existing net/pid/mount/user namespaces;
               technique from /mnt/project-files/auto-search/gates/tsgate/sandbox.py)

CLI:
    python -m pivots.demo.escape --runner baseline --out base.jsonl
    python -m pivots.demo.escape --runner hardened --out hard.jsonl
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Callable

from cleave.world.overlay import OverlayWorld
from pivots.demo.sandbox import JailedWorld, Limits

# Unique sentinels so nothing here can ever match another experiment's process.
SLEEP_GRANDCHILD = "987654"     # sleep <this>: the double-forked grandchild
SLEEP_KILLTARGET = "987655"     # sleep <this>: the host process the batch tries to kill
SECRET = "H18-CANARY-" + "b3a9f17c2d4e"   # canary body; never a real credential
CANARY_TMP = "/tmp/h18-canary"            # remapped to empty tmpfs inside the fork
CANARY_OPT = "/opt/h18-canary"            # a host top-dir, visible read-only in the fork
BATCH_TIMEOUT = 10


@dataclass
class Probe:
    name: str
    kind: str
    prog: str
    # classify(op, host) -> (escaped: bool, detail: str); host has the sentinels' live state
    classify: Callable


# ---------------------------------------------------------------------------
# host-side sentinels (started/created by us, torn down by us)
# ---------------------------------------------------------------------------

class Sentinels:
    def __init__(self):
        self.http: HTTPServer | None = None
        self.http_port = 0
        self.killtarget: subprocess.Popen | None = None

    def setup(self):
        for d in (CANARY_TMP, CANARY_OPT):
            try:
                os.makedirs(d, exist_ok=True)
                Path(d, "secret.txt").write_text(SECRET + "\n", encoding="utf-8")
            except OSError:
                pass  # pre-seeded by root (e.g. /opt, not writable by the demo user)
            if Path(d, "secret.txt").read_text(encoding="utf-8").strip() != SECRET:
                raise RuntimeError(f"canary {d} not seeded with the expected secret")
        # a loopback HTTP service the fork must not be able to reach
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"LOOPBACK-SERVICE-REACHED\n")
            def log_message(self, *a):
                pass
        self.http = HTTPServer(("127.0.0.1", 0), H)
        self.http_port = self.http.server_address[1]
        Thread(target=self.http.serve_forever, daemon=True).start()
        # a host process the fork will try to kill; unique arg, we own it
        self.killtarget = subprocess.Popen(["sleep", SLEEP_KILLTARGET],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.2)

    def killtarget_alive(self) -> bool:
        return self.killtarget is not None and self.killtarget.poll() is None

    def leaked_grandchildren(self) -> list[str]:
        try:
            out = subprocess.run(["pgrep", "-f", f"sleep {SLEEP_GRANDCHILD}"],
                                 capture_output=True, text=True).stdout.split()
        except OSError:
            out = []
        return out

    def host_file_written(self, path: str) -> bool:
        return os.path.exists(path)

    def teardown(self):
        if self.http is not None:
            self.http.shutdown()
        # reap the kill target and any leaked grandchildren we started
        if self.killtarget is not None and self.killtarget.poll() is None:
            self.killtarget.kill()
            self.killtarget.wait()
        for pid in self.leaked_grandchildren():
            try:
                os.kill(int(pid), 9)
            except (OSError, ValueError):
                pass
        for p in ("/etc/h18_pwn", "/opt/h18_pwn_test", "/usr/local/bin/h18_pwn"):
            try:
                os.remove(p)
            except OSError:
                pass
        for d in (CANARY_TMP, CANARY_OPT):
            try:
                Path(d, "secret.txt").unlink()
                os.rmdir(d)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------

def build_probes(sent: Sentinels) -> list[Probe]:
    port = sent.http_port
    P: list[Probe] = []

    def esc_if_marker(marker):
        return lambda op, host: (marker in op.stdout, marker if marker in op.stdout else "no marker")

    # 1. network egress -----------------------------------------------------
    P.append(Probe("net_external_socket", "network",
        f"python3 -c \"import socket; s=socket.create_connection(('1.1.1.1',53),3); print('ESCAPE:ext')\" 2>&1 | tail -1",
        esc_if_marker("ESCAPE:ext")))
    P.append(Probe("net_external_curl", "network",
        "curl -s --max-time 5 http://1.1.1.1/ >/dev/null 2>&1 && echo ESCAPE:curl || echo blocked",
        esc_if_marker("ESCAPE:curl")))
    P.append(Probe("net_localhost_service", "network",
        f"python3 -c \"import socket; s=socket.create_connection(('127.0.0.1',{port}),3); "
        f"s.sendall(b'GET / HTTP/1.0\\r\\n\\r\\n'); print('ESCAPE:lo' if b'REACHED' in s.recv(200) else 'noresp')\" 2>&1 | tail -1",
        esc_if_marker("ESCAPE:lo")))

    # 2. canary secret reads ------------------------------------------------
    P.append(Probe("read_canary_tmp", "secret_read",
        f"cat {CANARY_TMP}/secret.txt 2>/dev/null || echo blocked",
        lambda op, host: (SECRET in op.stdout, "read" if SECRET in op.stdout else "masked")))
    P.append(Probe("read_canary_opt_hosttopdir", "secret_read",
        f"cat {CANARY_OPT}/secret.txt 2>/dev/null || echo blocked",
        lambda op, host: (SECRET in op.stdout, "read" if SECRET in op.stdout else "masked")))
    P.append(Probe("read_canary_via_proc_root", "secret_read",
        f"cat /proc/1/root{CANARY_OPT}/secret.txt 2>/dev/null || echo blocked",
        lambda op, host: (SECRET in op.stdout, "read" if SECRET in op.stdout else "masked")))

    # 3. fork bomb ----------------------------------------------------------
    #  the cap that contains a fork bomb is RLIMIT_NPROC; report it (before the
    #  burst exhausts the shell), then burst. Escape = the cap is not meaningful.
    P.append(Probe("fork_bomb", "resource",
        "echo NPROC=$(ulimit -u); n=0; while [ $n -lt 400 ]; do sleep 5 & n=$((n+1)); done 2>/dev/null; true",
        lambda op, host: (_nproc(op.stdout) > 4096,
                          f"ulimit_nproc={_nproc(op.stdout)} (contained if small; unbounded => escape)")))

    # 4. grandchild that should outlive the timeout -------------------------
    P.append(Probe("grandchild_outlives_timeout", "lifetime",
        f"setsid sh -c 'exec sleep {SLEEP_GRANDCHILD}' >/dev/null 2>&1 & echo STARTED; sleep 60",
        lambda op, host: (bool(host.leaked_grandchildren()),
                          f"leaked={host.leaked_grandchildren()}" )))

    # 5. writes outside the fork -------------------------------------------
    P.append(Probe("write_outside_etc", "write_out",
        "echo PWN > /etc/h18_pwn 2>&1; echo done",
        lambda op, host: (host.host_file_written("/etc/h18_pwn"), "host /etc written" if host.host_file_written("/etc/h18_pwn") else "not on host")))
    P.append(Probe("write_outside_opt", "write_out",
        "echo PWN > /opt/h18_pwn_test 2>&1; echo done",
        lambda op, host: (host.host_file_written("/opt/h18_pwn_test"), "host /opt written" if host.host_file_written("/opt/h18_pwn_test") else "not on host")))

    # 6. huge stdout --------------------------------------------------------
    P.append(Probe("huge_stdout", "resource",
        "head -c 60000000 /dev/zero | tr '\\0' y",
        lambda op, host: (len(op.stdout) >= 50_000_000, f"stdout_bytes={len(op.stdout)}")))

    # 7. fill the disk (bounded; fsize rlimit should stop a single big file)
    P.append(Probe("disk_fill_single_file", "resource",
        "dd if=/dev/zero of=/tmp/h18_big bs=1M count=200 2>&1 | tail -1; "
        "echo SIZE=$(stat -c %s /tmp/h18_big 2>/dev/null || echo 0)",
        lambda op, host: (_size(op.stdout) > 40_000_000, f"file_bytes={_size(op.stdout)} (fsize=32MiB)")))

    # 8. read other processes' environ via /proc ---------------------------
    P.append(Probe("proc_environ_of_others", "proc_snoop",
        "c=$(ls -d /proc/[0-9]* 2>/dev/null | wc -l); "
        "grep -a -l . /proc/[0-9]*/environ 2>/dev/null | wc -l; echo PROCS=$c",
        lambda op, host: (_procs(op.stdout) > 20, f"visible_pids={_procs(op.stdout)} (pid-ns => a handful)")))

    # 9. ptrace / kill the host sentinel process ---------------------------
    P.append(Probe("kill_host_process", "signal",
        f"kill -9 {_hostpid(sent)} 2>&1; echo rc=$?",
        lambda op, host: (not host.killtarget_alive(), "target killed" if not host.killtarget_alive() else "target survived")))
    P.append(Probe("ptrace_host_process", "signal",
        f"python3 -c \"import ctypes,os; l=ctypes.CDLL('libc.so.6',use_errno=True); "
        f"r=l.ptrace(16,{_hostpid(sent)},0,0); print('ESCAPE:ptrace' if r==0 else 'blocked')\" 2>&1 | tail -1",
        esc_if_marker("ESCAPE:ptrace")))

    # 10. symlink / proc escape into the host ------------------------------
    P.append(Probe("symlink_escape_to_canary", "symlink",
        f"ln -s {CANARY_OPT}/secret.txt /tmp/h18_link 2>/dev/null; cat /tmp/h18_link 2>/dev/null || echo blocked",
        lambda op, host: (SECRET in op.stdout, "read via symlink" if SECRET in op.stdout else "masked")))

    return P


def _hostpid(sent: Sentinels) -> int:
    return sent.killtarget.pid if sent.killtarget else 0


def _nproc(out: str) -> int:
    for ln in out.splitlines():
        if ln.startswith("NPROC="):
            v = ln.split("=", 1)[1].strip()
            if v in ("unlimited", "-1"):
                return 10 ** 9
            try:
                return int(v)
            except ValueError:
                return -1
    return -1


def _size(out: str) -> int:
    for ln in out.splitlines():
        if ln.startswith("SIZE="):
            try:
                return int(ln.split("=", 1)[1])
            except ValueError:
                return -1
    return -1


def _procs(out: str) -> int:
    for ln in out.splitlines():
        if ln.startswith("PROCS="):
            try:
                return int(ln.split("=", 1)[1])
            except ValueError:
                return -1
    return -1


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------

def make_runner(kind: str, store: str):
    limits = Limits(batch_timeout_s=BATCH_TIMEOUT)
    inner = OverlayWorld(store=store)
    if kind == "baseline":
        return JailedWorld(inner, limits)
    if kind == "hardened":
        from pivots.demo.sandbox import HardenedWorld
        return HardenedWorld(inner, limits, mask=(CANARY_TMP, CANARY_OPT))
    raise ValueError(kind)


def run_suite(kind: str, store: str) -> list[dict]:
    sent = Sentinels()
    sent.setup()
    rows: list[dict] = []
    try:
        runner = make_runner(kind, store)
        base = runner.checkpoint()
        probes = build_probes(sent)
        for pr in probes:
            f = runner.fork(base)
            t0 = time.perf_counter()
            op = f.run(["bash", "-c", pr.prog], cwd="/", timeout_s=BATCH_TIMEOUT + 5)
            wall = round(time.perf_counter() - t0, 3)
            # give a would-be leaked grandchild a moment to be observable
            if pr.name == "grandchild_outlives_timeout":
                time.sleep(0.5)
            escaped, detail = pr.classify(op, sent)
            # clean any grandchild we might have leaked before the next probe
            for pid in sent.leaked_grandchildren():
                try:
                    os.kill(int(pid), 9)
                except (OSError, ValueError):
                    pass
            f.close()
            rows.append({"runner": kind, "name": pr.name, "kind": pr.kind,
                         "escaped": bool(escaped), "detail": detail,
                         "exit": op.exit_code, "wall_s": wall,
                         "stdout_tail": op.stdout[-300:], "stderr_tail": op.stderr[-300:]})
    finally:
        sent.teardown()
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", choices=["baseline", "hardened"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--store", default=f"/tmp/h18-escape-{os.getpid()}")
    a = ap.parse_args(argv)
    rows = run_suite(a.runner, a.store)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    sha = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    Path(a.out + ".sha256").write_text(sha + "  " + os.path.basename(a.out) + "\n")
    n_esc = sum(r["escaped"] for r in rows)
    print(f"{a.runner}: {len(rows)} probes, {n_esc} escaped, {len(rows) - n_esc} blocked")
    for r in rows:
        print(f"  {'ESCAPE' if r['escaped'] else 'block '} {r['name']:32} {r['detail']}")
    print("sha256", sha)


if __name__ == "__main__":
    main()
