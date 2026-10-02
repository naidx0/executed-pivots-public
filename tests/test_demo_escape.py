"""H18 escape suite as tests: the hardened runner must block every escape, and at least as many as the
baseline runner. Each probe's *host* effect is checked on the host, never trusted from the batch's output.

These run only where the demo runs (user namespaces + overlayfs) and as a non-root user, the same
security model the demo requires; as real root, uid 0 inside the fork maps to real root and the checks
would not mean what they say, so the module skips under root.
"""
import os
import socket
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import pytest

from cleave.world.overlay import OverlayWorld, available
from pivots.demo.sandbox import HardenedWorld, JailedWorld, Limits

pytestmark = pytest.mark.skipif(not available() or os.geteuid() == 0,
                                reason="needs user namespaces + overlayfs, and a non-root user")

SECRET = "H18-TEST-CANARY-" + "9f2c17ab"
GC = "876541"          # unique sleep arg for the double-forked grandchild
TIMEOUT = 6


@pytest.fixture(scope="module")
def canary(tmp_path_factory):
    # a canary on a host top-dir that is visible read-only inside the fork (unlike /tmp, which is remapped
    # to a fresh tmpfs). /var/tmp is world-writable and not remapped, so a non-root test can seed it here.
    d = Path("/var/tmp") / f"h18-canary-{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "secret.txt").write_text(SECRET + "\n")
    yield str(d)
    try:
        (d / "secret.txt").unlink()
        d.rmdir()
    except OSError:
        pass


@pytest.fixture(scope="module")
def loopback():
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"REACHED\n")
        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), H)
    Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def _leaked():
    return subprocess.run(["pgrep", "-f", f"sleep {GC}"], capture_output=True, text=True).stdout.split()


def _reap_leaked():
    for pid in _leaked():
        try:
            os.kill(int(pid), 9)
        except (OSError, ValueError):
            pass


def _probes(canary, port):
    """name -> (prog, escaped(op)->bool). Host effects are read back on the host inside escaped()."""
    return {
        "net_external": (
            "python3 -c \"import socket; socket.create_connection(('1.1.1.1',53),3); print('X')\" 2>&1 | tail -1",
            lambda op: "X" in op.stdout and "unreachable" not in op.stdout),
        "net_localhost": (
            f"python3 -c \"import socket; s=socket.create_connection(('127.0.0.1',{port}),3); "
            f"s.sendall(b'GET / HTTP/1.0\\r\\n\\r\\n'); print('X' if b'REACHED' in s.recv(99) else 'n')\" 2>&1 | tail -1",
            lambda op: "X" in op.stdout),
        "read_canary": (
            f"cat {canary}/secret.txt 2>/dev/null || echo blocked",
            lambda op: SECRET in op.stdout),
        "read_canary_proc_root": (
            f"cat /proc/1/root{canary}/secret.txt 2>/dev/null || echo blocked",
            lambda op: SECRET in op.stdout),
        "symlink_canary": (
            f"ln -s {canary}/secret.txt /tmp/l 2>/dev/null; cat /tmp/l 2>/dev/null || echo blocked",
            lambda op: SECRET in op.stdout),
        "fork_bomb": (
            "echo NPROC=$(ulimit -u)",
            lambda op: not any(l.startswith("NPROC=") and l.split("=")[1].strip().isdigit()
                               and int(l.split("=")[1]) <= 4096 for l in op.stdout.splitlines())),
        "grandchild": (
            f"setsid sh -c 'exec sleep {GC}' >/dev/null 2>&1 & echo started; sleep 60",
            lambda op: bool(_leaked())),
        "write_outside": (
            "echo PWN > /var/tmp/h18_pwn_probe 2>&1; echo done",
            lambda op: os.path.exists("/var/tmp/h18_pwn_probe")),
        "huge_stdout": (
            "head -c 60000000 /dev/zero | tr '\\0' y",
            lambda op: len(op.stdout) >= 50_000_000),
        "disk_fill": (
            "dd if=/dev/zero of=/tmp/big bs=1M count=200 2>/dev/null; "
            "echo SIZE=$(stat -c %s /tmp/big 2>/dev/null || echo 0)",
            lambda op: any(l.startswith("SIZE=") and int(l.split("=")[1]) > 40_000_000
                           for l in op.stdout.splitlines())),
        "proc_snoop": (
            "echo PROCS=$(ls -d /proc/[0-9]* 2>/dev/null | wc -l)",
            lambda op: any(l.startswith("PROCS=") and int(l.split("=")[1]) > 20 for l in op.stdout.splitlines())),
    }


def _run_all(runner, canary, port):
    base = runner.checkpoint()
    out = {}
    for name, (prog, esc) in _probes(canary, port).items():
        f = runner.fork(base)
        op = f.run(["bash", "-c", prog], cwd="/", timeout_s=TIMEOUT + 4)
        if name == "grandchild":
            time.sleep(0.4)
        out[name] = esc(op)
        _reap_leaked()
        f.close()
    try:
        os.remove("/var/tmp/h18_pwn_probe")
    except OSError:
        pass
    return out


@pytest.fixture(scope="module")
def results(canary, loopback, tmp_path_factory):
    lim = Limits(batch_timeout_s=TIMEOUT)
    store_b = str(tmp_path_factory.mktemp("h18-b"))
    store_h = str(tmp_path_factory.mktemp("h18-h"))
    base = _run_all(JailedWorld(OverlayWorld(store=store_b), lim), canary, loopback)
    hard = _run_all(HardenedWorld(OverlayWorld(store=store_h), lim, mask=(canary,)), canary, loopback)
    return base, hard


def test_hardened_blocks_every_escape(results):
    _base, hard = results
    escaped = [k for k, v in hard.items() if v]
    assert escaped == [], f"hardened runner let these escape: {escaped}"


def test_hardened_blocks_at_least_as_many_as_baseline(results):
    base, hard = results
    base_blocked = {k for k, v in base.items() if not v}
    hard_blocked = {k for k, v in hard.items() if not v}
    assert base_blocked <= hard_blocked


def test_baseline_leaks_only_the_host_visible_canary(results):
    # documents the delta: the shipped runner blocks the isolation rows but exposes a secret placed on a
    # host-visible directory (read directly, via /proc/1/root, and via symlink); hardened closes these.
    base, _hard = results
    leaked = {k for k, v in base.items() if v}
    assert leaked == {"read_canary", "read_canary_proc_root", "symlink_canary"}
