"""The overlay backend passes the same fork gate as the Docker backend, with no daemon.

Positive control: a file written in the fork is visible in the fork.
Negative control: the same file is absent in the parent and in a second fork
from the same checkpoint. Plus: the host is never written, network is off by
default, checkpoints survive a new process.
"""
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from cleave.world.overlay import OverlayWorld, available

pytestmark = pytest.mark.skipif(not available(), reason="needs user namespaces + overlayfs")

STORE = f"/tmp/cleave-overlay-test-{os.getpid()}"


def test_fork_is_a_new_machine_from_the_checkpoint():
    parent = OverlayWorld(store=STORE)
    assert parent.run(["sh", "-c", "echo anchor > /work/anchor.txt"]).ok
    anchor = parent.checkpoint()
    fork_a = parent.fork(anchor)
    assert fork_a.run(["cat", "/work/anchor.txt"]).stdout.strip() == "anchor"
    assert fork_a.run(["sh", "-c", "echo mutated > /work/only_in_fork.txt"]).ok
    assert fork_a.run(["test", "-f", "/work/only_in_fork.txt"]).ok
    assert not parent.run(["test", "-f", "/work/only_in_fork.txt"]).ok
    fork_b = parent.fork(anchor)
    assert not fork_b.run(["test", "-f", "/work/only_in_fork.txt"]).ok
    assert fork_b.run(["cat", "/work/anchor.txt"]).stdout.strip() == "anchor"


def test_run_reports_exit_code_and_duration():
    with OverlayWorld(store=STORE) as w:
        op = w.run(["sh", "-c", "exit 3"])
        assert op.exit_code == 3 and not op.ok and op.duration_s > 0


def test_host_is_never_written_and_deletes_are_private():
    marker = f"/etc/cleave-{uuid.uuid4().hex}"
    w = OverlayWorld(store=STORE)
    assert w.run(["sh", "-c", f"echo x > {marker} && rm -f /etc/hostname && test ! -e /etc/hostname"]).ok
    assert not os.path.exists(marker) and os.path.exists("/etc/hostname")
    assert w.run(["test", "-f", marker]).ok  # but the world keeps its own writes


def test_network_is_off_unless_asked():
    probe = ["python3", "-c", "import socket;socket.create_connection(('1.1.1.1', 53), 3);print('NET')"]
    w = OverlayWorld(store=STORE)
    op = w.run(probe)
    assert op.exit_code != 0 and "NET" not in op.stdout
    assert "unreachable" in op.stderr.lower() or "errno" in op.stderr.lower()


def test_put_get_roundtrip(tmp_path: Path):
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "f.txt").write_text("payload\n")
    w = OverlayWorld(store=STORE)
    w.put(src, "/app/data")
    assert w.run(["cat", "/app/data/sub/f.txt"]).stdout == "payload\n"
    out = tmp_path / "out"
    w.get("/app/data", out)
    assert (out / "sub" / "f.txt").read_text() == "payload\n"


def test_checkpoint_survives_a_new_process():
    w = OverlayWorld(store=STORE)
    w.run(["sh", "-c", "echo persisted > /work/p.txt"])
    wid = w.checkpoint()
    code = ("from cleave.world.overlay import OverlayWorld;"
            f"w=OverlayWorld({wid!r}, store={STORE!r});print(w.run(['cat','/work/p.txt']).stdout, end='')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parents[1])
    assert out.stdout == "persisted\n", out.stderr


def test_remove_dir_from_lower_layer():
    w = OverlayWorld(store=STORE)
    assert w.run(["bash", "-c", "mkdir -p /work/d/e && touch /work/d/f /work/d/e/g"]).ok
    op = w.run(["bash", "-c", "rm -rf /work/d && mkdir /work/d && echo ok"])
    assert op.ok and "ok" in op.stdout, op.stderr
    w.close()


def test_host_home_hidden():
    w = OverlayWorld(store=STORE)
    op = w.run(["bash", "-c", "ls -A /root /home | wc -l; test -e /root/.gitconfig && echo LEAK || echo clean"])
    assert "clean" in op.stdout
    w.close()


def test_nested_host_directories_are_writable_inside_the_world():
    # Tasks write under host directories (/var/log/<app>, /etc/cron.d, replay's /var/lib/xp). Without
    # real root, host dirs owned by uid 0 are "nobody" inside the user namespace, so this needs the
    # owned directory mirror; as real root it passes either way.
    d = f"/var/lib/cleave-{uuid.uuid4().hex}"
    w = OverlayWorld(store=STORE)
    op = w.run(["sh", "-c", f"mkdir -p {d}/sub && echo ok > {d}/sub/f && cat {d}/sub/f && touch /usr/share/cleave-probe"])
    assert op.ok and op.stdout.strip() == "ok", op.stderr
    assert not os.path.exists(d) and not os.path.exists("/usr/share/cleave-probe")


def test_owned_mirror_is_built_only_without_real_root(tmp_path: Path, monkeypatch):
    from cleave.world import overlay as ov

    host = tmp_path / "host"
    (host / "etc" / "cron.d").mkdir(parents=True)
    (host / "var" / "lib" / "ro").mkdir(parents=True)
    (host / "etc" / "passwd").write_text("root:x:0:0::/root:/bin/sh\n")
    os.chmod(host / "var" / "lib" / "ro", 0o555)
    monkeypatch.setattr(ov.os, "geteuid", lambda: 1000)
    st = ov._Store(str(tmp_path / "store"), host_root=str(host))
    mirror = Path(st.owned)
    assert (mirror / "etc" / "cron.d").is_dir() and (mirror / "var" / "lib" / "ro").is_dir()
    assert not (mirror / "etc" / "passwd").exists()  # directories only; files still come from the host
    assert (mirror / "var" / "lib" / "ro").stat().st_mode & 0o777 == 0o755  # host mode, plus u+w for cleanup
    monkeypatch.setattr(ov.os, "geteuid", lambda: 0)
    assert ov._Store(str(tmp_path / "store-root"), host_root=str(host)).owned is None
