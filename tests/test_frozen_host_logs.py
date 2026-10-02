"""H19: a host process writing to a watched log during a step must not move X.

On WSL Ubuntu rsyslogd appends to /var/log/syslog and kern.log every second. The overlay's lower layer is
the live host, so every world saw those appends, the probe (which watches /var/log when a batch names a
path there) listed them as the step's effect, and the expert's own step re-run scored 0.733 against itself:
X drifted 1 -> 0 on 51 of 218 stored demo actions. The fix gives every world a frozen snapshot of
/var/log taken when the store is created.

A normal user cannot write host /var/log, so the test freezes a host directory it can write, under
/var/tmp, through the same store setting (CLEAVE_OVERLAY_FROZEN) that freezes /var/log, and runs a fake
background writer on it. Controls: the writer really moves the host file; the step's own write under the
frozen directory is still seen; and without freezing, the same writer breaks self-agreement.
"""
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import pytest

from cleave.world import overlay
from cleave.world.overlay import OverlayWorld, available
from pivots.demo.sandbox import HardenedWorld, JailedWorld, Limits
from pivots.effect import EffectJudge, Pivot

pytestmark = pytest.mark.skipif(not available() or os.geteuid() == 0,
                                reason="needs user namespaces + overlayfs, and a non-root user")


def _act(*cmds: str) -> str:
    return json.dumps({"analysis": "", "plan": "", "task_complete": False,
                       "commands": [{"keystrokes": c, "duration": 0.1} for c in cmds]})


@pytest.fixture
def host_log(monkeypatch):
    """A host log directory, frozen for worlds made from a fresh store, with a writer appending to it."""
    base = Path("/var/tmp") / f"h19-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    (base / "app").mkdir(parents=True)
    log = base / "app" / "app.log"
    log.write_text("line 0\n")
    other = base / "app" / "kern.log"      # rsyslogd writes two logs (syslog and kern.log); so does this one
    other.write_text("line 0\n")
    stop = threading.Event()

    def writer():
        i = 0
        while not stop.is_set():
            i += 1
            for f in (log, other):
                with open(f, "a") as fh:
                    fh.write(f"host line {i} {time.time()}\n")
            stop.wait(0.02)

    t = threading.Thread(target=writer, daemon=True)
    yield base, log, t, stop
    stop.set()
    if t.is_alive():
        t.join(5)
    shutil.rmtree(base, ignore_errors=True)


def _world(kind: str, store: str):
    w = OverlayWorld(store=store, workdir="/app")
    return HardenedWorld(w, Limits()) if kind == "hardened" else JailedWorld(w, Limits())


def _judge_run(jw, base: Path):
    anchor = jw.checkpoint()
    gold = _act(f"ls {base}/app/ && touch /app/done\n")
    pv = Pivot(f"h19-{uuid.uuid4().hex[:6]}", anchor, "/app", gold, cmd_timeout=10)
    j = EffectJudge(jw, check_determinism=True)
    prep = j.prepare(pv)
    return j, pv, gold, prep


@pytest.mark.parametrize("kind", ["current", "hardened"])
def test_background_host_writer_does_not_move_self_agreement(kind, host_log, monkeypatch, tmp_path):
    base, log, writer, stop = host_log
    monkeypatch.setenv("CLEAVE_OVERLAY_FROZEN", f"/var/log:{base}")
    store = f"/tmp/h19-frozen-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        jw = _world(kind, store)          # the snapshot is taken here, before any teacher or student run
        size0 = log.stat().st_size
        writer.start()
        time.sleep(0.2)
        j, pv, gold, prep = _judge_run(jw, base)
        assert log.stat().st_size > size0, "control: the fake writer must really move the host file"
        ref = prep["ref"]
        assert not {str(log), str(log.parent / "kern.log")} & set(ref.meaningful_changes()), ref.meaningful_changes()
        assert prep["self_agreement"] == 1.0
        assert j.score(pv, gold).binary == 1
        # the step's own write under the frozen directory is still its effect
        own = j.score(pv, _act(f"ls {base}/app/ && touch /app/done\n", f"echo pwn >> {log}\n"))
        assert own.binary == 0 and str(log) in own.detail["extra"], own.detail
    finally:
        stop.set()
        shutil.rmtree(store, ignore_errors=True)
        overlay._STORES.pop(os.path.abspath(store), None)


def test_without_freezing_the_writer_breaks_self_agreement(host_log, monkeypatch):
    """Negative control: the same writer on an unfrozen host directory is seen as the step's effect."""
    base, log, writer, stop = host_log
    monkeypatch.setenv("CLEAVE_OVERLAY_FROZEN", "/var/log")
    store = f"/tmp/h19-unfrozen-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        jw = _world("current", store)
        writer.start()
        time.sleep(0.2)
        _j, _pv, _gold, prep = _judge_run(jw, base)
        assert prep["self_agreement"] < 0.75
    finally:
        stop.set()
        shutil.rmtree(store, ignore_errors=True)
        overlay._STORES.pop(os.path.abspath(store), None)


def test_frozen_view_hides_host_files_created_after_the_snapshot(host_log, monkeypatch):
    base, log, _writer, _stop = host_log
    monkeypatch.setenv("CLEAVE_OVERLAY_FROZEN", f"/var/log:{base}")
    store = f"/tmp/h19-late-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        w = OverlayWorld(store=store, workdir="/app")
        before = w.run(["cat", str(log)]).stdout
        (base / "app" / "late.log").write_text("late\n")
        with open(log, "a") as fh:
            fh.write("appended after the snapshot\n")
        assert w.run(["cat", str(log)]).stdout == before
        assert not w.run(["test", "-e", str(base / "app" / "late.log")]).ok
        assert w.run(["sh", "-c", f"echo mine >> {log} && tail -n1 {log}"]).stdout.strip() == "mine"
        assert "mine" not in log.read_text()  # the host was never written
    finally:
        shutil.rmtree(store, ignore_errors=True)
        overlay._STORES.pop(os.path.abspath(store), None)
