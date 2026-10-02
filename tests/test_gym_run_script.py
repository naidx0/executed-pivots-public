"""The GPU lock around gym/run_local_policy.sh: taken in the protocol's line format, refused when held,
released only by its holder, and released when the run fails. Every test uses its own lock file under
tmp_path; the machine's real lock (~/.gpu/gpu.lock) is never read or written here.
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "gym/run_local_policy.sh"
_spec = importlib.util.spec_from_file_location("gpu_lock", REPO_ROOT / "gym/gpu_lock.py")
gl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gl)

# take_the_card.py's ruled order: lane since pid born what
PROTOCOL = re.compile(r"^lane=\S+ since=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ pid=\d+ born=(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ|unknown) what=.+$")
RELEASED = re.compile(r"^released lane=\S+ at=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ held=\d+$")
FOREIGN = "lane=sequence since=2026-09-24T14:06:39Z pid=5128 born=2026-09-24T14:06:38Z what=sequence bank 6 arm guard\n"


def _ledger(lock: Path) -> list[str]:
    p = lock.parent / gl.LOG_NAME
    return p.read_text(encoding="utf-8").splitlines() if p.exists() else []


def test_take_writes_the_protocol_line_and_the_ledger(tmp_path):
    lock = tmp_path / "gpu.lock"
    ok, line = gl.take(lock, "gym-test", "rollouts\n 2 x 89", os.getpid())
    assert ok and PROTOCOL.match(line) and line.endswith("what=rollouts 2 x 89")
    assert lock.read_text(encoding="utf-8") == line + "\n" and _ledger(lock) == [line]
    if os.name == "nt":  # born= is this process's real start time on Windows
        assert "born=unknown" not in line


def test_take_refuses_a_held_lock_and_leaves_it_alone(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text(FOREIGN, encoding="utf-8")
    ok, holder = gl.take(lock, "gym-test", "rollouts", os.getpid())
    assert not ok and holder == FOREIGN.strip()
    assert lock.read_text(encoding="utf-8") == FOREIGN and _ledger(lock) == []


def test_only_one_of_many_racing_takers_wins(tmp_path):
    lock = tmp_path / "gpu.lock"
    results, barrier = [], threading.Barrier(8)

    def taker(i):
        barrier.wait()
        results.append(gl.take(lock, f"racer{i}", "race", 1000 + i)[0])

    threads = [threading.Thread(target=taker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [False] * 7 + [True] and len(_ledger(lock)) == 1


def test_release_removes_only_the_holders_own_lock(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text(FOREIGN, encoding="utf-8")
    assert gl.release(lock, "gym-test", os.getpid()) == (4, FOREIGN.strip())  # negative: not ours
    assert gl.release(lock, "sequence", 9999)[0] == 4  # right lane, wrong pid
    assert lock.read_text(encoding="utf-8") == FOREIGN
    lock.unlink()
    assert gl.release(lock, "gym-test", os.getpid()) == (2, "")
    ok, _ = gl.take(lock, "gym-test", "rollouts", os.getpid())
    code, line = gl.release(lock, "gym-test", os.getpid())  # positive: ours
    assert ok and code == 0 and RELEASED.match(line) and not lock.exists()
    assert _ledger(lock)[-1] == line and len(_ledger(lock)) == 2


# --- the script itself ------------------------------------------------------------------------------------
def _bash() -> str | None:
    if os.name == "nt":  # Git Bash, not the WSL launcher in System32
        git = shutil.which("git")
        for d in Path(git).resolve().parents if git else []:
            if (d / "bin" / "bash.exe").exists() and (d / "usr" / "bin").exists():
                return str(d / "bin" / "bash.exe")
        return None
    return shutil.which("bash")


BASH = _bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no bash (Git Bash on Windows)")


def _run(args, lock: Path, **env) -> subprocess.CompletedProcess:
    e = {**os.environ, "GPU_LOCK": lock.as_posix(), "PY": sys.executable.replace("\\", "/"), **env}
    return subprocess.run([BASH, SCRIPT.as_posix(), *args], cwd=REPO_ROOT, env=e, capture_output=True, text=True,
                          timeout=120)


@needs_bash
def test_script_takes_and_releases_the_lock(tmp_path):
    lock = tmp_path / "gpu.lock"
    r = _run(["--lock-only"], lock)
    assert r.returncode == 0, r.stderr
    take, release = _ledger(lock)
    assert PROTOCOL.match(take) and take.startswith("lane=gym-rollouts ") and RELEASED.match(release)
    assert not lock.exists()


@needs_bash
def test_script_exits_3_on_a_held_lock_without_touching_it(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text(FOREIGN, encoding="utf-8")
    r = _run(["--lock-only"], lock)
    assert r.returncode == 3 and "held" in r.stderr
    assert lock.read_text(encoding="utf-8") == FOREIGN and _ledger(lock) == []


@pytest.fixture
def prepared_out():
    """A tiny prepared run dir inside the repo (the script only takes paths relative to it)."""
    rel = f"out/gym/test-script-{os.getpid()}"
    d = REPO_ROOT / rel / "data"
    d.mkdir(parents=True, exist_ok=True)
    row = {"uuid": "billing-invoice-bugfix:3", "expected_answer": "{}", "responses_create_params": {"input": []},
           "metadata": {"task": "billing-invoice-bugfix", "turn": 3}}
    (d / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (d / "anchors.jsonl").write_text("", encoding="utf-8")
    yield rel
    shutil.rmtree(REPO_ROOT / rel, ignore_errors=True)


@needs_bash
def test_script_releases_the_lock_when_the_run_fails(tmp_path, prepared_out):
    lock = tmp_path / "gpu.lock"
    # Nothing listens on the discard port, so the endpoint check fails after the lock is taken.
    r = _run(["--skip-prepare", "--out", prepared_out, "--serve-wait", "0"], lock,
             LOCAL_POLICY_BASE_URL="http://127.0.0.1:9/v1", LOCAL_POLICY_RETRIES="0", LOCAL_POLICY_TIMEOUT="2")
    assert r.returncode == 1 and "endpoint not up" in r.stdout, r.stdout + r.stderr
    take, release = _ledger(lock)
    assert "what=executed_pivot rollouts: 1 pivots x 2" in take and RELEASED.match(release)
    assert not lock.exists()


@needs_bash
def test_script_does_not_run_rollouts_under_someone_elses_lock(tmp_path, prepared_out):
    lock = tmp_path / "gpu.lock"
    lock.write_text(FOREIGN, encoding="utf-8")
    r = _run(["--skip-prepare", "--out", prepared_out], lock, LOCAL_POLICY_BASE_URL="http://127.0.0.1:9/v1")
    assert r.returncode == 3 and "nothing ran on the GPU" in r.stdout
    assert lock.read_text(encoding="utf-8") == FOREIGN and _ledger(lock) == []
    assert not (REPO_ROOT / prepared_out / "endpoint.json").exists()
