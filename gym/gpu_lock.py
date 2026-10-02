"""Take and release the shared GPU lock for a rollout run, in the protocol's lock-line format.

The format and the ledger are copied from ml-harness-stage/scripts/take_the_card.py (not imported across
repos):

    lock line (the whole lock file):  lane=<lane> since=<UTC Z> pid=<pid> born=<pid's start, UTC Z> what=<line>
    ledger (gpu.lock.log beside it):  the full lock line on take, `released lane=<lane> at=<UTC Z> held=<s>` on
                                      release; append-only, one line each

Two things are stricter here than take_the_card.py, and neither changes the format:
  - the take is atomic (O_CREAT | O_EXCL), so two takers racing cannot both win;
  - a release removes the lock only when its pid= (and lane=) are the holder's own, so a run that lost the
    card, or never had it, cannot delete another lane's lock.

    python gym/gpu_lock.py take --lock ~/.gpu/gpu.lock --lane gym-rollouts --what "..." --pid N
    python gym/gpu_lock.py release --lock ~/.gpu/gpu.lock --lane gym-rollouts --pid N
    python gym/gpu_lock.py status --lock ~/.gpu/gpu.lock

Exit codes: 0 done; 3 take refused (the card is held; the holder's line goes to stderr);
4 release refused (the lock is not this holder's); 2 nothing to release.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import sys
import time
from pathlib import Path

FILETIME_EPOCH_OFFSET = 11_644_473_600  # seconds between 1601-01-01 and the Unix epoch
LOG_NAME = "gpu.lock.log"
DEFAULT_LOCK = os.environ.get("GPU_LOCK", "~/.gpu/gpu.lock")
LINE_RE = re.compile(r"^lane=(?P<lane>\S+) since=(?P<since>\S+) pid=(?P<pid>\d+) born=(?P<born>\S+) what=(?P<what>.+)$")


def _utc(dt: datetime.datetime | None = None) -> str:
    return (dt or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


def born_utc(pid: int) -> str:
    """That process's start time as UTC Z (Windows FILETIME), or "" when it cannot be read."""
    if pid <= 0 or os.name != "nt":
        return ""
    import ctypes
    import ctypes.wintypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        created = ctypes.wintypes.FILETIME()
        spare = (ctypes.wintypes.FILETIME * 3)()
        ok = kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(spare[0]),
                                      ctypes.byref(spare[1]), ctypes.byref(spare[2]))
    finally:
        kernel32.CloseHandle(handle)  # released before anyone asks whether the pid is alive
    if not ok:
        return ""
    seconds = ((created.dwHighDateTime << 32) | created.dwLowDateTime) / 10_000_000 - FILETIME_EPOCH_OFFSET
    return _utc(datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc))


def the_line(lane: str, what: str, pid: int) -> str:
    what = " ".join(what.split())  # one line
    return f"lane={lane} since={_utc()} pid={pid} born={born_utc(pid) or 'unknown'} what={what}"


def append(log: Path, line: str) -> bool:
    try:
        with log.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
        return True
    except OSError as e:  # never refuse the card over the ledger, but say so
        print(f"gpu_lock: could not append to {log}: {e}", file=sys.stderr)
        return False


def take(lock: str | Path, lane: str, what: str, pid: int) -> tuple[bool, str]:
    """(True, our line) or (False, the holder's line). Atomic: exactly one concurrent taker wins."""
    path = Path(lock)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = the_line(lane, what, pid)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            return False, path.read_text(encoding="utf-8").strip()
        except OSError:
            return False, "(held; the lock file could not be read)"
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    append(path.parent / LOG_NAME, line)
    return True, line


def held_seconds(line: str, now: datetime.datetime | None = None) -> int | None:
    m = LINE_RE.match(line)
    if not m:
        return None
    try:
        began = datetime.datetime.strptime(m["since"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
    except ValueError:
        return None
    return max(0, int(((now or datetime.datetime.now(datetime.timezone.utc)) - began).total_seconds()))


def release(lock: str | Path, lane: str, pid: int) -> tuple[int, str]:
    """(0, release line) when the lock is ours; (4, holder's line) when it is not; (2, '') when there is none."""
    path = Path(lock)
    if not path.exists():
        return 2, ""
    line = path.read_text(encoding="utf-8").strip()
    m = LINE_RE.match(line)
    if not m or int(m["pid"]) != int(pid) or m["lane"] != lane:
        return 4, line
    held = held_seconds(line)
    path.unlink()
    out = f"released lane={lane} at={_utc()} held={held if held is not None else 0}"
    append(path.parent / LOG_NAME, out)
    return 0, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("take", "release", "status"))
    ap.add_argument("--lock", default=DEFAULT_LOCK)
    ap.add_argument("--lane", default="gym-rollouts")
    ap.add_argument("--what", help="take: one line saying what the card is for")
    ap.add_argument("--pid", type=int, default=os.getpid(), help="the process that holds the card")
    ap.add_argument("--wait", type=float, default=0, help="take: keep trying this many seconds (every 10 s)")
    a = ap.parse_args(argv)
    if a.action == "status":
        p = Path(a.lock)
        print(p.read_text(encoding="utf-8").strip() if p.exists() else "free")
        return 0
    if a.action == "take":
        if not a.what:
            ap.error("take needs --what")
        deadline = time.time() + a.wait
        while True:
            ok, line = take(a.lock, a.lane, a.what, a.pid)
            if ok:
                print(line)
                return 0
            if time.time() >= deadline:
                print(f"gpu_lock: the card is held, not taking it:\n  {line}", file=sys.stderr)
                return 3
            time.sleep(min(10.0, max(0.0, deadline - time.time())))
    code, line = release(a.lock, a.lane, a.pid)
    if code == 0:
        print(line)
    elif code == 4:
        print(f"gpu_lock: not releasing a lock this holder (lane={a.lane} pid={a.pid}) does not own:\n  {line}",
              file=sys.stderr)
    else:
        print(f"gpu_lock: nothing to release at {a.lock}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
