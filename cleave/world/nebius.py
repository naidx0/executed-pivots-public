"""Nebius Token Factory Sandboxes (ConTree) as a step runner, on contree-sdk (H32).

The sibling cleave/world/contree.py talks to the low-level contree-client; this one uses the SDK Nebius
documents today (`pip install contree-sdk`):

    client = ContreeSync()                       # IAMAuth: NEBIUS_API_KEY and NEBIUS_PROJECT_ID, env only
    img = client.images.use(ref)                 # a tag or an image uuid; no import, no request yet
    done = img.run(cmd, args=[...], disposable=False).wait()
    done.result.exit_code, done.result.stdout, done.uuid (the image after the run)

The ForkableWorld contract maps onto it the way contree.py does: an image uuid is the checkpoint, forking
is running again from that uuid, a kept run (disposable=False) advances the world to its result image, a
disposable run (keep=False) does not. `put` bakes files into a new image (`apply_files`), `get` reads one
file back. A specimen's state therefore lives in the image chain: `pivots.specimen.build` runs setup.sh as
one kept run on the base image, the expert replay adds one kept run per turn (cwd and env live in files,
pivots/replay/step.py), and the probe (pivots/effect/probe.py) is a single disposable `bash -c` run on a
fork of the pivot's image, so it sees exactly the files the local overlay runner sees.

What the local hardened runner does that this one does not: no setpriv/prlimit jail (the remote sandbox is
the boundary) and no network switch (contree-sdk 0.3.6 has no networking field on a run; the H32 script
checks egress once before it starts). Timeouts are capped by `run_cap_s` and a timed-out run reads as exit
124 with "timeout after Ns" in stderr, like OverlayWorld.

Spend guard. Every platform call goes through a SpendGuard: a cap on steps (the caller's unit, one stored
demo action) and on platform operations, an abort on anything that looks like billing, quota, 402, 403 or
429, an abort after `max_consecutive_errors` failed calls in a row, and one JSONL row per step with its wall
time, operation count and reported cost. The abort is a BaseException, so the probe's and the demo core's
`except Exception` handlers cannot swallow it, and once tripped the guard refuses every later call.

Credentials come from the environment only and are never held by this module: `require_env` checks that
both variables are set, and the SDK reads them itself. Everything this module writes or raises goes through
`redact`, which removes both values (and any bearer token) from the text.
"""
from __future__ import annotations

import json
import math
import os
import posixpath
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from .base import Op, WorldId

ENV_KEY = "NEBIUS_API_KEY"
ENV_PROJECT = "NEBIUS_PROJECT_ID"
DEFAULT_IMAGE = "xp-h32-base"


class MissingCredentials(RuntimeError):
    pass


class NebiusError(RuntimeError):
    pass


class StepCapReached(RuntimeError):
    pass


class SpendAbort(BaseException):
    """Stop now: a billing, quota or rate error, or a cap. Not an Exception, so broad handlers let it pass."""


def require_env(environ: Mapping[str, str] | None = None) -> None:
    env = os.environ if environ is None else environ
    missing = [k for k in (ENV_KEY, ENV_PROJECT) if not (env.get(k) or "").strip()]
    if missing:
        raise MissingCredentials(
            f"{' and '.join(missing)} not set: export {'them' if len(missing) > 1 else 'it'} in the environment "
            f"(never in a file). The Nebius Sandboxes runner reads {ENV_KEY} and {ENV_PROJECT} from the "
            f"environment only.")


def make_client():
    """A ContreeSync client whose credentials come from the environment (checked first, fail fast)."""
    require_env()
    from contree_sdk import ContreeSync
    return ContreeSync()  # default IAMAuth: token and project are looked up by env var name


_BEARER = re.compile(r"(?i)(bearer\s+)[^\s\"',;]+")


def redact(text: object) -> str:
    s = str(text)
    for k in (ENV_KEY, ENV_PROJECT):
        v = (os.environ.get(k) or "").strip()
        if len(v) >= 4:
            s = s.replace(v, "[redacted]")
    return _BEARER.sub(r"\1[redacted]", s)


_BILLING_STATUS = {402, 403, 429}
_BILLING_NAMES = {"TooManyRequestsError", "ForbiddenError", "PaymentRequiredError"}
_BILLING_TEXT = re.compile(r"(?i)billing|quota|payment|insufficient|balance|\bcredits?\b|rate.?limit|"
                           r"too many requests|forbidden|budget|spend(ing)? limit|\b40[23]\b|\b429\b")


def looks_like_billing(exc: BaseException) -> bool:
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status in _BILLING_STATUS:
        return True
    if type(exc).__name__ in _BILLING_NAMES:
        return True
    return bool(_BILLING_TEXT.search(f"{exc} {getattr(exc, 'error', '') or ''}"))


class SpendGuard:
    """Caps and the per-step log for one run. Thread-safe: the demo core runs X and P in parallel."""

    def __init__(self, max_steps: int = 5, max_ops: int | None = None, log_path: str | Path | None = None,
                 max_consecutive_errors: int = 3):
        self.max_steps = max_steps
        self.max_ops = max_ops or None
        self.log_path = Path(log_path) if log_path else None
        self.max_consecutive_errors = max_consecutive_errors
        self.tripped: str | None = None
        self.steps = 0
        self.ops = 0
        self.cost = 0.0
        self._errors_in_a_row = 0
        self._lock = threading.Lock()

    def _trip(self, reason: str) -> SpendAbort:
        with self._lock:
            if self.tripped is None:
                self.tripped = redact(reason)
            return SpendAbort(self.tripped)

    def before_op(self) -> None:
        with self._lock:
            tripped = self.tripped
            over = self.max_ops is not None and self.ops >= self.max_ops
            if not tripped and not over:
                self.ops += 1
                return
        if tripped:
            raise SpendAbort(tripped)
        raise self._trip(f"max_ops={self.max_ops} reached: no more sandbox operations in this run")

    def after_op(self, *, cost: float = 0.0, ok: bool = True) -> None:
        with self._lock:
            self.cost += float(cost or 0.0)
            if ok:
                self._errors_in_a_row = 0

    def on_error(self, exc: BaseException) -> None:
        """Called with every platform exception; raises SpendAbort when the run must stop."""
        msg = f"{type(exc).__name__}: {exc}"
        if looks_like_billing(exc):
            raise self._trip(f"billing, quota or rate error from the platform, run stopped: {msg}")
        with self._lock:
            self._errors_in_a_row += 1
            n = self._errors_in_a_row
        if n >= self.max_consecutive_errors:
            raise self._trip(f"{n} platform errors in a row, run stopped; last: {msg}")

    @contextmanager
    def step(self, **meta) -> Iterator[dict]:
        with self._lock:
            if self.tripped:
                raise SpendAbort(self.tripped)
            if self.steps >= self.max_steps:
                raise StepCapReached(f"max_steps={self.max_steps} reached")
            self.steps += 1
            row = {"step": self.steps, **meta}
            ops0, cost0 = self.ops, self.cost
        t0 = time.perf_counter()
        status, err = "ok", None
        try:
            yield row
        except BaseException as e:
            status, err = "error", f"{type(e).__name__}: {e}"
            raise
        finally:
            if self.tripped:
                status, err = "abort", self.tripped
            row.update(wall_s=round(time.perf_counter() - t0, 3), ops=self.ops - ops0,
                       cost=round(self.cost - cost0, 6), status=status)
            if err:
                row["error"] = err[:2000]
            self.write(row)

    def write(self, row: dict) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        line = redact(json.dumps(row)) + "\n"
        with self._lock, open(self.log_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)


def _text(v) -> str:
    if v is None:
        return ""
    return v.decode("utf-8", "replace") if isinstance(v, (bytes, bytearray)) else str(v)


class NebiusWorld:
    backend = "nebius"

    def __init__(self, image: str = DEFAULT_IMAGE, *, workdir: str = "/work", client=None,
                 guard: SpendGuard | None = None, run_cap_s: float | None = None,
                 truncate_output_at: int = 4 << 20, _fresh: bool = True):
        self.client = client if client is not None else make_client()
        self.guard = guard if guard is not None else SpendGuard(max_steps=1 << 30)
        self.workdir = workdir
        self.run_cap_s = run_cap_s
        self.truncate_output_at = truncate_output_at
        self._img = self.client.images.use(image)
        self.ops: list[dict] = []
        if _fresh and workdir:
            op = self.run(["mkdir", "-p", workdir], cwd="/")
            if not op.ok:
                raise NebiusError(f"could not create {workdir} in {image}: exit {op.exit_code}: {op.stderr[-500:]}")

    def __repr__(self) -> str:
        return f"NebiusWorld(checkpoint={self.checkpoint()!r}, workdir={self.workdir!r})"

    def checkpoint(self) -> WorldId:
        u = getattr(self._img, "uuid", None)
        return WorldId(str(u) if u is not None else str(self._img.tag))

    def fork(self, wid: WorldId) -> "NebiusWorld":
        return NebiusWorld(str(wid), workdir=self.workdir, client=self.client, guard=self.guard,
                           run_cap_s=self.run_cap_s, truncate_output_at=self.truncate_output_at, _fresh=False)

    def run(self, cmd: Sequence[str], *, cwd: str | None = None, env: Mapping[str, str] | None = None,
            timeout_s: float = 600.0, keep: bool = True, network: bool | None = None) -> Op:
        if network:
            raise ValueError("NebiusWorld cannot enable networking per run (contree-sdk has no such field)")
        timeout = min(timeout_s, self.run_cap_s) if self.run_cap_s else timeout_s
        timeout = max(1, math.ceil(timeout))
        self.guard.before_op()
        t0 = time.perf_counter()
        parent = self.checkpoint()
        try:
            # the sandbox starts runs with HOME unset; the local runner has HOME=/root, and git needs it
            # to read /root/.gitconfig (H34: ledger-git-revert's revert could not commit without it)
            run_env = {"HOME": "/root", **(dict(env) if env else {})}
            done = self._img.run(cmd[0], args=list(cmd[1:]), cwd=cwd or self.workdir or "/",
                                 env=run_env, timeout=timeout, disposable=not keep,
                                 truncate_output_at=self.truncate_output_at).wait()
        except Exception as e:  # noqa: BLE001 - a platform error becomes a failed Op unless it must stop the run
            self.guard.on_error(e)
            self.guard.after_op(ok=False)
            code = 124 if type(e).__name__ == "OperationTimedOutError" else 125
            err = f"[nebius] {type(e).__name__}: {redact(e)}"
            if code == 124:
                err += f"\ntimeout after {timeout}s"
            self.ops.append({"parent": parent, "image": None, "exit": code, "error": type(e).__name__})
            return Op(id="", exit_code=code, stdout="", stderr=err, duration_s=time.perf_counter() - t0,
                      cmd=tuple(cmd))
        res = done.result
        self.guard.after_op(cost=getattr(res, "cost", 0.0), ok=True)
        state = getattr(getattr(getattr(res, "_raw", None), "result", None), "state", None)
        timed_out = bool(getattr(state, "timed_out", False))
        code = res.exit_code
        stderr = _text(res.stderr)
        if timed_out:
            code = 124
            stderr += f"\ntimeout after {timeout}s"
        if getattr(res, "truncated", False):
            stderr += f"\n[nebius] output truncated at {self.truncate_output_at} bytes"
        if keep and getattr(done, "uuid", None) is not None:
            self._img = done
        self.ops.append({"parent": parent, "image": self.checkpoint() if keep else None, "exit": code,
                         "timed_out": timed_out})
        return Op(id=str(getattr(done, "uuid", "") or ""), exit_code=code, stdout=_text(res.stdout),
                  stderr=stderr, duration_s=time.perf_counter() - t0, cmd=tuple(cmd))

    def put(self, local: Path, remote: str) -> None:
        local = Path(local)
        pairs = [(remote, local)] if local.is_file() else [
            (posixpath.join(remote, p.relative_to(local).as_posix()), p) for p in sorted(local.rglob("*"))
            if p.is_file()]
        files = {dest: src.read_bytes() for dest, src in pairs}
        self.guard.before_op()
        try:
            new = self._img.apply_files(files=files)
        except Exception as e:  # noqa: BLE001
            self.guard.on_error(e)
            self.guard.after_op(ok=False)
            raise NebiusError(f"put {local.name} -> {remote}: {type(e).__name__}: {redact(e)}") from None
        self.guard.after_op(ok=True)
        self._img = new

    def get(self, remote: str, local: Path) -> None:
        self.guard.before_op()
        try:
            data = self._img.read(remote)
        except Exception as e:  # noqa: BLE001
            self.guard.on_error(e)
            self.guard.after_op(ok=False)
            raise NebiusError(f"get {remote}: {type(e).__name__}: {redact(e)}") from None
        self.guard.after_op(ok=True)
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)

    def close(self) -> None:
        return None  # images outlive worlds; nothing to release per world

    def __enter__(self) -> "NebiusWorld":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
