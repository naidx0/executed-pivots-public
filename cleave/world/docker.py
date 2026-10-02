"""Docker backend: the development and reference implementation of ForkableWorld.

checkpoint  -> `docker commit`  (an image id; immutable, content-addressed by Docker)
fork        -> `docker create` + `docker start` from that image
run         -> `docker exec`

Slower than a real fork and local-only, but it makes "two backends, one
interface" a true sentence and keeps every probe runnable without a key.
"""
from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

from .base import Op, WorldId

DOCKER = shutil.which("docker") or "docker"


class DockerError(RuntimeError):
    pass


def _docker(*args: str, timeout: float = 600.0) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [DOCKER, *args], capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace"
    )
    return proc


def daemon_available() -> bool:
    try:
        return _docker("version", "--format", "{{.Server.Version}}", timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class DockerWorld:
    """One running container. `fork` makes a sibling from a committed image."""

    backend = "docker"

    def __init__(self, image: str, *, workdir: str = "/work", name: str | None = None):
        self.image = image
        self.workdir = workdir
        self.name = name or f"cleave-{uuid.uuid4().hex[:12]}"
        create = _docker(
            "create", "--name", self.name, "-w", workdir, image, "sleep", "infinity"
        )
        if create.returncode != 0:
            raise DockerError(create.stderr.strip())
        self.container_id = create.stdout.strip()
        start = _docker("start", self.name)
        if start.returncode != 0:
            raise DockerError(start.stderr.strip())
        self._closed = False

    # -- the contract -------------------------------------------------------

    def checkpoint(self) -> WorldId:
        proc = _docker("commit", "--change", f"WORKDIR {self.workdir}", self.name)
        if proc.returncode != 0:
            raise DockerError(proc.stderr.strip())
        return WorldId(proc.stdout.strip())  # "sha256:..."

    def fork(self, wid: WorldId) -> "DockerWorld":
        return DockerWorld(str(wid), workdir=self.workdir)

    def run(
        self,
        cmd: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float = 600.0,
    ) -> Op:
        args = ["exec"]
        if cwd:
            args += ["-w", cwd]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        args += [self.name, *cmd]
        t0 = time.perf_counter()
        try:
            proc = _docker(*args, timeout=timeout_s)
            code, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            code = 124
            out = (e.stdout or b"").decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = f"timeout after {timeout_s}s"
        return Op(
            id=uuid.uuid4().hex, exit_code=code, stdout=out, stderr=err,
            duration_s=time.perf_counter() - t0, cmd=tuple(cmd),
        )

    def put(self, local: Path, remote: str) -> None:
        proc = _docker("cp", str(local), f"{self.name}:{remote}")
        if proc.returncode != 0:
            raise DockerError(proc.stderr.strip())

    def get(self, remote: str, local: Path) -> None:
        proc = _docker("cp", f"{self.name}:{remote}", str(local))
        if proc.returncode != 0:
            raise DockerError(proc.stderr.strip())

    def close(self) -> None:
        if self._closed:
            return
        _docker("rm", "-f", self.name, timeout=60)
        self._closed = True

    def __enter__(self) -> "DockerWorld":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def remove_image(wid: WorldId) -> None:
    """Delete a checkpoint image. Tests use this to clean up after themselves."""
    _docker("rmi", "-f", str(wid), timeout=60)
