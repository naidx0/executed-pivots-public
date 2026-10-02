"""ForkableWorld: the one contract every backend implements.

A world is a machine whose whole state (files, packages, environment, running
processes) can be checkpointed to an immutable, content-addressed id and forked
from that id into a new machine. Cleave's probes are written against this
interface and never against a vendor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, NewType, Protocol, Sequence

WorldId = NewType("WorldId", str)


@dataclass(frozen=True)
class Op:
    """The result of one command run inside a world."""

    id: str
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    cmd: Sequence[str] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class ForkableWorld(Protocol):
    """A machine you can checkpoint, fork, and run commands in."""

    @property
    def backend(self) -> str: ...

    def checkpoint(self) -> WorldId:
        """Freeze the current state and return its immutable id."""
        ...

    def fork(self, wid: WorldId) -> "ForkableWorld":
        """Return a new, independent world started from the state `wid`."""
        ...

    def run(
        self,
        cmd: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float = 600.0,
    ) -> Op:
        """Run `cmd` inside this world and wait for it."""
        ...

    def put(self, local: Path, remote: str) -> None:
        """Copy a local file or directory into the world at `remote`."""
        ...

    def get(self, remote: str, local: Path) -> None:
        """Copy a file or directory out of the world to `local`."""
        ...

    def close(self) -> None:
        """Release the machine. Checkpoints outlive the world that made them."""
        ...
