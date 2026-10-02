"""The demo's sandbox: an OverlayWorld whose every run is jailed and bounded.

The overlay backend is built for correctness, not as a boundary against hostile code: inside a fork the
batch is root of a user namespace, with the capabilities that come with it (chroot, mount), so a chroot
escape would reach the host's mount tree. The demo runs visitors' text, so every run here, the teacher's
included (X compares like with like), goes through `setpriv` and `prlimit` inside the fork:

  - no capabilities at all: the bounding, inheritable and ambient sets are emptied and the securebits
    stop uid 0 from regaining them on exec, so chroot, mount, ptrace of other namespaces and raw
    device access fail; no_new_privs blocks setuid binaries;
  - rlimits: processes, address space, file size, CPU seconds;
  - no network: the fork gets its own empty network namespace (the backend's default, forced here);
  - a wall-clock cap on every run, on top of the batch timeout the probe applies;
  - output memory bound on the server: first and last bytes only (OverlayWorld.run max_output).

The server itself must not run as real root: under real root, uid 0 inside the fork owns the host's
root-owned files and devices even without capabilities. `python -m pivots.demo` refuses to start as root.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from cleave.world.overlay import OverlayWorld


@dataclass(frozen=True)
class Limits:
    batch_timeout_s: int = 10          # the candidate's (and teacher's) batch, enforced by the probe's `timeout`
    run_cap_s: float = 60.0            # hard wall clock for any one sandbox run (probe, step, verifier)
    nproc: int = 256
    address_space: int = 2 << 30       # bytes per process
    file_size: int = 32 << 20          # bytes per file written
    cpu_s: int = 60
    output_head: int = 256 << 10       # bytes kept from the start of a run's output
    output_tail: int = 4 << 20         # bytes kept from its end (the probe's markers and listing live there)

    def jail(self) -> list[str]:
        return ["setpriv", "--no-new-privs", "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all",
                "--securebits=+noroot,+noroot_locked,+no_setuid_fixup,+no_setuid_fixup_locked,+keep_caps_locked",
                "prlimit", f"--nproc={self.nproc}", f"--as={self.address_space}", f"--fsize={self.file_size}",
                f"--cpu={self.cpu_s}", "--core=0", "--"]


class JailedWorld:
    """ForkableWorld wrapper: same contract as OverlayWorld, every run jailed (see module doc)."""

    backend = "overlay"

    def __init__(self, inner: OverlayWorld, limits: Limits):
        self.inner = inner
        self.limits = limits

    @property
    def layers(self) -> tuple[str, ...]:
        return self.inner.layers

    def checkpoint(self) -> str:
        return self.inner.checkpoint()

    def fork(self, wid: str) -> "JailedWorld":
        return JailedWorld(self.inner.fork(wid), self.limits)

    def run(self, cmd: Sequence[str], *, cwd: str | None = None, env: Mapping[str, str] | None = None,
            timeout_s: float = 600.0, network: bool | None = None, keep: bool = True):
        lim = self.limits
        return self.inner.run([*lim.jail(), *cmd], cwd=cwd, env=env, timeout_s=min(timeout_s, lim.run_cap_s),
                              network=False, keep=keep, max_output=(lim.output_head, lim.output_tail))

    def put(self, local: Path, remote: str) -> None:
        self.inner.put(local, remote)

    def get(self, remote: str, local: Path) -> None:
        self.inner.get(remote, local)

    def close(self) -> None:
        self.inner.close()


class HardenedWorld(JailedWorld):
    """JailedWorld plus tmpfs masking of host paths (the hardened tier, H18).

    The overlay backend already gives every run its own user, mount, pid and network namespaces and a
    chroot, and JailedWorld drops all capabilities, sets rlimits and bounds output. What it does not do is
    hide the host's top-level directories, which appear read-only in the fork; a secret placed under one of
    them (say /opt/x) is therefore readable. This tier covers each path in `mask` with an empty tmpfs
    inside the run's mount namespace, so it cannot be read at all. Grandchild kill on timeout, no network,
    and the process caps are inherited unchanged from JailedWorld / the overlay.

    The masking technique is adapted from the Auto search lane's hardened sandbox,
    /mnt/project-files/auto-search/gates/tsgate/sandbox.py (tier "hardened", `mask`); that runner masks
    from a fresh process, this one masks inside the overlay's existing namespaces.
    """

    def __init__(self, inner: OverlayWorld, limits: Limits, mask: Sequence[str] = ()):
        super().__init__(inner, limits)
        self.mask = tuple(mask)

    def fork(self, wid: str) -> "HardenedWorld":
        return HardenedWorld(self.inner.fork(wid), self.limits, self.mask)

    def run(self, cmd: Sequence[str], *, cwd: str | None = None, env: Mapping[str, str] | None = None,
            timeout_s: float = 600.0, network: bool | None = None, keep: bool = True):
        lim = self.limits
        return self.inner.run([*lim.jail(), *cmd], cwd=cwd, env=env, timeout_s=min(timeout_s, lim.run_cap_s),
                              network=False, keep=keep, max_output=(lim.output_head, lim.output_tail),
                              mask=self.mask)


def running_as_root() -> bool:
    return os.geteuid() == 0
