"""Nebius Sandboxes backend (ConTree), on the low-level contree-client 0.4.0.

contree-sdk 0.3.6 cannot turn networking off per run, so this talks to
`contree_client` directly (R3). The platform has no fork call: an image id is
the checkpoint, and forking is spawning again from that id.

    checkpoint  -> the world's current image uuid (immutable)
    fork        -> a new world whose current image is that uuid
    run         -> spawn_instance(cmd, image=current, disposable=False);
                   the operation's result_image_uuid becomes current
    run(keep=False) -> disposable spawn; the world does not advance
    put         -> upload_file + a no-op spawn with `files=` (bakes a new image)
    get         -> inspect_image_archive (tar stream of a path)

Checkpoints hold the filesystem only (no processes, cwd or env): the replay
layer keeps cwd/env in files inside the image (step.sh, plan §7.4).
Documented beta limits: 50 concurrent operations; output truncated at <= 10 MiB.
"""
from __future__ import annotations

import io
import os
import shlex
import shutil
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

from .base import Op, WorldId


class ContreeError(RuntimeError):
    pass


def _v(obj, name):
    val = getattr(obj, name, None)
    return None if val is ... else val


def _text(stream) -> str:
    if stream is None or stream is ...:
        return ""
    return stream.as_bytes().decode("utf-8", "replace")


class ContreeWorld:
    backend = "contree"

    def __init__(self, image: str = "tag:ubuntu:24.04", *, workdir: str = "/work", client=None,
                 network: bool = False, truncate_output_at: int = 1 << 20, _fresh: bool = True):
        if client is None:
            from contree_client.sync import ContreeClient
            client = ContreeClient.from_profile()
        self.client = client
        self.image = image
        self.workdir = workdir
        self.network = network
        self.truncate_output_at = truncate_output_at
        self.ops: list[dict] = []  # ledger rows: op id, parent image, result image, exit, duration
        if _fresh and workdir:
            self.run(["mkdir", "-p", workdir], cwd="/")

    def checkpoint(self) -> WorldId:
        return WorldId(self.image)

    def fork(self, wid: WorldId) -> "ContreeWorld":
        return ContreeWorld(str(wid), workdir=self.workdir, client=self.client, network=self.network,
                            truncate_output_at=self.truncate_output_at, _fresh=False)

    def run(self, cmd: Sequence[str], *, cwd: str | None = None, env: Mapping[str, str] | None = None,
            timeout_s: float = 600.0, keep: bool = True, network: bool | None = None) -> Op:
        from contree_client import models
        net = self.network if network is None else network
        t0 = time.perf_counter()
        kwargs = dict(shell=True, disposable=not keep, cwd=cwd or self.workdir or "/", timeout=int(timeout_s),
                      networking=models.InstanceNetworking(enabled=bool(net)),
                      truncate_output_at=self.truncate_output_at)
        if env:
            kwargs["env"] = dict(env)
        command = " ".join(shlex.quote(c) for c in cmd)
        resp = self.client.spawn_instance(command, self.image, **kwargs)
        op = self.client.wait_operation(resp.uuid, timeout=timeout_s + 120)
        md = _v(op, "metadata")
        res = _v(md, "result") if md is not None else None
        st = _v(res, "state") if res is not None else None
        code = _v(st, "exit_code") if st is not None else None
        timed_out = bool(_v(st, "timed_out")) if st is not None else False
        status = str(_v(op, "status"))
        if code is None:
            code = 124 if timed_out else (0 if status == "SUCCESS" else 125)
        err = _text(_v(res, "stderr") if res is not None else None)
        if status != "SUCCESS":
            err += f"\n[contree] operation {status}: {_v(op, 'error')}"
        new_image = _v(op, "result_image_uuid")
        parent = self.image
        if keep and new_image:
            self.image = new_image
        self.ops.append({"op": resp.uuid, "parent": parent, "image": self.image if keep else None,
                         "exit": code, "timed_out": timed_out, "status": status})
        return Op(id=str(resp.uuid), exit_code=code, stdout=_text(_v(res, "stdout") if res is not None else None),
                  stderr=err, duration_s=time.perf_counter() - t0, cmd=tuple(cmd))

    def put(self, local: Path, remote: str) -> None:
        from contree_client import FileSpec
        local = Path(local)
        pairs = [(remote, local)] if local.is_file() else [
            (os.path.join(remote, str(p.relative_to(local))), p) for p in sorted(local.rglob("*")) if p.is_file()]
        files = {}
        for dest, src in pairs:
            with open(src, "rb") as fh:
                up = self.client.upload_file(fh)
            files[dest] = FileSpec(uuid=up.uuid, mode="0" + oct(src.stat().st_mode & 0o777)[2:])
        resp = self.client.spawn_instance("true", self.image, shell=True, disposable=False, files=files)
        op = self.client.wait_operation(resp.uuid, timeout=600)
        new_image = _v(op, "result_image_uuid")
        if not new_image:
            raise ContreeError(f"put {local} -> {remote}: operation {_v(op, 'status')}: {_v(op, 'error')}")
        self.ops.append({"op": resp.uuid, "parent": self.image, "image": new_image, "exit": 0, "put": remote})
        self.image = new_image

    def get(self, remote: str, local: Path) -> None:
        buf = io.BytesIO()
        for chunk in self.client.inspect_image_archive(self.image, remote):
            buf.write(chunk)
        buf.seek(0)
        with tempfile.TemporaryDirectory() as td, tarfile.open(fileobj=buf, mode="r:*") as tf:
            tf.extractall(td, filter="data")
            entries = [Path(td) / n for n in os.listdir(td)]
            src = entries[0] if len(entries) == 1 else Path(td)
            local = Path(local)
            if src.is_dir():
                shutil.copytree(src, local, dirs_exist_ok=True)
            else:
                local.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, local)

    def close(self) -> None:
        # Images outlive worlds (retained 180 days, untag to hide); nothing to release per world.
        return None

    def __enter__(self) -> "ContreeWorld":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def new_run_tag() -> str:
    return f"xp/{uuid.uuid4().hex[:8]}"
