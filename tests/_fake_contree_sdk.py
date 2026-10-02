"""A stand-in for the `contree_sdk` package (no key, no network), injected through sys.modules.

It mirrors the surface cleave.world.nebius uses: ContreeSync() reads its credentials from the environment
like the real IAMAuth does, `images.use(ref)` returns an image, `image.run(cmd, args=[...]).wait()` returns
a new image whose `.result` carries exit_code, stdout, stderr, cost and the raw timed_out flag, and a kept
(non-disposable) run returns a new image uuid. Image state is a dict of key -> value, so checkpoint and
fork semantics can be checked. Commands understood by the fake sandbox:

  mkdir ...          no-op, exit 0
  set K V            store V under K in the new image
  cat K              print the value of K
  exit N             exit with code N
  sleep              report a timed-out run (exit -1, timed_out true)
  boom               the platform raises a plain (non-billing) API error
  billing            the platform raises a 402 whose text quotes the key and the project id (so redaction is tested)
  quota429           the platform raises a TooManyRequestsError (status 429)
  bash ...           prints Sandbox.bash_stdout (the H32 preflight's report)
"""
from __future__ import annotations

import itertools
import os
import types
import uuid as _uuid
from dataclasses import dataclass


class ApiStatusCodeError(Exception):
    def __init__(self, status, error):
        self.status = status
        self.error = error
        super().__init__(f"status={status} error={error}")


class TooManyRequestsError(ApiStatusCodeError):
    def __init__(self, error="Too Many Requests"):
        super().__init__(429, error)


class FailedOperationError(Exception):
    pass


@dataclass
class _State:
    exit_code: int
    timed_out: bool


@dataclass
class _Proc:
    state: _State


@dataclass
class _Raw:
    result: _Proc


@dataclass
class Result:
    exit_code: int
    stdout: str
    stderr: str
    cost: float
    _raw: _Raw


class Sandbox:
    """Shared by every image of one client: the images table and a call log."""

    def __init__(self, key: str, project: str):
        self.key = key
        self.project = project
        self.images: dict[str, dict] = {}
        self.calls: list[dict] = []
        self.ids = itertools.count(1)
        self.bash_stdout = "EGRESS closed\n"

    def new_image(self, files: dict) -> str:
        u = str(_uuid.UUID(int=next(self.ids)))
        self.images[u] = dict(files)
        return u


class Image:
    def __init__(self, box: Sandbox, uuid=None, tag=None, request=None, result=None):
        self._box = box
        self.uuid = uuid
        self.tag = tag
        self._request = request
        self._result = result

    def _files(self) -> dict:
        if self.uuid is not None:
            return self._box.images[str(self.uuid)]
        return self._box.images.setdefault(f"tag:{self.tag}", {})

    def run(self, command=None, *, shell=None, args=None, env=None, cwd=None, hostname=None, timeout=None,
            disposable=True, truncate_output_at=None, **_):
        req = {"command": shell if shell is not None else command, "args": list(args or []), "shell": shell is not None,
               "env": dict(env or {}), "cwd": cwd, "timeout": timeout, "disposable": disposable,
               "image": str(self.uuid) if self.uuid is not None else f"tag:{self.tag}"}
        return Image(self._box, self.uuid, self.tag, request=req)

    def wait(self) -> "Image":
        req = self._request
        self._box.calls.append(req)
        files = dict(self._files())
        cmd, args = req["command"], req["args"]
        out, code, timed_out = "", 0, False
        if cmd == "billing":
            raise ApiStatusCodeError(402, f"Payment Required: billing quota exhausted for project "
                                          f"{self._box.project} (token {self._box.key})")
        if cmd == "quota429":
            raise TooManyRequestsError()
        if cmd == "boom":
            raise FailedOperationError("Operation 1234 has failed: node lost")
        if cmd == "set":
            files[args[0]] = args[1]
        elif cmd == "cat":
            out = files.get(args[0], "")
        elif cmd == "exit":
            code = int(args[0])
        elif cmd == "bash":
            out = self._box.bash_stdout
        elif cmd == "sleep":
            code, timed_out = -1, True
        new = None if req["disposable"] else self._box.new_image(files)
        res = Result(code, out, "", 0.001, _Raw(_Proc(_State(code, timed_out))))
        return Image(self._box, new, None, result=res)

    @property
    def result(self) -> Result:
        return self._result

    def apply_files(self, files=None, **_):
        cur = dict(self._files())
        for dest, src in (files or {}).items():
            cur[str(dest)] = src.decode() if isinstance(src, bytes) else open(src).read()
        self._box.calls.append({"command": "apply_files", "files": sorted((files or {}))})
        return Image(self._box, self._box.new_image(cur), None)

    def tag_as(self, tag):
        self._box.calls.append({"command": "tag_as", "tag": tag, "image": str(self.uuid)})
        self._box.images[f"tag:{tag}"] = dict(self._files())
        return Image(self._box, self.uuid, tag)

    def read(self, path) -> bytes:
        return self._files()[str(path)].encode()


class _Images:
    def __init__(self, box: Sandbox):
        self._box = box

    def oci(self, ref, **_):
        self._box.calls.append({"command": "oci", "ref": str(ref)})
        return Image(self._box, self._box.new_image({}), None)

    def use(self, ref):
        s = str(ref)
        try:
            _uuid.UUID(s)
            return Image(self._box, s, None)
        except ValueError:
            return Image(self._box, None, s)


class ContreeSync:
    instances: list["ContreeSync"] = []

    def __init__(self, *a, **kw):
        # like IAMAuth: the token and the project come from the environment
        self.box = Sandbox(os.environ["NEBIUS_API_KEY"], os.environ["NEBIUS_PROJECT_ID"])
        self.images = _Images(self.box)
        ContreeSync.instances.append(self)

    def __repr__(self):  # a careless client repr that would leak, like printing headers
        return f"ContreeSync(token={self.box.key}, project={self.box.project})"


def module() -> types.ModuleType:
    m = types.ModuleType("contree_sdk")
    m.ContreeSync = ContreeSync
    m.ApiStatusCodeError = ApiStatusCodeError
    m.TooManyRequestsError = TooManyRequestsError
    m.FailedOperationError = FailedOperationError
    ContreeSync.instances = []
    return m
