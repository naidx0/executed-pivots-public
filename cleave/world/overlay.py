"""Overlay backend: ForkableWorld with no Docker daemon.

Linux user + mount (+ network, + pid) namespaces and overlayfs. The host root
filesystem is the base image. Every run mounts the world's layer stack under a
fresh upper layer, chroots in, runs the command, and the upper layer becomes
the world's new top layer. A checkpoint is the tuple of layer dirs, so it is
immutable by construction and a fork costs nothing (no copy, no commit).

    checkpoint  -> freeze the layer tuple, return its id
    fork        -> a new world whose stack is that tuple
    run         -> unshare + overlay mounts + chroot + exec

Why it exists: the Docker backend needs a daemon (Docker Desktop was showing an
error dialog on the dev box). This runs anywhere with unprivileged user
namespaces and overlayfs: WSL2, a Linux laptop, a CI runner, a Nebius VM.
Measured on a 4-core cloud container: 0.1-0.3 s per run, fork free.

Isolation is for correctness (the host is never written, network off by
default), not a security boundary against hostile code; production runs use
Nebius Sandboxes (microVMs) through the ContreeWorld backend.

Limits: base image is the host's root (a Dockerfile's FROM is not pulled);
directories on filesystems that refuse to overlay are bind-mounted read-only
(reported in `readonly_dirs`); a checkpoint holds files only, not processes,
like Sandboxes checkpoints.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence

from .base import Op, WorldId

SKIP_TOP = {"proc", "sys", "dev", "tmp", "run", "mnt", "lost+found", "old_root"}
HIDE_TOP = {"root", "home"}
BASE_ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": "/root",
    "TERM": "dumb",
    "LANG": "C.UTF-8",
}
# Host directories every world sees as a snapshot taken when the store is created, not live (H19). A host
# daemon writing there (rsyslogd appends to /var/log/syslog and kern.log every second on WSL Ubuntu) would
# otherwise reach every world through the live lower layer, and the probe, which watches /var/log when a
# batch names a path there, counted those appends as the step's effect: the expert's own step scored 0.733
# against itself. CLEAVE_OVERLAY_FROZEN (colon-separated absolute paths) replaces the default.
FROZEN_DEFAULT = ("/var/log",)
FROZEN_MAX_BYTES = 8 << 20      # a bigger host file is frozen as a same-size sparse placeholder,
FROZEN_BUDGET_BYTES = 128 << 20  # and so is every file once this much content is copied (systemd's journal)
PROXY_ENV = ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
             "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "PIP_CERT", "NODE_EXTRA_CA_CERTS")


class OverlayError(RuntimeError):
    pass


def available() -> bool:
    """True if this machine can build overlay worlds (user namespaces + overlayfs)."""
    if shutil.which("unshare") is None or not Path("/proc/filesystems").exists():
        return False
    if "overlay" not in Path("/proc/filesystems").read_text():
        return False
    try:
        return subprocess.run(["unshare", "-rm", "true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class _Store:
    """Layer storage shared by every world made from one store (one per root dir)."""

    def __init__(self, root: str | None = None, host_root: str = "/"):
        root = root or os.environ.get("CLEAVE_OVERLAY_ROOT") or os.path.join(tempfile.gettempdir(), "cleave-overlay")
        self.root = os.path.abspath(root)
        self.host_root = host_root
        for d in ("layers", "mnt", "empty", "checkpoints"):
            os.makedirs(os.path.join(self.root, d), exist_ok=True)
        self.checkpoints: dict[str, tuple[str, ...]] = {}
        self._top = self._top_dirs()
        self.owned = self._owned_dirs()
        self.frozen = self._frozen_dirs()
        self.skel = os.path.join(self.root, "skel")
        if not os.path.isdir(self.skel):
            os.makedirs(self.skel)
            for n in sorted(os.listdir(host_root)):
                p = os.path.join(host_root, n)
                if os.path.islink(p):
                    os.symlink(os.readlink(p), os.path.join(self.skel, n))
                elif os.path.isdir(p):
                    os.makedirs(os.path.join(self.skel, n), exist_ok=True)

    def _top_dirs(self) -> list[str]:
        store_real = os.path.realpath(self.root)
        out = []
        for n in sorted(os.listdir(self.host_root)):
            p = os.path.join(self.host_root, n)
            if n in SKIP_TOP or os.path.islink(p) or not os.path.isdir(p):
                continue
            real = os.path.realpath(p)
            if store_real == real or store_real.startswith(real + os.sep):
                # overlayfs refuses an upper dir inside its own lower dir; keep the store out of the image
                raise OverlayError(f"overlay store {self.root} must not live under /{n}; use a path under /tmp")
            out.append(n)
        return out

    def _owned_dirs(self) -> str | None:
        """Without real root, mirror the host's directory tree (directories only) as a layer owned by us.

        Under `unshare -r` a non-root caller is root only over its own uid: host directories owned by
        uid 0 show up as nobody and refuse writes, so a task cannot create /var/log/app or /etc/cron.d/x
        and replay cannot keep /var/lib/xp. Overlay takes a merged directory's owner and mode from the
        topmost layer that has it, so this empty mirror, stacked just above the host, makes every host
        directory writable inside the world while host files still come from the host, read-only.
        Real root needs none of this and gets None.
        """
        if os.geteuid() == 0:
            return None
        owned = os.path.join(self.root, "owned")
        if os.path.isdir(owned):
            return owned
        tmp = f"{owned}.{os.getpid()}.tmp"
        modes = []
        for n in self._top:
            if n in HIDE_TOP:
                continue
            for dirpath, _dirs, _files in os.walk(os.path.join(self.host_root, n)):
                dst = os.path.join(tmp, os.path.relpath(dirpath, self.host_root))
                os.makedirs(dst, exist_ok=True)
                try:
                    modes.append((dst, os.lstat(dirpath).st_mode & 0o7777))
                except OSError:
                    pass
        for dst, mode in reversed(modes):  # children first, so a read-only parent is set last
            os.chmod(dst, mode | 0o700)
        try:
            os.rename(tmp, owned)
        except OSError:  # another process built it first
            shutil.rmtree(tmp, ignore_errors=True)
        return owned

    def _frozen_dirs(self) -> str | None:
        """Snapshot the frozen host directories (FROZEN_DEFAULT) once, as a layer stacked above the live host.

        Each frozen directory's copy is marked opaque, so the host's version below it is never consulted:
        neither appends to files that existed at the snapshot nor files the host creates later reach a world.
        A world's own writes there land in its upper layer as before. Taken when the store is created, so
        before any teacher or student run, and shared by every world of the store (like `owned`).
        """
        env = os.environ.get("CLEAVE_OVERLAY_FROZEN")
        raw = env.split(":") if env is not None else FROZEN_DEFAULT
        paths = sorted({"/" + p.strip("/") for p in raw if p.strip("/")})
        paths = [p for p in paths if p.split("/")[1] in self._top and p.split("/")[1] not in HIDE_TOP
                 and os.path.isdir(os.path.join(self.host_root, p.lstrip("/")))
                 and not os.path.islink(os.path.join(self.host_root, p.lstrip("/")))]
        if not paths:
            return None
        import hashlib
        frozen = os.path.join(self.root, "frozen-" + hashlib.sha256("\n".join(paths).encode()).hexdigest()[:12])
        if os.path.isdir(frozen):
            return frozen
        tmp = f"{frozen}.{os.getpid()}.tmp"
        for p in paths:
            _snapshot_dir(os.path.join(self.host_root, p.lstrip("/")), os.path.join(tmp, p.lstrip("/")))
        try:
            os.rename(tmp, frozen)
        except OSError:  # another process built it first
            _rmtree(tmp)
        return frozen

    def save_checkpoint(self, wid: str, layers: tuple[str, ...]) -> None:
        self.checkpoints[wid] = layers
        with open(os.path.join(self.root, "checkpoints", wid.split(":", 1)[1]), "w") as fh:
            fh.write("\n".join(layers))

    def load_checkpoint(self, wid: str) -> tuple[str, ...]:
        if wid not in self.checkpoints:
            path = os.path.join(self.root, "checkpoints", str(wid).split(":", 1)[-1])
            if not os.path.exists(path):
                raise OverlayError(f"unknown checkpoint {wid}")
            text = open(path).read()
            self.checkpoints[wid] = tuple(x for x in text.split("\n") if x)
        return self.checkpoints[wid]

    def new_layer(self) -> str:
        path = os.path.join(self.root, "layers", uuid.uuid4().hex[:16])
        os.makedirs(path)
        return path

    def layer_path(self, layer: str, abs_path: str) -> str:
        parts = [p for p in abs_path.split("/") if p]
        if not parts:
            raise OverlayError("cannot address /")
        if parts[0] == "tmp":
            return os.path.join(layer, "_tmp", *parts[1:])
        if parts[0] in self._top:
            return os.path.join(layer, *parts)
        return os.path.join(layer, "_root", *parts)

    def mount_script(self, layers: tuple[str, ...], upper: str, mnt: str,
                     mask: Sequence[str] = ()) -> str:
        q = shlex.quote
        work = os.path.join(upper, ".work")

        def lower(sub: str, base: str) -> str:
            return ":".join([os.path.join(layer, sub) for layer in reversed(layers)] + [base])

        subs = ["_root", *self._top, "_tmp"]
        lines = ["set -e", f"mkdir -p {q(mnt)}"]
        lines += [f"mkdir -p {q(os.path.join(upper, s))} {q(os.path.join(work, s))}" for s in subs]
        for layer in layers:
            lines += [f"mkdir -p {q(os.path.join(layer, s))}" for s in subs]

        def ov(sub: str, base: str, target: str) -> str:
            # userxattr (Linux >= 5.11) lets an unprivileged overlay mark directories opaque, without it
            # removing a directory that exists in a lower layer fails with EIO; fall back for old kernels
            opts = (f"lowerdir={q(lower(sub, base))},upperdir={q(os.path.join(upper, sub))},"
                    f"workdir={q(os.path.join(work, sub))}")
            return (f"{{ mount -t overlay overlay -o {opts},userxattr {q(target)} 2>/dev/null || "
                    f"mount -t overlay overlay -o {opts} {q(target)}; }}")

        lines.append(ov("_root", self.skel, mnt))
        for n in self._top:
            tgt = os.path.join(mnt, n)
            # host home directories stay out of worlds (dotfiles such as a signing gitconfig, host repos)
            src = os.path.join(self.root, "empty") if n in HIDE_TOP else os.path.join(self.host_root, n)
            base = [src]
            if n not in HIDE_TOP and self.frozen and os.path.isdir(os.path.join(self.frozen, n)):
                base.insert(0, os.path.join(self.frozen, n))
            if self.owned and n not in HIDE_TOP:
                base.insert(0, os.path.join(self.owned, n))
            base = ":".join(base)
            lines.append(f"{ov(n, base, tgt)} 2>/dev/null || {{ mount --rbind {q(src)} {q(tgt)} && "
                         f"mount -o remount,bind,ro {q(tgt)}; echo __RO__:{n} >&2; }}")
        lines.append(ov("_tmp", os.path.join(self.root, "empty"), os.path.join(mnt, "tmp")))
        lines.append(f"chmod 1777 {q(os.path.join(mnt, 'tmp'))}")
        lines.append(f"mount -t proc proc {q(os.path.join(mnt, 'proc'))}")
        lines.append(f"mount --rbind /dev {q(os.path.join(mnt, 'dev'))}")
        # hardened tier (tsgate-style): cover each masked host path with an empty tmpfs so the run
        # cannot read it at all. Done here, inside the mount namespace and before chroot, while we still
        # hold the mount capability (setpriv later drops all caps). An empty `mask` is the default and
        # emits nothing, so an ordinary run is byte-for-byte the same script as before.
        for m in mask:
            tgt = os.path.join(mnt, m.lstrip("/"))
            lines.append(f"mkdir -p {q(tgt)} 2>/dev/null; "
                         f"mount -t tmpfs -o size=4k,mode=000 none {q(tgt)} 2>/dev/null || true")
        lines.append("set +e")
        lines.append(f'exec chroot {q(mnt)} /bin/sh -c \'cd "$CLEAVE_CWD" 2>/dev/null || cd /; '
                     f'__a=$CLEAVE_ARGV; unset CLEAVE_ARGV CLEAVE_CWD; eval "exec $__a"\'')
        return "\n".join(lines)


_STORES: dict[str, _Store] = {}
_STORES_LOCK = threading.Lock()


def _store(root: str | None) -> _Store:
    key = os.path.abspath(root or os.environ.get("CLEAVE_OVERLAY_ROOT") or
                          os.path.join(tempfile.gettempdir(), "cleave-overlay"))
    with _STORES_LOCK:  # two threads opening a new store would both build its owned/ mirror in one tmp dir
        if key not in _STORES:
            _STORES[key] = _Store(key)
        return _STORES[key]


class OverlayWorld:
    """One world: a layer stack that grows by one layer per run."""

    backend = "overlay"

    def __init__(self, image: str = "host", *, workdir: str = "/work", store: str | None = None,
                 network: bool = False, _layers: tuple[str, ...] = ()):
        if image not in ("host", None) and not image.startswith("overlay:"):
            raise OverlayError("OverlayWorld's base image is the host root; pass image='host'")
        self._store = _store(store)
        self.workdir = workdir
        self.network = network
        self.layers: tuple[str, ...] = _layers
        if image and image.startswith("overlay:"):
            self.layers = self._store.load_checkpoint(image)
        self.readonly_dirs: list[str] = []
        self._closed = False
        if workdir and not _layers and not (image or "").startswith("overlay:"):
            self.run(["mkdir", "-p", workdir], cwd="/")

    # -- the contract -------------------------------------------------------

    def checkpoint(self) -> WorldId:
        wid = "overlay:" + uuid.uuid4().hex
        self._store.save_checkpoint(wid, self.layers)
        return WorldId(wid)

    def fork(self, wid: WorldId) -> "OverlayWorld":
        return OverlayWorld(workdir=self.workdir, store=self._store.root, network=self.network,
                            _layers=self._store.load_checkpoint(wid))

    def run(self, cmd: Sequence[str], *, cwd: str | None = None, env: Mapping[str, str] | None = None,
            timeout_s: float = 600.0, network: bool | None = None, keep: bool = True,
            max_output: tuple[int, int] | None = None, mask: Sequence[str] = ()) -> Op:
        """Run `cmd`. keep=False discards the run's writes (a disposable spawn): the world does not advance.

        max_output=(head, tail) bounds the memory a run's output can take: of each stream only the first
        `head` and the last `tail` bytes are kept, with a marker line where the middle was dropped. The
        default (None) keeps everything, as before.

        mask=(paths,...) covers each host path with an empty tmpfs inside the run's mount namespace, so the
        run cannot read it (the hardened tier). The default () masks nothing and leaves the run unchanged.
        """
        if self._closed:
            raise OverlayError("world is closed")
        net = self.network if network is None else network
        upper = self._store.new_layer()
        mnt = os.path.join(self._store.root, "mnt", os.path.basename(upper))
        full_env = dict(BASE_ENV)
        if net:
            full_env.update({k: os.environ[k] for k in PROXY_ENV if k in os.environ})
        full_env.update(env or {})
        full_env["CLEAVE_CWD"] = cwd or self.workdir or "/"
        full_env["CLEAVE_ARGV"] = " ".join(shlex.quote(c) for c in cmd)
        flags = "-rmpf" + ("" if net else "n")
        t0 = time.perf_counter()
        try:
            argv = ["unshare", flags, "--kill-child", "sh", "-c",
                    self._store.mount_script(self.layers, upper, mnt, mask)]
            if max_output is None:
                p = subprocess.run(argv, capture_output=True, timeout=timeout_s, env=full_env)
                code, out, err = p.returncode, p.stdout, p.stderr
            else:
                code, out, err = _run_capped(argv, full_env, timeout_s, *max_output)
        except subprocess.TimeoutExpired as e:
            code, out, err = 124, e.stdout or b"", (e.stderr or b"") + f"\ntimeout after {timeout_s}s".encode()
        dur = time.perf_counter() - t0
        _rmtree(mnt)
        _rmtree(os.path.join(upper, ".work"))
        err_s = err.decode("utf-8", "replace")
        ro = [ln.split(":", 1)[1] for ln in err_s.splitlines() if ln.startswith("__RO__:")]
        self.readonly_dirs = sorted(set(self.readonly_dirs) | set(ro))
        err_s = "\n".join(ln for ln in err_s.splitlines() if not ln.startswith("__RO__:"))
        if keep:
            self.layers = self.layers + (upper,)
        else:
            _rmtree(upper)
        return Op(id=os.path.basename(upper), exit_code=code, stdout=out.decode("utf-8", "replace"),
                  stderr=err_s, duration_s=dur, cmd=tuple(cmd))

    def put(self, local: Path, remote: str) -> None:
        layer = self._store.new_layer()
        tgt = self._store.layer_path(layer, remote)
        os.makedirs(os.path.dirname(tgt), exist_ok=True)
        local = Path(local)
        if local.is_dir():
            shutil.copytree(local, tgt, symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(local, tgt)
        self.layers = self.layers + (layer,)

    def get(self, remote: str, local: Path) -> None:
        op = self.run(["sh", "-c", f"tar -C / -cf - {shlex.quote(remote.lstrip('/'))} | base64 -w0"])
        if op.exit_code != 0:
            raise OverlayError(f"get {remote}: {op.stderr.strip()}")
        import base64
        import io
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(op.stdout)), mode="r:") as tf:
            with tempfile.TemporaryDirectory() as td:
                tf.extractall(td, filter="data")
                src = Path(td) / remote.lstrip("/")
                local = Path(local)
                if src.is_dir():
                    shutil.copytree(src, local, dirs_exist_ok=True)
                else:
                    local.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, local)

    def close(self) -> None:
        # Layers are shared with checkpoints and forks; they are reclaimed by gc(), not here.
        self._closed = True

    def __enter__(self) -> "OverlayWorld":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _rmtree(path: str) -> None:
    """rmtree that also works as a normal user: a world can leave directories without write permission
    (overlayfs's work/work is mode 000, and a batch may chmod its own dirs), which plain rmtree skips."""
    def fix(func, p, _exc):
        try:
            os.chmod(os.path.dirname(p), 0o700)
            if os.path.isdir(p) and not os.path.islink(p):
                os.chmod(p, 0o700)
                shutil.rmtree(p, onerror=fix)
            else:
                func(p)
        except OSError:
            pass
    shutil.rmtree(path, onerror=fix)
    if os.path.exists(path):  # a parent was fixed after its children failed: one more pass
        shutil.rmtree(path, ignore_errors=True)


def _snapshot_dir(src: str, dst: str) -> None:
    """Copy host directory `src` to `dst` as it is now, and mark `dst` opaque for overlayfs.

    Files keep their mode and times. A file we cannot read (btmp, a root-only log) or one over
    FROZEN_MAX_BYTES, or met after FROZEN_BUDGET_BYTES of content is copied, becomes a sparse placeholder
    of the same size, so a listing or `stat` still sees it. The walk is top-down, so a directory's own logs
    are copied before its subdirectories (/var/log/journal). An unreadable directory becomes an empty one. Directories get u+rwx so gc can remove the snapshot.
    """
    dirs = []
    budget = FROZEN_BUDGET_BYTES
    for dirpath, dirnames, filenames in os.walk(src):
        out = os.path.join(dst, os.path.relpath(dirpath, src))
        os.makedirs(out, exist_ok=True)
        dirs.append((dirpath, out))
        for name in dirnames + filenames:
            s, d = os.path.join(dirpath, name), os.path.join(out, name)
            try:
                st = os.lstat(s)
                if os.path.islink(s):
                    os.symlink(os.readlink(s), d)
                elif os.path.isdir(s):
                    if not os.access(s, os.R_OK | os.X_OK):
                        os.makedirs(d, exist_ok=True)  # os.walk skips it; keep it, empty
                    continue
                elif not os.path.isfile(s):
                    continue  # sockets, fifos: not log content
                elif st.st_size <= min(FROZEN_MAX_BYTES, budget) and os.access(s, os.R_OK):
                    shutil.copy2(s, d)
                    budget -= st.st_size
                else:
                    with open(d, "wb") as fh:
                        fh.truncate(st.st_size)
                    os.chmod(d, st.st_mode & 0o7777)
                    os.utime(d, ns=(st.st_atime_ns, st.st_mtime_ns))
            except OSError:
                pass  # the file vanished or changed type mid-walk (log rotation)
    for s, d in reversed(dirs):
        try:
            st = os.stat(s)
            os.chmod(d, (st.st_mode & 0o7777) | 0o700)
            os.utime(d, ns=(st.st_atime_ns, st.st_mtime_ns))
        except OSError:
            pass
    # opaque: overlay never looks below this directory into the live host. userxattr mounts (non-root)
    # read user.overlay.*; a real-root mount reads trusted.overlay.*
    for attr in ("user.overlay.opaque", "trusted.overlay.opaque"):
        try:
            os.setxattr(dst, attr, b"y")
        except OSError:
            pass


class _HeadTail:
    """Keep the first `head` and the last `tail` bytes of a stream."""

    def __init__(self, head: int, tail: int):
        self.head_n, self.tail_n = head, tail
        self.head = bytearray()
        self.tail = bytearray()
        self.dropped = 0

    def feed(self, chunk: bytes) -> None:
        room = self.head_n - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        if not chunk:
            return
        self.tail += chunk
        over = len(self.tail) - self.tail_n
        if over > 0:
            del self.tail[:over]
            self.dropped += over

    def value(self) -> bytes:
        if not self.dropped:
            return bytes(self.head + self.tail)
        return bytes(self.head) + f"\n[... {self.dropped} bytes of output dropped ...]\n".encode() + bytes(self.tail)


def _run_capped(argv: list[str], env: dict, timeout_s: float, head: int, tail: int) -> tuple[int, bytes, bytes]:
    """subprocess.run(capture_output=True) with bounded memory per stream (see OverlayWorld.run max_output)."""
    p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    bufs = (_HeadTail(head, tail), _HeadTail(head, tail))

    def pump(stream, buf):
        for chunk in iter(lambda: stream.read(65536), b""):
            buf.feed(chunk)
        stream.close()

    readers = [threading.Thread(target=pump, args=(s, b), daemon=True) for s, b in zip((p.stdout, p.stderr), bufs)]
    for t in readers:
        t.start()
    try:
        p.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        p.kill()  # --kill-child takes the whole pid namespace down with unshare, so the pipes close
        p.wait()
        for t in readers:
            t.join(10)
        raise subprocess.TimeoutExpired(argv, timeout_s, output=bufs[0].value(), stderr=bufs[1].value())
    for t in readers:
        t.join()
    return p.returncode, bufs[0].value(), bufs[1].value()


def gc(store: str | None = None, keep: set[str] | None = None) -> int:
    """Delete layers not referenced by any registered checkpoint (or `keep`). Returns count removed."""
    st = _store(store)
    live = set(keep or ())
    for name in os.listdir(os.path.join(st.root, "checkpoints")):
        live.update(st.load_checkpoint("overlay:" + name))
    removed = 0
    for name in os.listdir(os.path.join(st.root, "layers")):
        path = os.path.join(st.root, "layers", name)
        if path not in live:
            _rmtree(path)
            removed += not os.path.exists(path)
    return removed
