"""Execute a Terminus action in a fork of the anchor and capture what it did.

One sandbox run per action: the keystroke batch runs as a child bash (so
`exit` or a hang cannot skip the probe), its merged stdout/stderr is the
terminal output, and a read-only probe afterwards lists every file whose
ctime moved during the run (sha256 per file) plus the full path listing of
the watched roots, so deletions can be diffed against the pre-state.
"""

from __future__ import annotations

import base64
import hashlib
import re
import shlex
from dataclasses import dataclass, field


MARK = "@@PEPROBE-7f3c@@"
DEFAULT_ROOTS = ("/app", "/testbed", "/workspace", "/root", "/home", "/tmp", "/srv", "/opt/app", "/data")
PRUNE = ("*/.git/objects", "*/__pycache__", "*/node_modules", "*/.cache", "*/site-packages",
         "*/dist-packages", "*/.venv", "*/venv", "/proc", "/sys", "/dev")
# /var/log is noise only for the logs the system itself writes: task data can live there too
# (access-log-summary's inputs are /var/log/webapp/*), and appending to them scored 1.0 unseen (red team).
SYSTEM_LOGS = r"/var/log/(apt/|dpkg\.log|alternatives\.log|journal/|lastlog$|[wb]tmp$|faillog$|installer/|unattended-upgrades/)"
NOISE_RE = re.compile(
    r"(\.pyc$|/__pycache__/|/\.cache/|\.bash_history$|/\.python_history$|/\.lesshst$|/\.viminfo$|"
    r"/\.wget-hsts$|" + SYSTEM_LOGS + r"|/var/cache/|/\.npm/|/\.git/(index|logs/|ORIG_HEAD|FETCH_HEAD|COMMIT_EDITMSG)|"
    r"/\.pytest_cache/|/\.mypy_cache/|/tmp/tmp[^/]*$|/\.local/share/)")
VOLATILE_ENV = {"PWD", "OLDPWD", "SHLVL", "_"}
# Commits made by the batch get a fixed clock, so the teacher and a candidate that commit the same change
# produce the same commit hash (in `git commit` output and .git/refs) whichever second each ran in. Without
# it, `git revert`/`git commit` pivots scored below 1 against themselves and were masked at random as
# expert_not_reproducible when the two teacher runs straddled a second.
PINNED_ENV = {"GIT_AUTHOR_DATE": "2026-01-01T00:00:00+0000", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+0000"}
# a directory outside the roots is described by the names it holds ("/"-joined, base64), so a removed entry
# can be told from a created one (created entries are reported as themselves)
DIR_SIG_LOOP = ("while IFS= read -r -d \"\" __d; do printf 'D:%s  %s\\n' "
                "\"$(ls -A \"$__d\" 2>/dev/null | tr '\\n' / | base64 -w0)\" \"$__d\"; done")
# H28: always watched, whatever the step did (hashes of files whose ctime moved, plus full listings, so deletions
# count). A step that edited or deleted the global npm copy without reinstalling it went unseen: H24-H27 only saw
# the copy when the step itself reinstalled it. Other node_modules trees stay excluded, including those nested in a
# global package. A listing that reaches GLOBAL_LIST_CAP reports no deletions there.
GLOBAL_ROOTS = ("/usr/local/lib/node_modules", "/usr/local/bin")
GLOBAL_PRUNE = ("*/node_modules", "*/__pycache__", "*/.cache")
GLOBAL_LIST_CAP = 20000
# H42: the site-packages trees of the Python venvs that exist at the pivot (a pyvenv.cfg at most VENV_DEPTH levels
# under a watched root whose ctime predates the step) are watched like the global npm copy: hashes of files whose
# ctime moved, checked against the pivot (confirm_outside), plus full listings, so deletions count. PRUNE skips
# */site-packages and */venv, so a pip install or uninstall inside /app/venv went unseen (H41 reportgen fixtures #3
# and #7, X=1 with E=0). A venv the step itself creates stays pruned; a pyvenv.cfg is always listed, so deleting a
# venv shows. A listing that reaches VENV_LIST_CAP adds nothing but the confirmed files.
VENV_DEPTH = 4
VENV_PRUNE = ("*/__pycache__", "*/.cache", "*/node_modules")
VENV_LIST_CAP = 20000
PROC_WRAPPER_RE = re.compile(r"^(?:(?:setsid|nohup)(?: -[a-z]+)* )+")
ABS_PATH_RE = re.compile(r"(?<![\w.$-])(/(?:[\w.+@-]+/?)+)")


@dataclass
class Effects:
    output: str                      # merged terminal output of the keystrokes
    exit_code: int | None            # exit status of the last command
    final_cwd: str | None
    changed: dict[str, str]          # path -> sha256 (or "L:<target>" for symlinks, "D" for dirs)
    listing: set[str]                # all paths under the watched roots after the run
    timed_out: bool = False
    backend_error: str | None = None
    elapsed: float = 0.0
    warnings: list[str] = field(default_factory=list)
    roots: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)   # exported shell variables after the batch
    procs: list[str] = field(default_factory=list)      # command lines of processes the batch left running
    venv_s: float = 0.0                                 # seconds the H42 VENV section took (inside `elapsed`)

    def deleted_vs(self, pre_listing: set[str]) -> set[str]:
        """Pre-state paths gone after the run, limited to roots this run actually watched. A path inside a
        site-packages tree counts only when that tree was itself a watched root of this run (H42: an incomplete
        venv listing, or a probe without the watch, must not read as deleting the pivot's venv)."""
        def watched(p: str) -> bool:
            if "/site-packages/" in p:
                return any(p.startswith(r.rstrip("/") + "/") for r in self.roots if r.endswith("/site-packages"))
            return any(p == r or p.startswith(r.rstrip("/") + "/") for r in self.roots)
        return {p for p in pre_listing - self.listing if watched(p) and not NOISE_RE.search(p)}

    def meaningful_changes(self) -> dict[str, str]:
        return {p: h for p, h in self.changed.items() if not NOISE_RE.search(p)}

    def structural_changes(self, pre_listing: set[str]) -> dict[str, str]:
        """meaningful_changes without watched directories whose entries are the same as before the run.

        A directory's ctime also moves when a pruned or noise child appears: running python creates
        __pycache__ (pruned), so its parent showed as a changed directory in every run that imported a
        module, the teacher's included. A wrong edit then shared that free "effect" and scored 0.75 on
        state (H10 patch steps; red team bytecode_parent_ctime at sensor-site-report t4, 0.8 with P=0).
        Directories outside the watched roots (from the sweep) are kept: they were already confirmed."""
        mc = self.meaningful_changes()
        dirs = {p for p, h in mc.items() if h == "D" and self._watched(p)}
        if not dirs:
            return mc
        now = _children(self.listing, dirs)
        before = _children(pre_listing, dirs)
        return {p: h for p, h in mc.items() if p not in dirs or now.get(p, set()) != before.get(p, set())}

    def _watched(self, p: str) -> bool:
        return any(p == r or p.startswith(r.rstrip("/") + "/") for r in self.roots)


def _children(listing: set[str], dirs: set[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for p in listing:
        parent, _, name = p.rpartition("/")
        parent = parent or "/"
        if parent in dirs and not NOISE_RE.search(p) and name != "__pycache__":
            out.setdefault(parent, set()).add(name)
    return out


def roots_for(script: str, cwd: str, extra: tuple = ()) -> list[str]:
    """Watched roots: defaults, the cwd, and top-2-level prefixes of absolute paths in the commands."""
    roots = set(DEFAULT_ROOTS) | {cwd} | set(extra)
    for m in ABS_PATH_RE.finditer(script):
        parts = [p for p in m.group(1).split("/") if p]
        if not parts or parts[0] in ("dev", "proc", "sys", "usr", "bin", "sbin", "lib", "lib64", "etc"):
            # /etc edits matter; watch the exact file's dir rather than all of /etc
            if parts and parts[0] == "etc":
                roots.add("/" + "/".join(parts[:2]) if len(parts) > 1 else "/etc")
            continue
        roots.add("/" + "/".join(parts[:2]))
    return sorted(r for r in roots if r != "/")


def build_probe_script(cmds_script: str, roots: list[str], cmd_timeout: int, max_hash_mb: int = 50,
                       sweep: bool = True, venv: bool = True) -> str:
    """sweep: also look for changes outside the watched roots (the rest of /, stat only, ~0.5-2 s) and list
    the processes the batch left running. Without it, a write whose path never appears literally in the
    batch (built from base64, say) to /usr/local/bin or /var/spool/cron, or a detached background process,
    went unseen: the red team's expert-batch-plus-backdoor scored 1.0 at all 39 pivots it was tried on."""
    b64 = base64.b64encode(cmds_script.encode()).decode()
    q = " ".join(shlex.quote(r) for r in roots)
    prune = " -o ".join(f"-path {shlex.quote(p)}" for p in PRUNE)
    pinned = " ".join(f"{k}={shlex.quote(v)}" for k, v in PINNED_ENV.items())
    sweep_prune = " -o ".join(f"-path {shlex.quote(p)}" for p in (*PRUNE, "/run", *roots))
    procs0 = '__PE_P0=" $(cd /proc && echo [0-9]*) "' if sweep else ""
    outside = f"""echo "{MARK} OUTSIDE"
find / \\( {sweep_prune} \\) -prune -o -type f -size -{max_hash_mb}M -newerct "@$__PE_T0" -print0 2>/dev/null | head -z -n 2000 | xargs -0 -r sha256sum 2>/dev/null
find / \\( {sweep_prune} \\) -prune -o -type l -newerct "@$__PE_T0" -printf 'L:%l  %p\\n' 2>/dev/null | head -n 500
find / \\( {sweep_prune} \\) -prune -o -type d -newerct "@$__PE_T0" -print0 2>/dev/null | head -z -n 500 | {DIR_SIG_LOOP}""" if sweep else ""
    gq = " ".join(shlex.quote(r) for r in GLOBAL_ROOTS)
    gprune = " -o ".join(f"-path {shlex.quote(p)}" for p in GLOBAL_PRUNE)
    # the prefix itself is a node_modules directory: it is the one such path not pruned
    gfind = f"find $__PE_G -xdev \\( {gprune} \\) ! -path {shlex.quote(GLOBAL_ROOTS[0])} -prune -o"
    watch = f"""echo "{MARK} GLOBAL $(date +%s.%N)"
__PE_G=""
for r in {gq}; do [ -e "$r" ] && __PE_G="$__PE_G $r"; done
if [ -n "$__PE_G" ]; then
  {gfind} -type f -size -{max_hash_mb}M -newerct "@$__PE_T0" -print0 2>/dev/null | head -z -n 5000 | xargs -0 -r sha256sum 2>/dev/null
  {gfind} -type l -newerct "@$__PE_T0" -printf 'L:%l  %p\\n' 2>/dev/null | head -n 2000
  {gfind} -type d -newerct "@$__PE_T0" -printf 'D  %p\\n' 2>/dev/null | head -n 2000
fi
echo "{MARK} GLOBALLIST"
[ -n "$__PE_G" ] && {gfind} -print 2>/dev/null | head -n {GLOBAL_LIST_CAP + 1}
echo "{MARK} GLOBALEND $(date +%s.%N)\""""
    vprune = " -o ".join(f"-path {shlex.quote(p)}" for p in VENV_PRUNE)
    vfind = f"find $__PE_V -xdev \\( {vprune} \\) -prune -o"
    venv_sec = f"""echo "{MARK} VENV $(date +%s.%N)"
__PE_V=""
if [ -n "$__PE_ROOTS" ]; then
  while IFS= read -r -d "" __c; do
    printf 'C  %s\\n' "$__c"
    if [ -z "$(find "$__c" -maxdepth 0 -newerct "@$__PE_T0" 2>/dev/null)" ]; then
      for __s in "${{__c%/pyvenv.cfg}}"/lib/python*/site-packages; do
        [ -d "$__s" ] && [ ! -L "$__s" ] && __PE_V="$__PE_V $__s" && printf 'S  %s\\n' "$__s"
      done
    fi
  done < <(find $__PE_ROOTS -xdev -maxdepth {VENV_DEPTH} \\( -path '*/node_modules' -o -path '*/.git' -o -path '*/site-packages' \\) -prune -o -name pyvenv.cfg -type f -print0 2>/dev/null | sort -zu)
fi
if [ -n "$__PE_V" ]; then
  {vfind} -type f -size -{max_hash_mb}M -newerct "@$__PE_T0" -print0 2>/dev/null | head -z -n 5000 | xargs -0 -r sha256sum 2>/dev/null
  {vfind} -type l -newerct "@$__PE_T0" -printf 'L:%l  %p\\n' 2>/dev/null | head -n 2000
  {vfind} -type d -newerct "@$__PE_T0" -printf 'D  %p\\n' 2>/dev/null | head -n 2000
fi
echo "{MARK} VENVLIST"
[ -n "$__PE_V" ] && {vfind} -print 2>/dev/null | head -n {VENV_LIST_CAP + 1}
echo "{MARK} VENVEND $(date +%s.%N)\"""" if venv else ""
    procs = f"""echo "{MARK} PROCS"
for __p in /proc/[0-9]*; do __n=${{__p#/proc/}}
  case "$__PE_P0" in *" $__n "*) continue;; esac
  [ "$__n" = "$$" ] && continue
  __c=""; while IFS= read -r -d "" __a; do __c="$__c$__a "; done < "$__p/cmdline" 2>/dev/null
  [ -n "$__c" ] && printf '%s\\n' "${{__c//$'\\n'/ }}"
done""" if sweep else ""
    return f"""exec 3>&1
{procs0}
__PE_T0=$(date +%s.%N)
# file timestamps come from the coarse kernel clock, up to one tick behind date: wait out a tick
# so nothing the batch changes can carry a ctime at or before T0
sleep 0.02
export PE_CMDS="$(printf %s {b64} | base64 -d)"
{pinned} timeout -k 2 {int(cmd_timeout)} bash -c '[ -f /var/lib/xp/env.sh ] && . /var/lib/xp/env.sh 2>/dev/null
__pe_end() {{ __rc=$?; echo; echo "{MARK} END rc=$__rc pwd=$PWD" >&3
  while IFS= read -r -d "" __kv; do printf "{MARK} ENV %s=%q\\n" "${{__kv%%=*}}" "${{__kv#*=}}"; done < <(env -0 | LC_ALL=C sort -z) >&3; }}
__pe_c=$PE_CMDS; unset PE_CMDS; trap __pe_end EXIT; eval "$__pe_c"' 2>&1 < /dev/null
__PE_RC=$?
echo "{MARK} RC $__PE_RC"
{procs}
__PE_ROOTS=""
for r in {q}; do [ -e "$r" ] && __PE_ROOTS="$__PE_ROOTS $r"; done
echo "{MARK} CHANGED"
if [ -n "$__PE_ROOTS" ]; then
  find $__PE_ROOTS -xdev \\( {prune} \\) -prune -o -type f -size -{max_hash_mb}M -newerct "@$__PE_T0" -print0 2>/dev/null | head -z -n 5000 | xargs -0 -r sha256sum 2>/dev/null
  find $__PE_ROOTS -xdev \\( {prune} \\) -prune -o -type l -newerct "@$__PE_T0" -printf 'L:%l  %p\\n' 2>/dev/null | head -n 2000
  find $__PE_ROOTS -xdev \\( {prune} \\) -prune -o -type d -newerct "@$__PE_T0" -printf 'D  %p\\n' 2>/dev/null | head -n 2000
fi
{watch}
{venv_sec}
{outside}
echo "{MARK} LISTING"
[ -n "$__PE_ROOTS" ] && find $__PE_ROOTS -xdev \\( {prune} \\) -prune -o -print 2>/dev/null | head -n 100000
echo "{MARK} DONE"
"""


def _sig_line(line: str) -> tuple[str, str] | None:
    if line.startswith("L:"):
        tgt, _, p = line[2:].partition("  ")
        return p, "L:" + tgt
    if line.startswith("D:") and "  " in line:
        sig, _, p = line.partition("  ")
        return p, sig
    if len(line) > 66 and line[64:66] in ("  ", " *"):
        return line[66:], line[:64]
    if line.startswith("A  "):
        return line[3:], "ABSENT"
    return None


def parse_outside(stdout: str) -> dict[str, str]:
    """Paths outside the watched roots whose ctime moved: path -> sha256 | L:<target> | D:<names digest>."""
    _, _, rest = stdout.partition(f"{MARK} OUTSIDE\n")
    body = rest.split(MARK, 1)[0] if rest else ""
    return dict(x for x in map(_sig_line, body.splitlines()) if x)


def signatures(world, anchor, paths: list[str]) -> dict[str, str]:
    """The anchor's view of `paths`, in parse_outside's format (ABSENT when a path does not exist)."""
    if not paths:
        return {}
    listing = base64.b64encode("\0".join(paths).encode()).decode()
    script = (f"printf %s {listing} | base64 -d | while IFS= read -r -d '' __p || [ -n \"$__p\" ]; do\n"
              "  if [ -L \"$__p\" ]; then printf 'L:%s  %s\\n' \"$(readlink \"$__p\")\" \"$__p\"\n"
              "  elif [ -f \"$__p\" ]; then sha256sum \"$__p\"\n"
              f"  elif [ -d \"$__p\" ]; then printf '%s\\0' \"$__p\" | {DIR_SIG_LOOP}\n"
              "  else printf 'A  %s\\n' \"$__p\"; fi\ndone\n")
    f = world.fork(anchor)
    try:
        kw = {"keep": False} if _accepts_keep(f) else {}
        op = f.run(["bash", "-c", script], cwd="/", timeout_s=120, **kw)
    finally:
        f.close()
    return dict(x for x in map(_sig_line, op.stdout.splitlines()) if x)


def confirm_outside(world, anchor, outside: dict[str, str]) -> dict[str, str]:
    """Outside-root paths that really differ from the anchor, as `changed` entries (dirs as "D").

    ctime alone is not evidence outside the roots: on the overlay backend the lower layer is the live
    host, and host processes (a proxy refreshing /etc/profile.d, rbenv rehashing its shims) move ctimes
    under every world. Their content is the same through the anchor, so comparing contents cancels them,
    while the batch's own writes differ."""
    if not outside:
        return {}
    pre = signatures(world, anchor, sorted(outside))
    out = {}
    for p, sig in outside.items():
        if sig.startswith("D:"):
            # a directory is an effect only when an entry it held at the anchor is gone: a created entry is
            # reported as itself, and a transient host temp file only ever adds a name
            before = pre.get(p, "")
            if before.startswith("D:") and _names(before) - _names(sig):
                out[p] = "D"
        elif pre.get(p) != sig:
            out[p] = sig
    return out


def _names(sig: str) -> set[str]:
    try:
        return {n for n in base64.b64decode(sig[2:]).decode("utf-8", "replace").split("/") if n}
    except ValueError:
        return set()


def parse_global(stdout: str) -> tuple[dict[str, str], set[str], bool, float | None]:
    """The GLOBAL section: (changed under GLOBAL_ROOTS, their listing, listing complete, seconds the section took)."""
    changed: dict[str, str] = {}
    listing: set[str] = set()
    t0 = t1 = None
    section = None
    for line in stdout.splitlines():
        if line.startswith(MARK):
            section, _, arg = line[len(MARK):].strip().partition(" ")
            try:
                t0 = float(arg) if section == "GLOBAL" else t0
                t1 = float(arg) if section == "GLOBALEND" else t1
            except ValueError:
                pass
            continue
        if section == "GLOBAL":
            if line.startswith("D  "):
                changed.setdefault(line[3:], "D")
            else:
                x = _sig_line(line)
                if x and not x[1].startswith(("D:", "ABSENT")):
                    changed[x[0]] = x[1]
        elif section == "GLOBALLIST" and line:
            listing.add(line)
    complete = t1 is not None and len(listing) <= GLOBAL_LIST_CAP
    return changed, listing, complete, (t1 - t0 if t0 is not None and t1 is not None else None)


def parse_venv(stdout: str) -> tuple[dict[str, str], set[str], list[str], bool, float | None]:
    """The H42 VENV section: (changed under the watched site-packages trees, their listing plus every pyvenv.cfg
    found, the site-packages trees watched, listing complete, seconds the section took)."""
    changed: dict[str, str] = {}
    listing: set[str] = set()
    trees: list[str] = []
    t0 = t1 = None
    section = None
    n_list = 0
    for line in stdout.splitlines():
        if line.startswith(MARK):
            section, _, arg = line[len(MARK):].strip().partition(" ")
            try:
                t0 = float(arg) if section == "VENV" else t0
                t1 = float(arg) if section == "VENVEND" else t1
            except ValueError:
                pass
            continue
        if section == "VENV":
            if line.startswith("C  "):
                listing.add(line[3:])
            elif line.startswith("S  "):
                trees.append(line[3:])
            elif line.startswith("D  "):
                changed.setdefault(line[3:], "D")
            else:
                x = _sig_line(line)
                if x and not x[1].startswith(("D:", "ABSENT")):
                    changed[x[0]] = x[1]
        elif section == "VENVLIST" and line:
            listing.add(line)
            n_list += 1
    complete = t1 is not None and n_list <= VENV_LIST_CAP
    return changed, listing, trees, complete, (t1 - t0 if t0 is not None and t1 is not None else None)


def parse_procs(stdout: str) -> list[str]:
    """Command lines the probe listed in its PROCS section (processes the batch left running)."""
    _, _, rest = stdout.partition(f"{MARK} PROCS\n")
    body = rest.split(MARK, 1)[0] if rest else ""
    # a process caught between fork and exec still shows its launcher: drop setsid/nohup so the teacher
    # and a candidate that start the same process compare equal whichever moment the listing ran
    return sorted(PROC_WRAPPER_RE.sub("", ln.strip())[:200] for ln in body.splitlines() if ln.strip())


def parse_probe(stdout: str) -> tuple[str, int | None, str | None, dict, set, bool, dict]:
    if MARK not in stdout:
        return stdout, None, None, {}, set(), False, {}
    head, _, rest = stdout.partition(f"{MARK} RC ")
    output, final_cwd, exit_code = head, None, None
    env: dict[str, str] = {}
    m = re.search(re.escape(MARK) + r" END rc=(-?\d+) pwd=(.*)\n?", head)
    if m:
        exit_code = int(m.group(1))
        final_cwd = m.group(2).strip()
        for line in head[m.end():].splitlines():
            # a batch process still writing when the timeout hits (vim, less) can prefix the first ENV line
            i = line.find(f"{MARK} ENV ")
            if i >= 0:
                k, _, v = line[i + len(MARK) + 5:].partition("=")
                if k not in VOLATILE_ENV and not k.startswith("CLEAVE_") and PINNED_ENV.get(k) != v:
                    env[k] = v
        output = head[: m.start()]
    rc_line, _, rest = rest.partition("\n")
    outer_rc = int(rc_line.strip()) if rc_line.strip().lstrip("-").isdigit() else None
    timed_out = outer_rc in (124, 137)
    changed: dict[str, str] = {}
    listing: set[str] = set()
    section = None
    for line in rest.splitlines():
        if line.startswith(MARK):
            section = line[len(MARK):].strip()
            continue
        if section == "CHANGED":
            if line.startswith("L:"):
                tgt, _, p = line[2:].partition("  ")
                changed[p] = "L:" + tgt
            elif line.startswith("D  "):
                changed.setdefault(line[3:], "D")
            elif len(line) > 66 and line[64:66] in ("  ", " *"):
                changed[line[66:]] = line[:64]
        elif section == "LISTING" and line:
            listing.add(line)
    if exit_code is None and not timed_out:
        exit_code = outer_rc
    if output.endswith("\n"):
        output = output[:-1]  # the newline the EXIT trap prints before the marker
    return output, exit_code, final_cwd, changed, listing, timed_out, env


def drop_explained_dirs(changed: dict[str, str], explained_by: set[str] = frozenset()) -> dict[str, str]:
    """Directories whose ctime moved only because something below them changed are not effects on their own.

    A directory is kept only if no changed path below it (file, symlink, directory, or a path in
    `explained_by`) accounts for it. The prefix is taken with the trailing slash stripped, so "/" is
    covered too: with the probe sweeping all of /, the root showed up as a changed directory in every
    run that wrote anything, and a wrong edit shared that free "effect" with the teacher (0.5 + 1 of 2).
    """
    below = set(changed) | set(explained_by)

    def explained(d: str) -> bool:
        pre = d.rstrip("/") + "/"
        return any(p != d and p.startswith(pre) for p in below)

    return {p: h for p, h in changed.items() if h != "D" or not explained(p)}


def execute(world, anchor, cmds_script: str, *, cwd: str, cmd_timeout: int = 30,
            network: bool = False, extra_roots: tuple = (), sweep: bool = True, venv: bool = True) -> Effects:
    """Fork `world` at checkpoint `anchor`, run the batch plus the probe, discard the fork's writes."""
    try:
        op, roots = run_probe(world, anchor, cmds_script, cwd=cwd, cmd_timeout=cmd_timeout, network=network,
                              extra_roots=extra_roots, sweep=sweep, venv=venv)
    except OSError as e:  # H50: a batch larger than the sandbox's argv limit (E2BIG) never ran
        return Effects("", None, None, {}, set(), False, f"sandbox run failed: {e}")
    return effects_from(world, anchor, op, roots, venv=venv)


def run_probe(world, anchor, cmds_script: str, *, cwd: str, cmd_timeout: int = 30, network: bool = False,
              extra_roots: tuple = (), sweep: bool = True, venv: bool = True):
    """The sandbox run behind `execute`: (the run's Op, the watched roots)."""
    roots = [r for r in roots_for(cmds_script, cwd, extra_roots) if r not in GLOBAL_ROOTS]
    probe = build_probe_script(cmds_script, roots, cmd_timeout, sweep=sweep, venv=venv)
    fork = world.fork(anchor)
    try:
        kw = {"keep": False} if _accepts_keep(fork) else {}
        op = fork.run(["bash", "-c", probe], cwd=cwd, timeout_s=cmd_timeout + 60, network=network, **kw) \
            if _accepts_network(fork) else fork.run(["bash", "-c", probe], cwd=cwd, timeout_s=cmd_timeout + 60, **kw)
    finally:
        fork.close()
    return op, roots


def effects_from(world, anchor, op, roots: list[str], *, venv: bool = True) -> Effects:
    """Effects from a probe run. venv=False reads the run as the probe without the H42 VENV section (the section
    only reads, so the rest of the output is the same) and takes the section's time out of `elapsed`."""
    if MARK not in op.stdout:
        return Effects(op.stdout, None, None, {}, set(), op.exit_code == 124,
                       f"probe did not complete (exit {op.exit_code}): {op.stderr[-500:]}", op.duration_s, roots=[])
    output, code, fcwd, changed, listing, cmd_timed_out, env = parse_probe(op.stdout)
    procs = parse_procs(op.stdout)
    g_changed, g_listing, g_complete, _ = parse_global(op.stdout)
    v_changed, v_listing, v_dirs, v_complete, v_s = parse_venv(op.stdout)
    elapsed = op.duration_s
    if not venv:
        v_changed, v_listing, v_dirs, v_complete, elapsed = {}, set(), [], False, op.duration_s - (v_s or 0.0)
    # H29: a file or symlink under the global roots counts only where it differs from the pivot. A reinstall moves
    # the ctime of every file in the copy; counting them all gave a candidate's other edit there half credit as a
    # shared key (csvsum t8 hand-redteam:4, X 0.9333 under H28). The watch's hashes outrank the sweep's.
    # H42: the venv site-packages files go through the same check (a pip reinstall rewrites every file of a package).
    g_dirs = {p: h for p, h in g_changed.items() if h == "D"}
    g_files = {p: h for p, h in g_changed.items() if h != "D"}
    v_files = {p: h for p, h in v_changed.items() if h != "D"}
    changed.update(confirm_outside(world, anchor, {**parse_outside(op.stdout), **g_files, **v_files}))
    changed.update(g_dirs)
    if v_complete:
        changed.update({p: h for p, h in v_changed.items() if h == "D"})
    changed = drop_explained_dirs(changed)
    roots = roots + list(GLOBAL_ROOTS) if g_complete else roots
    cfgs = {p for p in v_listing if p.endswith("/pyvenv.cfg")}
    listing = listing | g_listing | (v_listing if v_complete else cfgs)
    roots = roots + v_dirs if v_complete else roots
    return Effects(output, code, fcwd, changed, listing, cmd_timed_out, None, elapsed, roots=roots,
                   env=env, procs=procs, venv_s=(v_s or 0.0) if venv else 0.0)


def _accepts_keep(world) -> bool:
    import inspect
    return "keep" in inspect.signature(world.run).parameters


def _accepts_network(world) -> bool:
    import inspect
    return "network" in inspect.signature(world.run).parameters


def listing_of(world, anchor, cwd: str, roots: list[str], venv: bool = True) -> set[str]:
    """Pre-state path listing, computed once per pivot."""
    eff = execute(world, anchor, "true\n", cwd=cwd, extra_roots=tuple(roots), sweep=False, venv=venv)
    return eff.listing


def digest(effects: Effects) -> str:
    h = hashlib.sha256()
    for p in sorted(effects.meaningful_changes()):
        h.update(f"{p}\0{effects.changed[p]}\n".encode())
    return h.hexdigest()[:16]
