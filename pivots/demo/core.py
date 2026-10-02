"""Framework-free core of the demo: fork a pivot, run a visitor's batch, score it with J, X and P.

Every score comes from the code the audit and the Gym server use:
  J  pivots.effect.jreward.string_reward (the port of NeMo Gym's string-only terminus_judge)
  X  pivots.effect.EffectJudge.score (teacher and candidate run from the same checkpoint)
  P  the rule of pivots.audit.progress: verifier checks passing right after the step, against the
     expert's step, and a task_complete claim must be true
The terminal shown for a batch is the step replayed with the harness's semantics (pivots.replay.step), the
same program that built the pivot, so it reads like the expert's.
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from cleave.world import overlay
from cleave.world.overlay import OverlayWorld
from pivots import specimen as S
from pivots.audit.controls import controls
from pivots.audit.run import CONTROL_ORDER, _cwd_at
from pivots.demo.sandbox import HardenedWorld, JailedWorld, Limits
from pivots.effect import EffectJudge, Pivot, string_reward
from pivots.effect.terminus import parse_action
from pivots.replay.step import render, step_script

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TERMINAL_CHARS = 6000

# One click each: the "keystroke is not the effect" cases, all taken from the audit.
EXAMPLES = [
    {"id": "paraphrase", "pivot": "ledger-git-revert:3", "source": "student:opus:6",
     "title": "Different keystrokes, same effect",
     "claim": "J rejects it, X accepts it",
     "blurb": "A student reverts the bad commit with GIT_EDITOR=true instead of --no-edit. "
              "Similarity 0.70, so the string reward pays 0. The revert is made either way."},
    {"id": "flipped", "pivot": "billing-invoice-bugfix:3", "source": "ctrl:flipped",
     "title": "One flag dropped",
     "claim": "J accepts it, X rejects it",
     "blurb": "The teacher's sed -i fix without the -i: it prints the file and changes nothing. "
              "Similarity 0.99, so the string reward pays 1."},
    {"id": "early", "pivot": "billing-invoice-bugfix:3", "source": "ctrl:early_complete",
     "title": "“Done” said too early",
     "claim": "J accepts it, X rejects it",
     "blurb": "The teacher's own keystrokes, plus task_complete: true four turns early. "
              "The string reward only checks completion one way, so it pays 1."},
]


class Busy(RuntimeError):
    """Every sandbox slot is taken and the queue is full, or the wait ran out."""


class BadRequest(ValueError):
    pass


@dataclass
class InputLimits:
    max_keystroke_chars: int = 8000
    max_action_chars: int = 32000
    max_lines: int = 200


@dataclass
class _Spec:
    sp: S.Specimen
    world: OverlayWorld | None = None
    jw: JailedWorld | None = None
    rr: object = None
    judge: EffectJudge | None = None
    cwd: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)      # turn -> (before, after expert's step)
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class DemoCore:
    def __init__(self, store: str, *, limits: Limits | None = None, inputs: InputLimits | None = None,
                 max_concurrent: int = 2, max_queue: int = 6, queue_wait_s: float = 45.0,
                 specimens_dir: str | Path = REPO / "specimens", stored: str | Path = HERE / "stored.json",
                 hardened: bool = False, mask: tuple[str, ...] = (), world_factory=None,
                 pivots: str = "decisive"):
        self.store = os.path.abspath(store)
        # world_factory(workdir=...) -> ForkableWorld replaces the local overlay (H32: a NebiusWorld partial).
        # Such a world is used as is: no setpriv/prlimit jail, no overlay store or gc; it bounds its own runs.
        self.world_factory = world_factory
        self.runner = ("overlay" if world_factory is None
                       else getattr(getattr(world_factory, "func", world_factory), "backend", "custom"))
        self.limits = limits or Limits()
        self.hardened = hardened  # H18: wrap runs in HardenedWorld (tmpfs path masking) instead of JailedWorld
        self.mask = tuple(mask)
        self.inputs = inputs or InputLimits()
        self.data = json.loads(Path(stored).read_text(encoding="utf-8"))
        self.specs: dict[str, _Spec] = {}
        for p in sorted(Path(specimens_dir).iterdir()):
            if (p / "task.json").exists():
                sp = S.load(p)
                self.specs[sp.name] = _Spec(sp)
        self.decisive = [k for k in self.data["decisive"] if k.split(":")[0] in self.specs]
        # which pivots a visitor may open: the decisive ones (default), or every turn of every specimen ("all",
        # e.g. to replay a policy rollout at a non-decisive turn). Warm-up covers the decisive ones either way.
        if pivots not in ("decisive", "all"):
            raise ValueError(f"pivots must be 'decisive' or 'all', not {pivots!r}")
        self.openable = (self.decisive if pivots == "decisive" else
                         [f"{name}:{t}" for name, st in self.specs.items() for t in range(len(st.sp.actions))])
        # concurrency: `max_concurrent` visitor runs at once, `max_queue` more may wait; gc runs only when idle
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._state = threading.Condition()
        self._waiting = 0
        self._active = 0
        self._gc_running = False
        self.max_queue = max_queue
        self.queue_wait_s = queue_wait_s
        self._scratch_checkpoints: set[str] = set()

    # -- pivots --------------------------------------------------------------

    def pivot_list(self) -> list[dict]:
        out = []
        for key in self.openable:
            task, turn = key.rsplit(":", 1)
            sp = self.specs[task].sp
            out.append({"id": key, "task": task, "turn": int(turn), "turns": len(sp.actions)})
        return out

    def _split(self, key: str) -> tuple[_Spec, int]:
        if key not in self.openable:
            raise BadRequest(f"unknown pivot {key!r}; pick one of /api/pivots")
        task, turn = key.rsplit(":", 1)
        return self.specs[task], int(turn)

    def _ready(self, st: _Spec) -> _Spec:
        """Build the specimen's world and replay the expert once (about 2 to 5 s), then keep it."""
        with st.lock:
            if st.rr is None:
                try:  # a failed build is retried on the next request; st.error only reports the last failure
                    w = S.build(self.world_factory or functools.partial(OverlayWorld, store=self.store), st.sp)
                    st.rr = S.run_expert(w, st.sp)
                except Exception as e:  # noqa: BLE001 - reported to the visitor as a 503
                    st.error = f"{st.sp.name}: could not build the world: {type(e).__name__}: {e}"
                    raise RuntimeError(st.error) from e
                st.world, st.error = w, None
                if self.world_factory is not None:
                    st.jw = w
                else:
                    st.jw = (HardenedWorld(w, self.limits, self.mask) if self.hardened
                             else JailedWorld(w, self.limits))
                st.judge = EffectJudge(st.jw, check_determinism=True)
        return st

    def _pivot(self, st: _Spec, turn: int) -> Pivot:
        if turn not in st.cwd:
            st.cwd[turn] = st.sp.workdir if turn == 0 else _cwd_at(st.jw, st.rr.anchor(turn), st.sp.workdir)
        return Pivot(f"{st.sp.name}:{turn}", st.rr.anchor(turn), st.cwd[turn], st.sp.actions[turn],
                     network=False, cmd_timeout=self.limits.batch_timeout_s,
                     next_answer=st.sp.actions[turn + 1] if turn + 1 < len(st.sp.actions) else None,
                     final=turn + 1 == len(st.sp.actions))

    def candidates(self, key: str) -> list[dict]:
        st, turn = self._split(key)
        gold = st.sp.actions[turn]
        audit = self.data["audit"].get(key, {})
        ctrl = controls(gold, workdir=st.sp.workdir)
        out = []
        for name in [*CONTROL_ORDER, *sorted(set(ctrl) - set(CONTROL_ORDER))]:
            if name in ctrl:
                out.append(self._cand(f"ctrl:{name}", "control", name.replace("_", " "), ctrl[name],
                                      audit.get(f"ctrl:{name}")))
        seen: dict[str, int] = {}
        for s in self.data["students"].get(key, []):
            pool = s["id"].split(":")[1]
            seen[pool] = seen.get(pool, 0) + 1
            out.append(self._cand(s["id"], "student", f"{pool} sample {seen[pool]}", s["text"], audit.get(s["id"])))
        return out

    @staticmethod
    def _cand(cid: str, kind: str, label: str, text: str, audit: dict | None) -> dict:
        pa = parse_action(text)
        a = pa.action
        return {"id": cid, "kind": kind, "label": label, "text": text,
                "keystrokes": "".join(a.keystrokes()) if a else None,
                "task_complete": bool(a.task_complete) if a else False,
                "strict_json": pa.strict_ok, "audit": audit}

    def pivot_detail(self, key: str) -> dict:
        st, turn = self._split(key)
        if st.rr is None:
            self._acquire()  # building writes layers that an idle gc must not see half made
            try:
                self._ready(st)
            finally:
                self._release()
        sp, rr = st.sp, st.rr
        gold = parse_action(sp.actions[turn]).action
        return {"id": key, "task": sp.name, "turn": turn, "turns": len(sp.actions), "instruction": sp.instruction,
                "before": _tail(rr.steps[turn - 1].rendered if turn else f"root@{sp.host}:{sp.workdir}# "),
                "expert": {"keystrokes": "".join(gold.keystrokes()), "task_complete": gold.task_complete,
                           "commands": gold.keystrokes(), "text": sp.actions[turn]},
                "expert_after": _tail(rr.steps[turn].rendered),
                "candidates": self.candidates(key)}

    # -- a run ---------------------------------------------------------------

    def action_text(self, key: str, *, source: str | None = None, keystrokes: str | None = None,
                    task_complete: bool | None = None, action: str | None = None) -> str:
        """The Terminus-2 JSON that gets scored: a stored candidate as is, raw JSON, or an edited batch."""
        lim = self.inputs
        if action is not None:
            if len(action) > lim.max_action_chars:
                raise BadRequest(f"action is {len(action)} characters; the limit is {lim.max_action_chars}")
            return action
        if keystrokes is None:
            if not source:
                raise BadRequest("send keystrokes, action, or a stored candidate's source id")
            for c in self.candidates(key):
                if c["id"] == source:
                    return c["text"]
            raise BadRequest(f"no stored candidate {source!r} at {key}")
        ks = keystrokes.replace("\r\n", "\n").replace("\r", "\n")
        if len(ks) > lim.max_keystroke_chars:
            raise BadRequest(f"the batch is {len(ks)} characters; the limit is {lim.max_keystroke_chars}")
        if ks.count("\n") > lim.max_lines:
            raise BadRequest(f"the batch has more than {lim.max_lines} lines")
        if "\x00" in ks:
            raise BadRequest("the batch contains a NUL byte")
        if ks.strip() and not ks.endswith("\n"):
            ks += "\n"  # the textarea's last line is submitted, as if Enter was pressed
        return json.dumps({"analysis": "", "plan": "",
                           "commands": [{"keystrokes": ks, "duration": 1.0}] if ks.strip() else [],
                           "task_complete": bool(task_complete)})

    def run(self, key: str, text: str, *, with_p: bool = True) -> dict:
        st, turn = self._split(key)
        self._acquire()
        try:
            t0 = time.perf_counter()
            self._ready(st)
            pv = self._pivot(st, turn)
            j = string_reward(text, st.sp.actions[turn])
            result: dict = {}

            def do_x():
                result["X"] = self._x(st, pv, text)

            def do_p():
                result["P"] = self._p(st, turn, text, with_p)

            tx = threading.Thread(target=do_x)
            tx.start()
            try:
                do_p()
            finally:
                tx.join()
            pa = parse_action(text)
            return {"pivot": key, "text": text, "parsed": pa.action is not None, "parse_error": pa.error,
                    "J": {"reward": int(j["reward"]), "similarity": j["similarity"], "failure": j["failure"],
                          "threshold": 0.9},
                    "X": result["X"], "P": result["P"]["P"],
                    "terminal": {"expert": _tail(st.rr.steps[turn].rendered), **result["P"]["terminal"]},
                    "elapsed_s": round(time.perf_counter() - t0, 2)}
        finally:
            self._release()

    def _x(self, st: _Spec, pv: Pivot, text: str) -> dict:
        prep = st.judge.prepare(pv)
        s = st.judge.score(pv, text)
        ref, pre = prep["ref"], prep["pre"]
        ref_changed = {p: ("created" if p not in pre else "modified") for p in ref.meaningful_changes()}
        ref_changed.update({p: "deleted" for p in ref.deleted_vs(pre)})
        d = s.detail
        mine = {}
        if s.failure is None:
            missing, extra = set(d.get("missing", [])), set(d.get("extra", []))
            destructive = set(d.get("destructive_extra", []))
            for p, kind in ref_changed.items():
                if p not in missing:
                    mine[p] = kind
            for p in extra:
                mine[p] = "deleted" if p in destructive else ("created" if p not in pre else "modified")
        return {"reward": int(s.binary), "score": round(s.reward, 4), "threshold": st.judge.threshold,
                "state": None if s.state is None else round(s.state, 4),
                "output": None if s.output is None else round(s.output, 4),
                "penalties": s.penalties, "failure": s.failure,
                "expert_changed": [{"path": p, "kind": k} for p, k in sorted(ref_changed.items())],
                "your_changed": [{"path": p, "kind": k} for p, k in sorted(mine.items())],
                "missing": d.get("missing", []), "extra": d.get("extra", []),
                "destructive": d.get("destructive_extra", []), "shell_diff": d.get("shell_diff", []),
                "self_agreement": prep["self_agreement"], "warnings": d.get("warnings", []),
                "timed_out": "timeout" in s.penalties, "exit": d.get("cand_exit"), "parse_error": d.get("parse_error")}

    def _checks(self, st: _Spec, turn: int) -> tuple:
        if turn not in st.checks:
            before = S.verify(st.jw, st.rr.anchor(turn), st.sp)["checks_passed"]
            expert = S.verify(st.jw, st.rr.steps[turn].checkpoint, st.sp)["checks_passed"]
            st.checks[turn] = (before, expert)
        return st.checks[turn]

    def _p(self, st: _Spec, turn: int, text: str, with_p: bool) -> dict:
        """Run the batch as the harness would, show its terminal, and (with_p) label P like pivots.audit.progress."""
        sp = st.sp
        pa = parse_action(text)
        if pa.action is None:
            return {"P": None, "terminal": {"yours": "", "your_exit": None, "timed_out": False}}
        f = st.jw.fork(st.rr.anchor(turn))
        prog, _ = step_script(pa.action.commands, host=sp.host, default_cwd=sp.workdir)
        op = f.run(["bash", "-c", prog], cwd="/", timeout_s=self.limits.batch_timeout_s + 5)
        timed_out = op.exit_code == 124 and "timeout after" in op.stderr
        term = {"yours": _tail(render(op.stdout)), "your_exit": op.exit_code, "timed_out": timed_out}
        if not with_p:
            f.close()
            return {"P": None, "terminal": term}
        # a batch that hangs (vim, a server in the foreground) is killed at the timeout and the verifier
        # reads the state it left, as pivots.audit.progress does
        before, expert = self._checks(st, turn)
        cp = f.checkpoint()
        self._scratch_checkpoints.add(cp)
        try:
            v = S.verify(st.jw, cp, sp)
        finally:
            if self.runner == "overlay":
                _forget_checkpoint(self.store, cp)
            self._scratch_checkpoints.discard(cp)
            f.close()
        after = v["checks_passed"]
        complete = bool(pa.action.task_complete)
        ok = after is not None and expert is not None and after >= expert
        if complete and v["reward"] != 1:
            ok = False
        return {"P": {"P": int(ok), "before": before, "expert": expert, "after": after,
                      "harm": int(after is not None and before is not None and after < before),
                      "claims_complete": complete, "task_passes": v["reward"] == 1, "timed_out": timed_out},
                "terminal": term}

    # -- concurrency and cleanup ----------------------------------------------

    def _acquire(self) -> None:
        with self._state:
            if self._waiting >= self.max_queue:
                raise Busy("the demo is busy; try again in a few seconds")
            self._waiting += 1
        try:
            if not self._slots.acquire(timeout=self.queue_wait_s):
                raise Busy("the demo is busy; try again in a few seconds")
        finally:
            with self._state:
                self._waiting -= 1
        with self._state:
            while self._gc_running:
                self._state.wait()
            self._active += 1

    def _release(self) -> None:
        with self._state:
            self._active -= 1
            idle = self._active == 0 and not self._gc_running and self.runner == "overlay"
            if idle:
                self._gc_running = True
        self._slots.release()
        if idle:
            try:
                overlay.gc(self.store)
            finally:
                with self._state:
                    self._gc_running = False
                    self._state.notify_all()

    def warm(self, keys: list[str] | None = None) -> None:
        """Build every specimen and prepare the pivots' teacher runs (startup, in the background)."""
        for key in keys or self.decisive:
            st, turn = self._split(key)
            self._acquire()
            try:
                self._ready(st)
                st.judge.prepare(self._pivot(st, turn))
                self._checks(st, turn)
            except Exception:  # noqa: BLE001 - surfaced per request instead
                pass
            finally:
                self._release()

    def status(self) -> dict:
        return {"built": sorted(k for k, s in self.specs.items() if s.rr is not None),
                "errors": {k: s.error for k, s in self.specs.items() if s.error},
                "active": self._active, "waiting": self._waiting,
                "limits": {"batch_timeout_s": self.limits.batch_timeout_s,
                           "max_keystroke_chars": self.inputs.max_keystroke_chars,
                           "max_action_chars": self.inputs.max_action_chars}}


def _tail(text: str, n: int = TERMINAL_CHARS) -> str:
    return text if len(text) <= n else "[...]\n" + text[-n:]


def _forget_checkpoint(store: str, wid: str) -> None:
    """Unregister a scratch checkpoint so gc can reclaim its layers."""
    st = overlay._store(store)
    st.checkpoints.pop(wid, None)
    try:
        os.remove(os.path.join(st.root, "checkpoints", wid.split(":", 1)[1]))
    except OSError:
        pass
