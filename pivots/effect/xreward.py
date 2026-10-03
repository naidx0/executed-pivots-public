"""X: the effect-equivalence reward (executed, horizon 0, no verifier, no model).

Instead of asking "does the candidate's text look like the teacher's text?"
(upstream: SequenceMatcher ratio >= 0.9), we run both the teacher action and
the candidate action from the same sandbox snapshot and ask "did they do the
same thing?":

  state score   files created / modified / deleted, compared by content hash
  output score  token-multiset F1 of the normalized terminal output
                (what the agent learns from a read-only command)
  penalties     side effects the teacher did not cause, new errors,
                timeouts, destructive deletes

In the plan's vocabulary: J is the shipped string reward, E_h the executed
reward with teacher continuation and the rebuilt verifier. X sits between
them: it executes, but only the one batch, and compares effects with the
expert's batch. It costs one sandbox run per action and no model calls, so it
can label every sampled action while E_h labels a seeded fraction.

The teacher's effects and the pre-state listing are computed once per pivot
and reused for every GRPO sample, so a group of G candidates costs G+2
sandbox runs (+1 for the determinism check).
"""

from __future__ import annotations

import base64
import configparser
import json
import re
import shlex
import threading
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace

from .jreward import check_task_complete
from .probe import DEFAULT_ROOTS, NOISE_RE, Effects, execute, listing_of
from .terminus import Command, keystrokes_to_script, parse_action

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11: the key-set fallback then leaves TOML at X=0
    tomllib = None

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\r")
VOLATILE = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?\b"), "<TS>"),
    (re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+(\d{2}:\d{2}|\d{4})\b"), "<DATE>"),
    (re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b"), "<TIME>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<ADDR>"),
    (re.compile(r"\b[0-9a-f]{12,64}\b"), "<HEX>"),
    (re.compile(r"\b\d+(\.\d+)?\s?(ms|s|sec|seconds|µs|us)\b"), "<DUR>"),
    (re.compile(r"/tmp/tmp[\w-]+"), "/tmp/<TMP>"),
    (re.compile(r"\bpid[ =:]?\d+\b", re.I), "pid <PID>"),
    (re.compile(r"(?<=@)[0-9a-f]{12}\b"), "<HOST>"),
]
SHA256_RE = re.compile(r"[0-9a-f]{64}")
# generated outputs of a build or an install: they follow from the sources, so matching them is no evidence that
# the candidate made the source edit that produces them reproducibly (H17: 8 matching build files hid a missed
# pyproject.toml fix, X 0.867 with E=0)
ARTIFACT_DIRS = {"build", "dist", ".eggs", "site-packages", "dist-packages", "__pycache__", "node_modules"}
ARTIFACT_DIR_SUFFIXES = (".egg-info", ".dist-info")
ARTIFACT_SUFFIXES = (".pyc", ".pyo", ".o", ".a", ".so", ".class")


def is_artifact(path: str) -> bool:
    parts = [x for x in path.split("/") if x]
    return (any(x in ARTIFACT_DIRS or x.endswith(ARTIFACT_DIR_SUFFIXES) for x in parts[:-1])
            or bool(parts) and parts[-1].endswith(ARTIFACT_SUFFIXES))


def source_files(ref: Effects, pre_listing: set[str], created: frozenset[str] | set[str] = frozenset()) -> dict[str, str]:
    """Files the expert's step edited directly: regular files that existed at the pivot and are not generated
    artifacts, plus (H42) the paths in `created` (see created_files). A candidate must leave each with the
    expert's content (see EffectJudge.origin_gate)."""
    return {p: h for p, h in ref.meaningful_changes().items()
            if SHA256_RE.fullmatch(h) and (p in pre_listing or p in created) and not is_artifact(p)}


def created_files(ref: Effects, ref2: Effects, pre_listing: set[str]) -> frozenset[str]:
    """H42: regular files the expert's step wrote that did not exist at the pivot, are not generated artifacts, and
    have the same bytes in both of the expert's runs (the determinism check), so a file with volatile content (a
    timestamped log) stays out. The H20 gate covered only files that existed at the pivot: a file the step creates
    earned half credit for wrong bytes, averaged over every other key. The H41 acme-worker fixture (#21) processes
    the jobs twice: report.txt, created by the step, holds every line twice, and 9 matching keys of 10 carried it
    to 0.9378."""
    r2 = ref2.meaningful_changes()
    return frozenset(p for p, h in ref.meaningful_changes().items()
                     if SHA256_RE.fullmatch(h) and p not in pre_listing and not is_artifact(p) and r2.get(p) == h)


# H21: the line between a batch and what runs after it in the same shell (the expert's next batch)
NEXT_MARK = "@@H21-NEXT-5d2e@@"
KEYSET_SUFFIXES = (".toml", ".json", ".ini")


def _flat_keys(obj, prefix: str = "") -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out |= {key} | _flat_keys(v, key)
    return out


def key_set(path: str, text: str) -> set[str] | None:
    """The key paths of a TOML/JSON/INI file (dotted; lists are leaves), or None when it does not parse."""
    try:
        if path.endswith(".toml"):
            return _flat_keys(tomllib.loads(text)) if tomllib else None
        if path.endswith(".json"):
            return _flat_keys(json.loads(text))
        if path.endswith(".ini"):
            cp = configparser.ConfigParser(interpolation=None)
            cp.read_string(text)
            return set(cp.sections()) | {f"{s}.{k}" for s in cp.sections() for k in cp[s]}
    except Exception:  # noqa: BLE001 - a file that does not parse keeps X=0
        return None
    return None


def is_bare_claim(raw: dict | None) -> bool:
    """A Terminus action that claims completion and runs nothing (task_complete, no non-blank keystrokes)."""
    if not isinstance(raw, dict) or raw.get("task_complete") is not True:
        return False
    cmds = raw.get("commands") or []
    return not any(isinstance(c, dict) and str(c.get("keystrokes", "")).strip() for c in cmds)


# H39 (H36 Rule C): the first line of the expert's next batch is `cd /abs...`, which sets the cwd whatever it was
_ABS_CD_RE = re.compile(r"""^\s*cd\s+(?:--\s+)?["']?/""")


def next_resets_cwd(next_answer: str | None) -> bool:
    """True when the expert's next recorded action makes the cwd after this step irrelevant: it is a bare claim
    (the episode ends), or its batch's first line is a `cd` to an absolute path. Unknown next step: False."""
    if next_answer is None:
        return False
    try:
        raw = json.loads(next_answer)
    except (TypeError, ValueError):
        return False
    if is_bare_claim(raw):
        return True
    act = parse_action(next_answer).action
    if act is None:
        return False
    script, _ = keystrokes_to_script(act.commands)
    first = next((ln for ln in script.splitlines() if ln.strip()), "")
    return bool(_ABS_CD_RE.match(first))


ERROR_RE = re.compile(r"command not found|No such file or directory|Traceback \(most recent call last\)|"
                      r"Permission denied|syntax error|SyntaxError|ModuleNotFoundError|cannot access", re.I)


def normalize_output(text: str) -> str:
    t = ANSI_RE.sub("", text or "")
    for rx, rep in VOLATILE:
        t = rx.sub(rep, t)
    return t


def tokens(text: str) -> Counter:
    return Counter(re.findall(r"[\w./<>-]+", normalize_output(text)))


def output_f1(ref: str, cand: str) -> float:
    a, b = tokens(ref), tokens(cand)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    if overlap == 0:
        return 0.0
    p, r = overlap / sum(b.values()), overlap / sum(a.values())
    return 2 * p * r / (p + r)


# H30: at a read-only pivot (the expert changes nothing), what a read teaches is the facts it prints, not its
# layout. A working student who reads the same facts with a wider command (cat for grep, ls -la for ls) prints many
# more tokens, and whole-output F1 fell under the threshold (21 of 37 H24 misses). There, when the candidate also
# changes nothing, the output is scored by the recall of the expert output's informative tokens.
#
# Informative tokens: the distinct tokens of the normalized expert output (tokens(), trailing ".-/" stripped,
# volatile placeholders such as <TS> dropped) that are
#   paths        contain "/" next to a word character, or look like a file name (name.ext, ext 1-5 alnum chars);
#   identifiers  contain "_" or a lower-to-upper case change (snake_case, camelCase), or are dotted names (a.b);
#   numbers      contain a digit and are not a single bare digit (link counts, exit codes and list bullets are noise);
#   error names  end in Error, Exception or Warning, or are errno names (E + 3 or more capitals, ENOENT).
# Plain words (and, root, found) are not informative. Recall 0.6 maps to the threshold; an expert output with no
# informative token keeps the whole-output F1.
READONLY_RECALL_CREDIT = 0.6
_PATH_RE = re.compile(r"\w/|/\w|^[\w-]+\.[A-Za-z0-9]{1,5}$")
_IDENT_RE = re.compile(r"_|[a-z][A-Z]|^[A-Za-z_]\w*(\.[A-Za-z_]\w*)+$")
_NUM_RE = re.compile(r"\d")
_ERRNAME_RE = re.compile(r"(Error|Exception|Warning)$|^E[A-Z]{3,}$")


def _clean_tokens(text: str) -> set[str]:
    out = set()
    for t in tokens(text):
        t = t.rstrip(".-/") if len(t) > 1 else t
        if t and "<" not in t and ">" not in t:
            out.add(t)
    return out


# H35: a numbered read (grep -n, cat -n, nl -ba) prefixes each line with its number ("21:", "22-", "    21<tab>").
# Those numbers are layout, not facts: at csvsum t3 they were 24 of the expert's 32 informative tokens, and a working
# student who printed the same code with sed -n recalled 0.156. When at least half of an output's non-blank lines
# carry such a prefix, the prefixes are stripped (both outputs), and the stripped numbers are dropped from the expert's
# informative set. A "-" or ":" followed by a digit is not a prefix, so dates and times are kept.
_LINENO_RE = re.compile(r"^([ \t]*)(\d+)(?:[:-](?!\d)|\t)")


def _strip_numbered(text: str) -> tuple[str, set[str]]:
    lines = (text or "").splitlines(keepends=True)
    body = [ln for ln in lines if ln.strip() and ln.strip() != "--"]
    hits = [_LINENO_RE.match(ln) for ln in lines]
    if not body or 2 * sum(1 for h in hits if h) < len(body):
        return text or "", set()
    return "".join(ln[h.end():] if h else ln for ln, h in zip(lines, hits)), {h.group(2) for h in hits if h}


def strip_line_numbers(text: str) -> str:
    """The output with grep -n / cat -n / nl line-number prefixes removed (see _LINENO_RE)."""
    return _strip_numbered(text)[0]


def informative_tokens(text: str, linenums: bool = True) -> set[str]:
    """The informative tokens of an output (see the comment above READONLY_RECALL_CREDIT); with linenums, line-number
    prefixes are stripped first and the stripped numbers are not informative (H35)."""
    nums: set[str] = set()
    if linenums:
        text, nums = _strip_numbered(text)
    return {t for t in _clean_tokens(text) if t not in nums and (
            _PATH_RE.search(t) or _IDENT_RE.search(t) or _ERRNAME_RE.search(t)
            or (_NUM_RE.search(t) and not (len(t) == 1 and t.isdigit())))}


def informative_recall(ref: str, cand: str, linenums: bool = True) -> dict | None:
    """Share of the expert output's informative tokens found among the candidate output's tokens, or None when
    the expert output has none."""
    inf = informative_tokens(ref, linenums)
    if not inf:
        return None
    have = _clean_tokens(strip_line_numbers(cand) if linenums else cand)
    hit = sorted(inf & have)
    return {"recall": len(hit) / len(inf), "n": len(inf), "recalled": hit[:20], "missed": sorted(inf - have)[:20]}


def recall_score(recall: float, threshold: float) -> float:
    """Recall mapped onto the reward scale: READONLY_RECALL_CREDIT lands exactly on the threshold, 1.0 on 1.0."""
    c = READONLY_RECALL_CREDIT
    if recall >= c:
        return threshold + (1 - threshold) * (recall - c) / (1 - c)
    return threshold * recall / c


def effect_view(ref: Effects, cand: Effects, pre_listing: set[str]) -> tuple[dict, dict]:
    """The changes to compare. When the teacher changed something real, directories whose ctime moved only
    because a pruned child such as __pycache__ appeared are left out on both sides (a wrong edit shared that
    free entry and scored 0.75 on state). When such a directory is all the teacher changed (it ran python
    and nothing else), it stays: it is the only trace that the candidate ran the same thing, and dropping it
    made those pivots output-only and took credit from 4 working audit actions."""
    rs = ref.structural_changes(pre_listing)
    if rs or ref.deleted_vs(pre_listing):
        return rs, cand.structural_changes(pre_listing)
    return ref.meaningful_changes(), cand.meaningful_changes()


def state_score(ref: Effects, cand: Effects, pre_listing: set[str], modes: bool = True) -> tuple[float, dict]:
    rc, cc = effect_view(ref, cand, pre_listing)
    rd, cd = ref.deleted_vs(pre_listing), cand.deleted_vs(pre_listing)
    # H52: a path both runs touched must end with the same permission bits. Only ever lowers a score: a mode
    # mismatch is a half match, and a directory the view left out (its entries did not change) comes back as one.
    mode_diff = {p for p in set(ref.modes) & set(cand.modes)
                 if ref.modes[p] != cand.modes[p] and not NOISE_RE.search(p)} if modes else set()
    keys = set(rc) | set(cc) | rd | cd | mode_diff
    detail = {"ref_changed": len(rc), "cand_changed": len(cc), "ref_deleted": len(rd), "cand_deleted": len(cd),
              "missing": sorted((set(rc) | rd) - (set(cc) | cd))[:20],
              "extra": sorted((set(cc) | cd) - (set(rc) | rd))[:20],
              "destructive_extra": sorted(cd - rd)[:20], "mode_diff": sorted(mode_diff)[:20]}
    if not keys:
        return 1.0, detail
    s = 0.0
    for k in keys:
        if k in mode_diff and not (k in rc and k in cc and rc[k] != cc[k]):
            s += 0.5  # same bytes or an unchanged listing, different permission bits
        elif k in rd and k in cd:
            s += 1
        elif k in rc and k in cc:
            if rc[k] == cc[k]:
                s += 1
            elif rc[k] != "D" and cc[k] != "D":
                s += 0.5  # touched the same file, different content
            else:
                s += 0.75  # same directory modified
    return s / len(keys), detail


def load_references(path) -> dict[str, list[str]]:
    """H44: a reference file, {pivot uuid: [action texts]}.

    JSONL, one verified earlier action per line:
        {"uuid": "<task>:<turn>", "text": "<Terminus-2 JSON action>", "E": 1, "P": 1, "source": "h31#12"}
    uuid is the pivot's (the Gym row uuid); E and P are the action's executed outcome and verified progress labels
    (pivots.gym.labels). Only E=1 and P=1 rows may be references: any other row is refused with ValueError, so an
    unverified action can never widen what X credits. `source` is free text. Duplicate texts at a pivot count once."""
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("E") != 1 or r.get("P") != 1 or not isinstance(r.get("text"), str) or not r.get("uuid"):
                raise ValueError(f"{path}:{n}: a reference needs uuid, text, E=1 and P=1")
            texts = out.setdefault(str(r["uuid"]), [])
            if r["text"] not in texts:
                texts.append(r["text"])
    return out


@dataclass
class Pivot:
    """Everything needed to execute from one pivot point."""
    uuid: str
    anchor: str                  # checkpoint id of the world at the pivot (expert turns 0..t-1 replayed)
    cwd: str                     # shell cwd at the pivot
    expected_answer: str         # teacher's Terminus JSON
    network: bool = False
    cmd_timeout: int = 30
    # H21: the expert's next recorded action (Terminus JSON) and whether this is the expert's last step.
    # With neither (the Gym path has no trajectory), a missed source file keeps X=0 as in H20.
    next_answer: str | None = None
    final: bool = False
    # H49: the expert's later actions (turns t+1..end, Terminus JSON). With them, an early task_complete is checked
    # against the state the expert's whole rest of the episode leaves (EffectJudge.claim_covered); without them (an
    # anchor that did not record them) the premature_complete penalty always applies, as before.
    rest_answers: tuple[str, ...] = ()


@dataclass
class ExecScore:
    reward: float
    binary: float
    state: float | None
    output: float | None
    penalties: dict
    failure: str | None
    detail: dict

    def to_dict(self):
        return asdict(self)


class EffectJudge:
    def __init__(self, world, *, threshold: float = 0.75, state_weight: float = 0.8,
                 check_determinism: bool = True, max_workers: int = 8, sweep: bool = True,
                 origin_gate: bool = True, continuation: bool = False, readonly_recall: bool = True,
                 readonly_linenums: bool = True, cwd_reset: bool = True, created_gate: bool = True,
                 references: dict[str, list[str]] | None = None, min_reference_agreement: float = 1.0,
                 claim_rest: bool = True, mode_gate: bool = True, git_gate: bool = True,
                 effect_cache: bool = False):
        self.world = world  # any ForkableWorld; pivots' anchors are its checkpoints
        self.threshold = threshold
        self.state_weight = state_weight
        self.check_determinism = check_determinism
        self.max_workers = max_workers
        self.sweep = sweep  # probe outside the watched roots and list leftover processes (see probe.build_probe_script)
        # H20: a source file the expert edited must match, or X=0 however many generated artifacts match
        self.origin_gate = origin_gate
        # H21: a missed source file passes if both worlds converge after the expert's next step (see converges).
        # Off by default since H23 (H23, see RESEARCH.md): 7 hand-written wrong edits the next step does not reach
        # converged; the narrowing that stopped them also took all 4 of H21's recoveries, so the H20 gate stands.
        self.continuation = continuation
        # H30: read-only pairs are scored on informative-token recall (see READONLY_RECALL_CREDIT)
        self.readonly_recall = readonly_recall
        # H35: line-number prefixes stripped before that recall (see _LINENO_RE)
        self.readonly_linenums = readonly_linenums
        # H39: no cwd difference in the shell_state penalty when the expert's next step resets the cwd (next_resets_cwd)
        self.cwd_reset = cwd_reset
        # H42: the H20 gate also covers the files the expert's step created, reproducibly (see created_files). It
        # needs the determinism run; without it (check_determinism=False) the gate stays as in H20.
        self.created_gate = created_gate
        # H44: {pivot uuid: [action texts]} of earlier policy actions verified at that pivot (E=1 and P=1, see
        # load_references). Each is run at its pivot like the expert's step (twice, with its own pre-state listing and
        # created set), and a candidate the expert's comparison does not credit is credited when it scores at or
        # above the threshold against any reproducible reference under the same rules. None: expert only.
        self.references = references or {}
        self.min_reference_agreement = min_reference_agreement
        # H49: an early task_complete is not penalised when the candidate's state already covers the state the
        # expert's rest of the episode leaves (see claim_covered)
        self.claim_rest = claim_rest
        # H52: a path both runs touched must end with the same permission bits (see state_score, mode_mismatch)
        self.mode_gate = mode_gate
        # H55: every git repo under the roots must end with the expert's index (`git ls-files -s`), see score
        self.git_gate = git_gate
        # H53: a candidate batch already run at this pivot (same script, cwd and timeout) is not run again; its
        # effects are reused. The expert's two runs are never cached (they measure determinism).
        self.effect_cache = effect_cache
        self._effects: dict[tuple, Future] = {}
        self.effect_cache_hits = 0
        self._cache: dict[str, dict] = {}
        # One lock per pivot: under Gym, the G rollouts of a GRPO group reach /verify together, and without
        # it each of them ran the teacher twice before the first result landed in the cache.
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _run(self, pivot: Pivot, cmds: list[Command]) -> tuple[Effects, list[str]]:
        script, warns = keystrokes_to_script(cmds)
        eff = execute(self.world, pivot.anchor, script or "true\n", cwd=pivot.cwd, cmd_timeout=pivot.cmd_timeout,
                      network=pivot.network, sweep=self.sweep)
        eff.warnings = warns
        return eff, warns

    def _run_candidate(self, pivot: Pivot, cmds: list[Command]) -> tuple[Effects, list[str]]:
        if not self.effect_cache:
            return self._run(pivot, cmds)
        script, warns = keystrokes_to_script(cmds)
        key = (pivot.uuid, pivot.anchor, pivot.cwd, pivot.cmd_timeout, pivot.network, script)
        with self._locks_guard:
            fut = self._effects.get(key)
            owner = fut is None
            if owner:
                fut = self._effects[key] = Future()
            else:
                self.effect_cache_hits += 1
        if owner:
            try:
                eff, _ = self._run(pivot, cmds)
            except BaseException as e:
                with self._locks_guard:
                    self._effects.pop(key, None)
                fut.set_exception(e)
                raise
            if eff.backend_error:  # a failed run is not an effect: the next caller runs it again
                with self._locks_guard:
                    self._effects.pop(key, None)
            fut.set_result(eff)
        eff = fut.result()
        return replace(eff, warnings=list(warns)), warns

    def _run_then(self, pivot: Pivot, cmds: list[Command], tail: str) -> tuple[Effects, str | None]:
        """Run the batch, a marker line, then the script `tail` in the same shell, under one probe. Returns the
        effects and the output after the marker (None when the marker never printed: the batch exited or hung)."""
        script, _ = keystrokes_to_script(cmds)
        eff = execute(self.world, pivot.anchor, f"{script}echo {NEXT_MARK}\n{tail}", cwd=pivot.cwd,
                      cmd_timeout=2 * pivot.cmd_timeout, network=pivot.network, sweep=self.sweep)
        head, found, after = eff.output.partition(NEXT_MARK)
        if not found or eff.backend_error:
            return eff, None
        eff.output = head + after
        return eff, after

    def _pivot_once(self, pivot: Pivot, prep: dict, key: str, build):
        """Compute an expert-side value once per pivot (under the pivot's lock) and keep it in prep."""
        with self._locks_guard:
            lock = self._locks.setdefault(pivot.uuid, threading.Lock())
        with lock:
            if key not in prep:
                prep[key] = build()
            return prep[key]

    def converges(self, pivot: Pivot, prep: dict, cand_cmds: list[Command], missed: list[str]) -> tuple[bool, dict]:
        """H21: may the candidate's missed source files pass? (H21, see RESEARCH.md)

        With the expert's next step: continue both worlds with it; they converge when the next step prints what
        it printed for the expert (token F1 at least the expert's own across two runs) and the continued states
        score at or above the threshold under the comparison without the gate. With no next step: every missed
        file is TOML/JSON/INI and parses to the same key set in both worlds. Otherwise no."""
        ref_cmds = parse_action(pivot.expected_answer).action.commands
        nxt = parse_action(pivot.next_answer).action if pivot.next_answer is not None else None
        exp = prep["expected"]
        if nxt is not None and nxt.commands:
            tail, _ = keystrokes_to_script(nxt.commands)

            def expert_continuation():
                r1, o1 = self._run_then(pivot, ref_cmds, tail)
                r2, o2 = self._run_then(pivot, ref_cmds, tail)
                if o1 is None or o2 is None:
                    return None
                return {"eff": r1, "out": o1, "self_f1": output_f1(o1, o2),
                        "self_reward": self._compare(r1, r2, prep["pre"], exp, exp, gate=False)["reward"]}

            ref = self._pivot_once(pivot, prep, "h21_continuation", expert_continuation)
            if ref is None:
                return False, {"mode": "continuation", "why": "expert_continuation_incomplete"}
            if ref["self_reward"] < self.threshold:
                return False, {"mode": "continuation", "why": "expert_continuation_not_reproducible",
                               "expert_self_reward": round(ref["self_reward"], 4)}
            cand, out = self._run_then(pivot, cand_cmds, tail)
            if out is None:
                return False, {"mode": "continuation", "why": "next_step_not_reached"}
            f1 = output_f1(ref["out"], out)
            st = self._compare(ref["eff"], cand, prep["pre"], exp, exp, gate=False)
            return f1 >= ref["self_f1"] and st["reward"] >= self.threshold, {
                "mode": "continuation", "next_f1": round(f1, 4), "expert_next_f1": round(ref["self_f1"], 4),
                "cont_reward": round(st["reward"], 4), "cont_penalties": st["penalties"]}
        if not (pivot.final or nxt is not None):
            return False, {"why": "next_step_unknown"}
        if not all(p.endswith(KEYSET_SUFFIXES) for p in missed):
            return False, {"mode": "keyset", "why": "not_toml_json_ini"}
        dump = "".join(f"printf '\\n%s %s\\n' {NEXT_MARK}F {shlex.quote(p)}; base64 -w0 {shlex.quote(p)}\n"
                       for p in missed)

        def contents(cmds) -> dict[str, str] | None:
            _, out = self._run_then(pivot, cmds, dump)
            if out is None:
                return None
            files = {}
            for chunk in out.split(f"{NEXT_MARK}F ")[1:]:
                path, _, b64 = chunk.partition("\n")
                try:
                    files[path.strip()] = base64.b64decode(b64.strip(), validate=True).decode("utf-8")
                except ValueError:
                    pass
            return files

        ref_files = self._pivot_once(pivot, prep, "h21_keyset", lambda: contents(ref_cmds))
        cand_files = contents(cand_cmds)
        if ref_files is None or cand_files is None:
            return False, {"mode": "keyset", "why": "dump_not_reached"}
        for p in missed:
            a = key_set(p, ref_files[p]) if p in ref_files else None
            b = key_set(p, cand_files[p]) if p in cand_files else None
            if a is None or b is None or a != b:
                return False, {"mode": "keyset", "why": "key_sets_differ", "file": p}
        return True, {"mode": "keyset"}

    def claim_covered(self, pivot: Pivot, prep: dict, cand: Effects) -> tuple[bool, dict]:
        """H49: may a candidate that claims completion where the expert kept working keep its credit?

        The expert's step and every later step (pivot.rest_answers) run from the pivot in one shell, twice. The
        claim is covered when the candidate's run already holds that final state: every path the rest changed or
        deleted, the candidate changed or deleted too, with the same bytes. Left out: directories (their files
        carry the change), generated artifacts (a later check that ran python), scratch under /tmp and /var/tmp,
        and bytes of files whose content differs between the two rest runs (the file must still exist). Every
        process the rest leaves running (in both runs) must be running after the candidate."""
        if not pivot.rest_answers:
            return False, {"why": "no_rest"}

        def rest_runs():
            script = ""
            for ans in (pivot.expected_answer, *pivot.rest_answers):
                a = parse_action(ans).action
                if a is None:
                    return None
                part, _ = keystrokes_to_script(a.commands)
                script += part
            runs = []
            for _ in range(2):
                eff = execute(self.world, pivot.anchor, script or "true\n", cwd=pivot.cwd,
                              cmd_timeout=pivot.cmd_timeout * (1 + len(pivot.rest_answers)), network=pivot.network,
                              sweep=self.sweep)
                if eff.backend_error or eff.timed_out:
                    return None
                runs.append(eff)
            roots = sorted(set(DEFAULT_ROOTS) | set(runs[0].roots) | {pivot.cwd})
            return {"runs": runs, "pre": listing_of(self.world, pivot.anchor, pivot.cwd, roots)}

        rest = self._pivot_once(pivot, prep, "h49_rest", rest_runs)
        if rest is None:
            return False, {"why": "rest_not_reproducible"}
        r1, r2 = rest["runs"]
        pre = rest["pre"]

        def counted(p: str) -> bool:
            return not (p.startswith(("/tmp/", "/var/tmp/")) or is_artifact(p))

        c1, c2 = r1.meaningful_changes(), r2.meaningful_changes()
        cc = cand.meaningful_changes()
        missing, differ = [], []
        for p, h in c1.items():
            if h == "D" or not counted(p) or p not in c2:
                continue
            if p not in cc:
                missing.append(p)
            elif h == c2[p] and cc[p] != h:
                differ.append(p)
        cand_gone = cand.deleted_vs(pre)
        not_gone = sorted(p for p in r1.deleted_vs(pre) & r2.deleted_vs(pre) if counted(p) and p not in cand_gone)
        procs = sorted(((Counter(r1.procs) & Counter(r2.procs)) - Counter(cand.procs)).elements())
        ok = not (missing or differ or not_gone or procs)
        return ok, {"missing": sorted(missing)[:10], "differ": sorted(differ)[:10], "not_deleted": not_gone[:10],
                    "procs": procs[:5], "rest_steps": 1 + len(pivot.rest_answers)}

    def prepare(self, pivot: Pivot) -> dict:
        prep = self._cache.get(pivot.uuid)
        if prep is not None:
            return prep
        with self._locks_guard:
            lock = self._locks.setdefault(pivot.uuid, threading.Lock())
        with lock:
            prep = self._cache.get(pivot.uuid)
            if prep is None:
                prep = self._prepare(pivot)
                if not prep.pop("_backend_error"):  # a sandbox hiccup must not mask the pivot for good
                    self._cache[pivot.uuid] = prep
            return prep

    def _prepare(self, pivot: Pivot) -> dict:
        return self._record(pivot, pivot.expected_answer)

    def _record(self, pivot: Pivot, answer: str, lenient: bool = False) -> dict:
        """The effects of one reference action at the pivot: the expert's step, or (H44) a verified earlier action."""
        ref_action = parse_action(answer).action
        try:
            expected = json.loads(answer)
        except ValueError:  # H44 references are policy text, parsed leniently; the expert's must be strict JSON
            if not lenient or ref_action is None:
                raise
            expected = ref_action.raw
        ref, _ = self._run(pivot, ref_action.commands)
        roots = sorted(set(DEFAULT_ROOTS) | set(ref.roots) | {pivot.cwd})
        pre = listing_of(self.world, pivot.anchor, pivot.cwd, roots)
        prep = {"expected": expected, "ref": ref, "pre": pre, "self_agreement": None, "created": frozenset(),
                "_backend_error": bool(ref.backend_error)}
        if self.check_determinism:
            ref2, _ = self._run(pivot, ref_action.commands)
            prep["created"] = created_files(ref, ref2, pre)
            prep["self_agreement"] = self._compare(ref, ref2, pre, expected, expected,
                                                   created=self._created(prep))["reward"]
            prep["_backend_error"] = prep["_backend_error"] or bool(ref2.backend_error)
        return prep

    def reference_preps(self, pivot: Pivot, prep: dict) -> list[dict]:
        """H44: the recorded effects of the pivot's verified earlier actions (see `references`), computed once per
        pivot and kept in the expert's prep. A reference identical to the expert's text is left out. A round with a
        sandbox error is returned but not kept, so the next score records it again."""
        texts = [t for t in self.references.get(pivot.uuid, ()) if t != pivot.expected_answer]
        if not texts:
            return []
        with self._locks_guard:
            lock = self._locks.setdefault(pivot.uuid, threading.Lock())
        with lock:
            if "references" in prep:
                return prep["references"]

            def one(text: str) -> dict:
                if parse_action(text).action is None:
                    return {"text": text, "skip": "unparsable"}
                rp = self._record(pivot, text, lenient=True)
                rp["backend_error"] = rp.pop("_backend_error")
                rp["text"] = text
                return rp

            with ThreadPoolExecutor(self.max_workers) as ex:
                refs = list(ex.map(one, texts))
            if not any(r.get("backend_error") for r in refs):
                prep["references"] = refs
            return refs

    def _usable_reference(self, rp: dict) -> bool:
        sa = rp.get("self_agreement")
        return not rp.get("skip") and not rp.get("backend_error") and (sa is None or sa >= self.min_reference_agreement)

    def _created(self, prep: dict) -> frozenset[str]:
        return prep.get("created", frozenset()) if self.created_gate else frozenset()

    def _compare(self, ref: Effects, cand: Effects, pre: set[str], expected: dict, pred: dict,
                 gate: bool | None = None, cwd_reset: bool = False, created: frozenset[str] = frozenset(),
                 early_ok: bool = False) -> dict:
        penalties = {}
        st, detail = state_score(ref, cand, pre, modes=self.mode_gate)
        out = output_f1(ref.output, cand.output)
        ref_has_state = bool(ref.meaningful_changes() or ref.deleted_vs(pre))
        if self.readonly_recall and not ref_has_state and not (cand.meaningful_changes() or cand.deleted_vs(pre)):
            rec = informative_recall(ref.output, cand.output, self.readonly_linenums)
            if rec is not None:
                out = recall_score(rec["recall"], self.threshold)
                detail = {**detail, "readonly_recall": {**rec, "recall": round(rec["recall"], 4),
                                                        "output_f1": round(output_f1(ref.output, cand.output), 4)}}
        r = self.state_weight * st + (1 - self.state_weight) * out if ref_has_state else out
        if detail["extra"]:
            # a path the candidate changed or deleted and the teacher did not. Averaged into the state score
            # alone it only dilutes: with 3+ teacher files, the batch plus a corrupted Makefile or .git/config
            # still scored 0.8 (red team, corrupt_untouched). Read-only pivots had this penalty already.
            penalties["side_effects"] = 0.5
        if detail["destructive_extra"]:
            penalties["destructive"] = 0.3
        if detail["mode_diff"]:
            # H52: a path both runs touched ends with other permission bits. Averaged, one wrong chmod among
            # the expert's four scored 0.8 (H50 tb-processing-pipeline:3, 2 false credits): a gate, like H20
            penalties["mode_mismatch"] = 0.0
        git_diff = sorted(r for r in ref.gitstage if cand.gitstage.get(r) != ref.gitstage[r]) if self.git_gate else []
        if git_diff:
            # H55: the repo's index (staged blobs, modes, unmerged stages) ends elsewhere than the expert's. Writing
            # a merge's files and MERGE_HEAD by hand matched `git merge` file for file (H50 red team #306)
            penalties["git_index_mismatch"] = 0.0
            detail["git_index_diff"] = git_diff[:20]
        ref_err = len(ERROR_RE.findall(ref.output))
        cand_err = len(ERROR_RE.findall(cand.output))
        if cand_err > ref_err:
            penalties["new_errors"] = 0.7
        if cand.timed_out and not ref.timed_out:
            penalties["timeout"] = 0.5
        extra_procs = sorted((Counter(cand.procs) - Counter(ref.procs)).elements())
        if extra_procs:
            # a process the batch left running that the teacher's did not (a detached backdoor, a stray server)
            penalties["background_process"] = 0.5
            detail = {**detail, "extra_procs": extra_procs[:10]}
        shell = {k for k in set(ref.env) | set(cand.env) if ref.env.get(k) != cand.env.get(k)}
        if ref.final_cwd and cand.final_cwd and ref.final_cwd != cand.final_cwd:
            if cwd_reset:
                # H39: the expert's next step sets the cwd itself (absolute cd) or ends the episode; exported
                # variables still count
                detail = {**detail, "cwd_diff_reset_by_next": cand.final_cwd}
            else:
                shell.add(f"cwd:{cand.final_cwd}")
        if shell:
            # the next turn runs in this shell: a missing cd or export breaks the expert's continuation
            penalties["shell_state"] = 0.5
            detail = {**detail, "shell_diff": sorted(shell)[:10]}
        if self.origin_gate if gate is None else gate:
            missed = sorted(p for p, h in source_files(ref, pre, created).items() if cand.changed.get(p) != h)
            if missed:
                # left unchanged, edited differently or deleted: the averaged state score let 8 matching build
                # files carry a missed pyproject.toml fix past the threshold (H17 textstats turn 4)
                penalties["source_missed"] = 0.0
                detail = {**detail, "source_missed": missed[:10]}
        if pred.get("task_complete") and not expected.get("task_complete") and not early_ok:
            # ends the episode where the expert kept working; J's one-way check lets this through. H49: not when
            # the candidate already left the state the expert's rest of the episode ends in (claim_covered)
            penalties["premature_complete"] = 0.0
        for v in penalties.values():
            r *= v
        return {"reward": r, "state": st if ref_has_state else None, "output": out, "penalties": penalties,
                "detail": detail}

    def score(self, pivot: Pivot, model_text: str) -> ExecScore:
        prep = self.prepare(pivot)
        pr = parse_action(model_text)
        if pr.action is None:
            return ExecScore(0.0, 0.0, None, None, {}, "model_output_invalid", {"parse_error": pr.error})
        if not check_task_complete(pr.action.raw, prep["expected"]):
            return ExecScore(0.0, 0.0, None, None, {}, "task_complete_check_failed", {})
        cand, warns = self._run_candidate(pivot, pr.action.commands)
        if cand.backend_error:
            return ExecScore(0.0, 0.0, None, None, {}, "backend_error", {"error": cand.backend_error})
        reset = self.cwd_reset and next_resets_cwd(pivot.next_answer)
        early_ok, claim = False, None
        if self.claim_rest and pr.action.raw.get("task_complete") and not prep["expected"].get("task_complete"):
            early_ok, claim = self.claim_covered(pivot, prep, cand)
        c = self._compare(prep["ref"], cand, prep["pre"], prep["expected"], pr.action.raw, cwd_reset=reset,
                          created=self._created(prep), early_ok=early_ok)
        c["detail"]["x_gated_reward"] = c["reward"]  # H20's X
        if "source_missed" in c["penalties"]:
            u = self._compare(prep["ref"], cand, prep["pre"], prep["expected"], pr.action.raw, gate=False,
                              cwd_reset=reset, early_ok=early_ok)
            c["detail"]["x_ungated_reward"] = u["reward"]  # the X before H20
            if self.continuation and u["reward"] >= self.threshold:
                # H21: the only runs beyond H20's, made only where lifting the gate could change the label
                ok, why = self.converges(pivot, prep, pr.action.commands, c["detail"]["source_missed"])
                c["detail"]["source_converged"] = {"ok": ok, **why}
                if ok:
                    c = {**u, "detail": {**u["detail"], **c["detail"]}}
        else:
            c["detail"]["x_ungated_reward"] = c["reward"]
        if c["reward"] < self.threshold and self.references.get(pivot.uuid):
            # H44: the expert's comparison did not credit it; compare with each verified earlier action under the
            # same rules. The premature-completion check stays against the expert's step.
            c["detail"]["x_expert_reward"] = c["reward"]
            best = None
            for i, rp in enumerate(self.reference_preps(pivot, prep)):
                if not self._usable_reference(rp):
                    continue
                rc = self._compare(rp["ref"], cand, rp["pre"], prep["expected"], pr.action.raw, cwd_reset=reset,
                                   created=self._created(rp), early_ok=early_ok)
                if best is None or rc["reward"] > best[1]["reward"]:
                    best = (i, rc)
            if best is not None:
                c["detail"]["x_reference_best"] = round(best[1]["reward"], 4)
            if best is not None and best[1]["reward"] >= self.threshold:
                keep = {k: c["detail"][k] for k in ("x_gated_reward", "x_ungated_reward", "x_expert_reward",
                                                     "x_reference_best") if k in c["detail"]}
                if "source_missed" in c["detail"]:
                    keep["expert_source_missed"] = c["detail"]["source_missed"]
                c = {**best[1], "detail": {**best[1]["detail"], **keep, "reference": best[0]}}
        if claim is not None:
            c["detail"]["claim_rest"] = {"ok": early_ok, **claim}
        c["detail"].update({"warnings": warns, "strict_json": pr.strict_ok, "self_agreement": prep["self_agreement"],
                            "cand_output_head": cand.output[:400], "cand_exit": cand.exit_code,
                            "elapsed": round(cand.elapsed, 3)})
        return ExecScore(c["reward"], 1.0 if c["reward"] >= self.threshold else 0.0, c["state"], c["output"],
                         c["penalties"], None, c["detail"])

    def score_group(self, pivot: Pivot, texts: list[str]) -> list[ExecScore]:
        self.prepare(pivot)
        with ThreadPoolExecutor(self.max_workers) as ex:
            return list(ex.map(lambda t: self.score(pivot, t), texts))
