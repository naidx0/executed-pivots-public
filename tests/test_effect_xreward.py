"""X (effect equivalence) vs J (shipped string reward) on the plan's control actions (§7.5).

Each control must land in its expected cell:
  expert a*                     J=1  X=1
  paraphrase (ls -la / ls -al)  J=0  X=1
  flipped near-duplicate        J=1  X=0
  destructive                   J=0  X=0 (with the destructive penalty)
  early task_complete           J=1 by the one-way check; X gates it the same way (both see it)
"""
import json
import os

import pytest

from cleave.world.overlay import OverlayWorld, available
from pivots.effect import EffectJudge, Pivot, string_reward
from pivots.effect.xreward import output_f1

pytestmark = pytest.mark.skipif(not available(), reason="needs user namespaces + overlayfs")

SETUP = """mkdir -p /app/pkg && cd /app && printf 'def add(a, b):\\n    return a - b\\n' > pkg/calc.py
echo keep > notes.txt
"""


def act(*cmds, tc=False):
    return json.dumps({"analysis": "a", "plan": "p", "task_complete": tc,
                       "commands": [{"keystrokes": c, "duration": 0.1} for c in cmds]})


@pytest.fixture(scope="module")
def anchor():
    w = OverlayWorld(store=f"/tmp/cleave-effect-test-{os.getpid()}", workdir="/app")
    assert w.run(["bash", "-c", SETUP]).ok
    return w, w.checkpoint()


def test_edit_controls(anchor):
    w, wid = anchor
    ref = act("sed -i 's/return a - b/return a + b/' pkg/calc.py\n")
    pv = Pivot("p1", wid, "/app", ref)
    j = EffectJudge(w)
    assert j.prepare(pv)["self_agreement"] == 1.0
    cases = {
        "expert": (ref, 1, 1),
        "paraphrase_python": (act("python3 -c \"p='pkg/calc.py';s=open(p).read();"
                                  "open(p,'w').write(s.replace('a - b','a + b'))\"\n"), 0, 1),
        "paraphrase_sed_delim": (act("sed -i 's|a - b|a + b|' /app/pkg/calc.py\n"), 0, 1),
        "flipped_near_duplicate": (act("sed -i 's/return a - b/return a * b/' pkg/calc.py\n"), 1, 0),
        "flipped_dropped_i": (act("sed 's/return a - b/return a + b/' pkg/calc.py\n"), 1, 0),
        "truncating_one_liner": (act("python3 -c \"p='pkg/calc.py';"
                                     "open(p,'w').write(open(p).read().replace('a - b','a + b'))\"\n"), 0, 0),
        "destructive": (act("rm -rf pkg notes.txt\n"), 0, 0),
        "no_op": (act(), 0, 0),
    }
    for name, (text, want_j, want_x) in cases.items():
        got_j = string_reward(text, ref)["reward"]
        got_x = j.score(pv, text).binary
        assert (got_j, got_x) == (want_j, want_x), name


def test_readonly_controls(anchor):
    w, wid = anchor
    pv = Pivot("p2", wid, "/app", act("ls -la\n"))
    j = EffectJudge(w)
    assert j.score(pv, act("ls -al\n")).binary == 1 and string_reward(act("ls -al\n"), act("ls -la\n"))["reward"] == 0
    side = j.score(pv, act("touch z\n", "ls -la\n"))
    assert side.penalties.get("side_effects") and side.binary == 0


def test_early_task_complete_is_gated_like_j(anchor):
    w, wid = anchor
    pv = Pivot("p3", wid, "/app", act("ls\n", tc=True))
    j = EffectJudge(w)
    assert j.score(pv, act("ls\n", tc=False)).failure == "task_complete_check_failed"
    assert j.score(pv, "not json").failure == "model_output_invalid"


def test_output_f1_ignores_volatile_tokens():
    assert output_f1("done in 0.52s at 12:01:02", "done in 1.3s at 13:44:10") == 1.0


def test_shell_state_is_part_of_the_effect():
    w = OverlayWorld(store=f"/tmp/cleave-effect-test-{os.getpid()}", workdir="/app")
    assert w.run(["bash", "-c", "mkdir -p /app/sub /var/lib/xp && echo 'declare -x OUT=\"/app/sub\"' > /var/lib/xp/env.sh"]).ok
    wid = w.checkpoint()
    j = EffectJudge(w)
    # the pivot's exported env is restored before the batch runs
    pv = Pivot("env", wid, "/app", act('echo hi > "$OUT/f"\n'))
    assert j.score(pv, act("echo hi > /app/sub/f\n")).binary == 1
    # a batch that leaves the shell somewhere else is not equivalent, even with identical files
    pv2 = Pivot("cd", wid, "/app", act("cd sub\n", "touch g\n"))
    same_files = j.score(pv2, act("touch sub/g\n"))
    assert same_files.binary == 0 and "shell_state" in same_files.penalties
    assert j.score(pv2, act("cd /app/sub && touch g\n")).binary == 1


def test_concurrent_scores_share_one_teacher_run(anchor):
    # Gym sends a GRPO group's rollouts to /verify together; the teacher must still run only once (+1 check).
    from concurrent.futures import ThreadPoolExecutor

    w, wid = anchor
    ref = act("sed -i 's/return a - b/return a + b/' pkg/calc.py\n")
    pv = Pivot("group", wid, "/app", ref)
    j = EffectJudge(w)
    calls = []
    run = j._run
    j._run = lambda *a, **k: calls.append(1) or run(*a, **k)
    with ThreadPoolExecutor(4) as ex:
        got = list(ex.map(lambda _: j.score(pv, ref).binary, range(4)))
    assert got == [1.0] * 4
    assert len(calls) == 2 + 4  # teacher + determinism rerun, then one run per candidate


def test_teacher_backend_error_is_not_cached(anchor):
    from pivots.effect.probe import Effects

    w, wid = anchor
    ref = act("sed -i 's/return a - b/return a + b/' pkg/calc.py\n")
    pv = Pivot("flaky", wid, "/app", ref)
    j = EffectJudge(w)
    run = j._run
    j._run = lambda *a, **k: (Effects("", None, None, {}, set(), backend_error="sandbox unavailable"), [])
    assert j.prepare(pv)["ref"].backend_error  # reported to the caller (Gym masks the sample) ...
    j._run = run
    assert not j.prepare(pv)["ref"].backend_error  # ... but the next verify retries the teacher
    assert j.score(pv, ref).binary == 1


def test_git_commit_is_reproducible_across_seconds():
    # A commit's hash depends on its timestamp: without a pinned clock the teacher disagreed with itself
    # whenever its runs straddled a second (ledger-git-revert turn 3 was masked at random under Gym).
    import time

    w = OverlayWorld(store=f"/tmp/cleave-effect-test-{os.getpid()}", workdir="/app")
    assert w.run(["bash", "-c", "mkdir -p /app/repo && cd /app/repo && git init -q && git config user.email a@b "
                  "&& git config user.name a && echo a > f && git add f && git commit -qm init"]).ok
    wid = w.checkpoint()
    ref = act("echo b >> f\n", "git commit -qam change\n", "git log --oneline -1\n")
    pv = Pivot("git", wid, "/app/repo", ref)
    j = EffectJudge(w)
    assert j.prepare(pv)["self_agreement"] == 1.0
    time.sleep(1.1)
    s = j.score(pv, ref)
    assert s.reward == 1.0, s
    assert "GIT_COMMITTER_DATE" not in j.prepare(pv)["ref"].env  # the pin is not reported as the batch's export


def test_probe_env_survives_terminal_noise_before_the_marker():
    # vim killed by the timeout wrote its reset sequence ahead of the first ENV line; that export was lost and
    # the candidate took a spurious shell_state penalty (HOME "missing").
    from pivots.effect.probe import MARK, parse_probe

    out = (f"typed\n{MARK} END rc=0 pwd=/app\n\x1b[?1049l\x1b[23;0;0t{MARK} ENV HOME=/root\n{MARK} ENV PATH=/bin\n"
           f"{MARK} RC 124\n{MARK} CHANGED\n{MARK} LISTING\n/app\n{MARK} DONE\n")
    output, _, cwd, _, listing, timed_out, env = parse_probe(out)
    assert env == {"HOME": "/root", "PATH": "/bin"} and cwd == "/app" and timed_out and listing == {"/app"}
