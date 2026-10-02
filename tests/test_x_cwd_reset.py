"""H39 (H36 Rule C): no cwd part in the shell_state penalty when the expert's next step resets the cwd.

The shell_state penalty (x0.5) exists because the next turn runs in this shell: a missing cd breaks the expert's
continuation. When the expert's next recorded batch begins with a cd to an absolute path, or is a bare completion
claim (the episode ends), the cwd this step leaves behind cannot matter, so a cwd difference no longer counts.
Exported variables still count, and every other penalty stays.

Pure unit test: probe.execute is replaced by a small model of a project, no Linux needed.
"""
import hashlib
import json

import pytest

from pivots.effect import EffectJudge, Pivot
from pivots.effect import xreward
from pivots.effect.probe import Effects
from pivots.gym.core import ExecutedPivotCore

OUT = "/app/reports/out.csv"
PRE = {"/app", "/app/run.py", "/app/reports", OUT}


def h(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def fake_execute(world, anchor, script, *, cwd, cmd_timeout=30, network=False, sweep=True, extra_roots=()):
    out, changed, listing, env = "", {}, set(PRE), {}
    for line in script.splitlines():
        for part in line.split("&&"):
            part = part.strip()
            if part.startswith("cd "):
                cwd = part.split()[1]
            elif "python3" in part and "run.py" in part:
                changed[OUT] = h("good")
                out += "wrote 3 rows to reports/out.csv\n"
            elif part.startswith("rm -f"):
                listing.discard(OUT)
            elif part.startswith("export "):
                k, _, v = part[len("export "):].partition("=")
                env[k] = v
    return Effects(out, 0, cwd, changed, listing, roots=["/app"], env=env)


@pytest.fixture(autouse=True)
def fake_sandbox(monkeypatch):
    monkeypatch.setattr(xreward, "execute", fake_execute)
    monkeypatch.setattr(xreward, "listing_of", lambda world, anchor, cwd, roots: set(PRE))


def act(*cmds, tc=False):
    return json.dumps({"analysis": "a", "plan": "p", "task_complete": tc,
                       "commands": [{"keystrokes": c, "duration": 0.5} for c in cmds]})


RUN = act("python3 run.py\n")                              # the expert's step, from /app
RUN_ELSEWHERE = act("cd /app/reports\n", "python3 /app/run.py\n")  # same effect, leaves the shell in /app/reports
BARE = act(tc=True)
ABS_CD = act("cd /app && ls\n")


def pivot(next_answer=None):
    return Pivot("p", "anchor", "/app", RUN, next_answer=next_answer)


def judge():
    return EffectJudge(world=None, check_determinism=False)


@pytest.mark.parametrize("nxt", [ABS_CD, BARE, act("cd '/app/reports'\n", "ls\n"), act("  cd -- /tmp\n")],
                         ids=["abs_cd", "bare_claim", "quoted_abs_cd", "cd_dashdash"])
def test_cwd_difference_is_not_penalised_when_the_next_step_resets_the_cwd(nxt):
    s = judge().score(pivot(nxt), RUN_ELSEWHERE)
    assert s.binary == 1.0 and s.reward == pytest.approx(1.0)
    assert "shell_state" not in s.penalties and s.detail["cwd_diff_reset_by_next"] == "/app/reports"


@pytest.mark.parametrize("nxt", [None, act("cd reports\n"), act("ls\n", "cd /app\n"), act("cd ~\n"),
                                 act("cd $HOME\n"), act("python3 run.py\n", tc=True), "not json"],
                         ids=["unknown", "relative_cd", "cd_not_first", "tilde", "var", "claim_with_commands",
                              "unparsable"])
def test_cwd_difference_keeps_the_penalty_otherwise(nxt):
    s = judge().score(pivot(nxt), RUN_ELSEWHERE)
    assert s.binary == 0.0 and s.penalties.get("shell_state") == 0.5
    assert s.detail["shell_diff"] == ["cwd:/app/reports"]


def test_the_change_can_be_turned_off():
    j = judge()
    j.cwd_reset = False
    s = j.score(pivot(ABS_CD), RUN_ELSEWHERE)
    assert s.binary == 0.0 and s.penalties.get("shell_state") == 0.5


def test_an_exported_variable_still_draws_the_penalty_before_an_absolute_cd():
    s = judge().score(pivot(ABS_CD), act("cd /app/reports && export MODE=fast\n", "python3 /app/run.py\n"))
    assert s.binary == 0.0 and s.penalties.get("shell_state") == 0.5
    assert s.detail["shell_diff"] == ["MODE"]


@pytest.mark.parametrize("cand", [act("cd /app/reports && rm -f /app/reports/out.csv\n"), act("cd /app/reports\n")],
                         ids=["cd_plus_delete", "cd_only"])
def test_a_cd_with_a_wrong_or_missing_effect_stays_x0_before_an_absolute_cd(cand):
    s = judge().score(pivot(ABS_CD), cand)
    assert s.binary == 0.0 and "shell_state" not in s.penalties


def test_gym_core_reads_the_next_answer_from_the_anchor():
    anchors = {"u": {"uuid": "u", "anchor": "a", "cwd": "/app", "next_answer": ABS_CD},
               "v": {"uuid": "v", "anchor": "a", "cwd": "/app"}}
    core = ExecutedPivotCore(None, anchors)
    assert core.verify({"uuid": "u", "expected_answer": RUN}, RUN_ELSEWHERE)["reward"] == 1.0
    assert core.verify({"uuid": "v", "expected_answer": RUN}, RUN_ELSEWHERE)["reward"] == 0.0
