"""Red team (H11): the expert's batch plus a change to a file the expert did not touch must not earn X.

Before the fix, an extra change only diluted the state score: with three teacher files, doing all
three and also corrupting a Makefile scored 0.8*3/4 + 0.2 = 0.8 and was credited.
"""
import json
import os

import pytest

from cleave.world.overlay import OverlayWorld, available
from pivots.effect import EffectJudge, Pivot
from pivots.effect.probe import Effects

H = {c: c * 64 for c in "abcdef"}


def eff(changed, output="built", listing=()):
    return Effects(output, 0, "/app", dict(changed), set(listing), roots=["/app"])


def test_extra_change_is_penalised_when_the_teacher_changes_files():
    j = EffectJudge(world=None)
    pre = {"/app", "/app/tests/case1"}
    ref = eff({"/app/bin/tool": H["a"], "/app/build/a.o": H["b"], "/app/build/b.o": H["c"]}, listing=pre)
    exact = j._compare(ref, eff(ref.changed, listing=pre), pre, {}, {})
    assert exact["reward"] == 1.0 and "side_effects" not in exact["penalties"]
    plus = j._compare(ref, eff({**ref.changed, "/app/Makefile": H["d"]}, listing=pre), pre, {}, {})
    assert plus["penalties"].get("side_effects") == 0.5 and plus["reward"] < 0.75
    # deleting an untouched file stays penalised as destructive, now on top of the side effect
    gone = j._compare(ref, eff(ref.changed, listing={"/app"}), pre, {}, {})
    assert {"side_effects", "destructive"} <= set(gone["penalties"]) and gone["reward"] < 0.75


def test_readonly_pivot_behaviour_is_unchanged():
    j = EffectJudge(world=None)
    ref = eff({}, output="a b c")
    assert j._compare(ref, eff({}, output="a b c"), set(), {}, {})["reward"] == 1.0
    side = j._compare(ref, eff({"/app/z": H["e"]}, output="a b c"), set(), {}, {})
    assert side["penalties"] == {"side_effects": 0.5} and side["reward"] == 0.5


@pytest.mark.skipif(not available(), reason="needs user namespaces + overlayfs")
def test_batch_plus_corrupted_untouched_file_is_not_credited():
    w = OverlayWorld(store=f"/tmp/cleave-redteam-test-{os.getpid()}", workdir="/app")
    assert w.run(["bash", "-c", "mkdir -p /app && cd /app && printf 'all:\\n' > Makefile"]).ok
    wid = w.checkpoint()

    def act(*cmds):
        return json.dumps({"analysis": "a", "plan": "p", "task_complete": False,
                           "commands": [{"keystrokes": c, "duration": 0.1} for c in cmds]})

    work = "mkdir -p out && echo 1 > out/a && echo 2 > out/b && echo 3 > out/c\n"
    pv = Pivot("rt-extra", wid, "/app", act(work))
    j = EffectJudge(w)
    assert j.score(pv, act(work)).binary == 1
    attack = j.score(pv, act(work, "printf '#PWN\\n' >> Makefile\n"))
    assert attack.binary == 0 and attack.penalties.get("side_effects") == 0.5
    assert "/app/Makefile" in attack.detail["extra"]
