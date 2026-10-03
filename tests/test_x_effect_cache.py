"""H53: the candidate effect cache.

Rollouts repeat themselves: 8.5% to 27% of parsed actions in H43, H45, H47 and H50 run a batch another rollout at
the same pivot already ran. With effect_cache=True the judge runs such a batch once and reuses its effects; the
expert's two runs are never cached, a failed run is not kept, and concurrent callers of one batch wait for the
first run instead of starting their own.

Pure unit test: probe.execute is replaced by a counter, no Linux needed.
"""
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from pivots.effect import EffectJudge, Pivot
from pivots.effect import xreward
from pivots.effect.probe import Effects

CALLS = []
LOCK = threading.Lock()
FAIL = set()


def fake_execute(world, anchor, script, *, cwd, cmd_timeout=30, network=False, sweep=True, extra_roots=()):
    with LOCK:
        CALLS.append(script)
    time.sleep(0.02)
    if script in FAIL:
        return Effects("", None, None, {}, set(), backend_error="boom", roots=[])
    changed = {"/app/out": hashlib.sha256(script.encode()).hexdigest()} if "make" in script else {}
    return Effects("built\n" if "make" in script else "", 0, cwd, changed, {"/app", "/app/out"}, roots=["/app"])


@pytest.fixture(autouse=True)
def fake_sandbox(monkeypatch):
    CALLS.clear()
    FAIL.clear()
    monkeypatch.setattr(xreward, "execute", fake_execute)
    monkeypatch.setattr(xreward, "listing_of", lambda world, anchor, cwd, roots: {"/app"})


def act(cmd, analysis="a"):
    return json.dumps({"analysis": analysis, "plan": "p", "task_complete": False,
                       "commands": [{"keystrokes": cmd + "\n", "duration": 0.1}]})


PV = Pivot("p", "anchor", "/app", act("make"))


def cand_calls():
    return sum(1 for c in CALLS if c.startswith("make") or c.startswith("ls"))


@pytest.mark.parametrize("cache,runs", [(False, 2 + 3), (True, 2 + 1)])
def test_a_repeated_batch_runs_once_with_the_cache(cache, runs):
    j = EffectJudge(world=None, effect_cache=cache)
    scores = [j.score(PV, act("make", analysis=f"take {i}")) for i in range(3)]
    assert cand_calls() == runs  # the expert twice (determinism), then the candidates
    assert len({(s.reward, s.binary) for s in scores}) == 1 and scores[0].binary == 1.0
    assert j.effect_cache_hits == (2 if cache else 0)


def test_different_batches_are_not_shared():
    j = EffectJudge(world=None, effect_cache=True)
    a, b = j.score(PV, act("make")), j.score(PV, act("ls"))
    assert a.binary == 1.0 and b.binary == 0.0 and j.effect_cache_hits == 0


def test_concurrent_callers_wait_for_one_run():
    j = EffectJudge(world=None, effect_cache=True)
    j.prepare(PV)
    before = cand_calls()
    with ThreadPoolExecutor(8) as ex:
        scores = list(ex.map(lambda i: j.score(PV, act("make", analysis=str(i))), range(8)))
    assert cand_calls() - before == 1 and all(s.binary == 1.0 for s in scores)


def test_a_backend_error_is_not_kept():
    j = EffectJudge(world=None, effect_cache=True)
    j.prepare(PV)
    FAIL.add("ls\n")
    assert j.score(PV, act("ls")).failure == "backend_error"
    FAIL.clear()
    assert j.score(PV, act("ls")).failure is None and j.effect_cache_hits == 0
