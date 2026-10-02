import json
import os

import pytest

from cleave.world.overlay import OverlayWorld, available
from pivots.gym.core import ExecutedPivotCore

pytestmark = pytest.mark.skipif(not available(), reason="needs user namespaces + overlayfs")


def act(*cmds):
    return json.dumps({"analysis": "a", "plan": "p", "task_complete": False,
                       "commands": [{"keystrokes": c, "duration": 0.1} for c in cmds]})


def test_verify_rewards_effects_and_masks_infra_failures():
    w = OverlayWorld(store=f"/tmp/cleave-gym-test-{os.getpid()}", workdir="/app")
    w.run(["bash", "-c", "echo 'x = 1' > /app/conf.py"])
    anchor = w.checkpoint()
    gold = act("sed -i 's/x = 1/x = 2/' conf.py\n")
    core = ExecutedPivotCore(w, {"u1": {"uuid": "u1", "anchor": anchor, "cwd": "/app"},
                                 "u2": {"uuid": "u2", "anchor": "overlay:missing", "cwd": "/app"}})
    good = core.verify({"uuid": "u1", "expected_answer": gold}, act("echo 'x = 2' > conf.py\n"))
    assert good["reward"] == 1.0 and good["j_reward"] == 0.0 and not good["mask_sample"]
    bad = core.verify({"uuid": "u1", "expected_answer": gold}, act("sed -i 's/x = 1/x = 3/' conf.py\n"))
    assert bad["reward"] == 0.0 and bad["j_reward"] == 1.0
    assert core.verify({"uuid": "nope", "expected_answer": gold}, gold)["mask_sample"]
    assert core.verify({"uuid": "u2", "expected_answer": gold}, gold)["mask_sample"]
