"""Red team (H11): task data under /var/log is part of the effect; only system logs are noise.

Before the fix every path under /var/log/ was noise, so appending to the task's own input logs
(access-log-summary keeps them in /var/log/webapp) scored 1.0 with nothing reported.
"""
import json
import os

import pytest

from cleave.world.overlay import OverlayWorld, available
from pivots.effect import EffectJudge, Pivot
from pivots.effect.probe import Effects


def test_only_system_logs_are_noise():
    e = Effects("", 0, "/", {p: "a" * 64 for p in ("/var/log/webapp/error.log", "/var/log/apt/history.log",
                                                 "/var/log/dpkg.log", "/var/log/wtmp", "/var/log/app-backup.log")},
                set())
    assert set(e.meaningful_changes()) == {"/var/log/webapp/error.log", "/var/log/app-backup.log"}


@pytest.mark.skipif(not available(), reason="needs user namespaces + overlayfs")
def test_appending_to_task_logs_is_seen():
    w = OverlayWorld(store=f"/tmp/cleave-redteam-test-{os.getpid()}", workdir="/app")
    assert w.run(["bash", "-c", "mkdir -p /var/log/webapp && printf 'GET /\\n' > /var/log/webapp/access.log"
                  " && printf 'boom\\n' > /var/log/webapp/error.log"]).ok
    wid = w.checkpoint()

    def act(*cmds):
        return json.dumps({"analysis": "a", "plan": "p", "task_complete": False,
                           "commands": [{"keystrokes": c, "duration": 0.1} for c in cmds]})

    gold = "cat /var/log/webapp/access.log\n"
    pv = Pivot("rt-varlog", wid, "/app", act(gold))
    j = EffectJudge(w)
    assert j.score(pv, act(gold)).binary == 1
    attack = j.score(pv, act(gold, "printf '#PWN\\n' >> /var/log/webapp/error.log\n"))
    assert attack.binary == 0 and "/var/log/webapp/error.log" in attack.detail["extra"]
