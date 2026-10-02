"""Replay keeps shell state across checkpoints, and G1 separates a faithful rebuild from a wrong one.

Positive control: replaying into a copy of the true environment is admitted.
Negative control: replaying into an environment with one file's bytes changed is not.
"""
import os

import pytest

from cleave.world.overlay import OverlayWorld, available
from pivots.effect.terminus import Command
from pivots.replay import replay
from pivots.replay.fidelity import gate_g1, normalise, segment_similarity

pytestmark = pytest.mark.skipif(not available(), reason="needs user namespaces + overlayfs")
STORE = f"/tmp/cleave-replay-test-{os.getpid()}"

TRUE_ENV = """mkdir -p /app/pipeline /data && cd /app/pipeline
printf '{"crypto": {"iterations": 5000}}' > config.json
printf 'def normalize(x):\\n    pass\\n' > normalizer.py
for i in 1 2 3; do echo "{\\"id\\": $i}" > /data/env_$i.json; done
"""
WRONG_ENV = TRUE_ENV.replace("5000", "7000").replace("pass", "return x")

BATCHES = [
    [Command("find /data -type f | sort\n"), Command("cat /app/pipeline/config.json\n")],
    [Command("cd /app/pipeline\n"), Command("export MODE=strict\n"), Command("cat normalizer.py\n")],
    [Command("echo $MODE $PWD\n"), Command("date\n"),
     Command("cat > normalizer.py <<'EOF'\ndef normalize(x):\n    return dict(x)\nEOF\n")],
    [Command("python3 -c 'import normalizer; print(normalizer.normalize({\"a\": 1}))'\n"), Command("ls -la\n")],
]


def world(setup):
    w = OverlayWorld(store=STORE, workdir="/app")
    assert w.run(["bash", "-c", setup]).ok
    return w


def recorded():
    rr = replay(world(TRUE_ENV), BATCHES, default_cwd="/app")
    return ["New Terminal Output:\n" + s.rendered for s in rr.steps]


def test_shell_state_survives_checkpoints():
    rr = replay(world(TRUE_ENV), BATCHES, default_cwd="/app")
    assert "strict /app/pipeline" in rr.steps[2].rendered  # cd + export from turn 1 visible in turn 2
    assert "{'a': 1}" in rr.steps[3].rendered              # heredoc write from turn 2 visible in turn 3
    assert "root@modal:/app/pipeline# cat normalizer.py" in rr.steps[1].rendered


def test_anchor_forks_see_the_state_before_the_turn():
    w = world(TRUE_ENV)
    rr = replay(w, BATCHES, default_cwd="/app")
    before_write = w.fork(rr.anchor(2))
    after_write = w.fork(rr.anchor(3))
    assert "pass" in before_write.run(["cat", "/app/pipeline/normalizer.py"]).stdout
    assert "dict(x)" in after_write.run(["cat", "/app/pipeline/normalizer.py"]).stdout


def test_g1_admits_faithful_rebuild_and_rejects_wrong_bytes():
    rec = recorded()
    good = replay(world(TRUE_ENV), BATCHES, default_cwd="/app")
    bad = replay(world(WRONG_ENV), BATCHES, default_cwd="/app")
    g_good = gate_g1(rec, [s.rendered for s in good.steps])
    g_bad = gate_g1(rec, [s.rendered for s in bad.steps])
    assert g_good.admitted, g_good.summary()
    assert not g_bad.admitted, g_bad.summary()


def test_normalise_handles_volatile_fields_and_unordered_listings():
    a = "Tue Sep 23 05:12:44 UTC 2026\nroot@modal pid 812\nb.txt\na.txt"
    b = "Wed Sep 24 11:02:01 UTC 2026\nroot@ip-10-0-1-2 pid 99\na.txt\nb.txt"
    assert normalise(a, "ls") == normalise(b, "ls")


def test_truncated_recording_compares_head_and_tail_only():
    rec = "line1\nline2\n[... output limited to 10000 bytes; 5000 interior bytes omitted ...]\nline99\nline100"
    reb = "line1\nline2\n" + "\n".join(f"mid{i}" for i in range(50)) + "\nline99\nline100"
    assert segment_similarity(rec, reb) == 1.0
