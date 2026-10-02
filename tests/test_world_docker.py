"""S0 gate: the Docker backend forks a world, and the fork is a different machine.

Positive control: a file written in the fork is visible in the fork.
Negative control: the same file is absent in the parent and in a second fork
from the same checkpoint. Both must hold, or `fork` is a rename, not a fork.
"""
import pytest

from cleave.world.base import WorldId
from cleave.world.docker import DockerWorld, daemon_available, remove_image

IMAGE = "python:3.12-slim"

pytestmark = pytest.mark.skipif(not daemon_available(), reason="docker daemon not running")


def test_fork_is_a_new_machine_from_the_checkpoint():
    parent = DockerWorld(IMAGE)
    made: list[WorldId] = []
    try:
        assert parent.run(["sh", "-c", "echo anchor > /work/anchor.txt"]).ok
        anchor = parent.checkpoint()
        made.append(anchor)

        fork_a = parent.fork(anchor)
        try:
            # inherits the anchor's state
            assert fork_a.run(["cat", "/work/anchor.txt"]).stdout.strip() == "anchor"
            # mutate the fork
            assert fork_a.run(["sh", "-c", "echo mutated > /work/only_in_fork.txt"]).ok
            assert fork_a.run(["test", "-f", "/work/only_in_fork.txt"]).ok

            # the parent never sees it
            assert not parent.run(["test", "-f", "/work/only_in_fork.txt"]).ok

            # a second fork from the same checkpoint never sees it either
            fork_b = parent.fork(anchor)
            try:
                assert not fork_b.run(["test", "-f", "/work/only_in_fork.txt"]).ok
                assert fork_b.run(["cat", "/work/anchor.txt"]).stdout.strip() == "anchor"
            finally:
                fork_b.close()
        finally:
            fork_a.close()
    finally:
        parent.close()
        for wid in made:
            remove_image(wid)


def test_run_reports_exit_code_and_duration():
    with DockerWorld(IMAGE) as w:
        op = w.run(["sh", "-c", "exit 3"])
        assert op.exit_code == 3
        assert not op.ok
        assert op.duration_s > 0
