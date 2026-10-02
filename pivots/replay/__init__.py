"""Replay a recorded Terminus-2 trajectory into a chain of checkpoints (plan §7.4).

Each expert turn becomes one run of `step.sh` semantics on the previous
checkpoint:

    restore env (/opt/xp/env.sh) and cwd (/opt/xp/cwd)
    run the batch, one shell-complete command at a time, echoing the prompt
    save cwd and exported env back into the image, exit with the last rc

so that a checkpoint (files only, on Sandboxes and on overlay alike) still
carries the shell state Terminus-2 would have had. The rendered output uses
the recorded prompt shape (`root@<host>:<cwd># <cmd>`), which lets G1
compare it with the recorded observation segment by segment.
"""
from .step import ReplayResult, StepResult, replay, step_script

__all__ = ["ReplayResult", "StepResult", "replay", "step_script"]
