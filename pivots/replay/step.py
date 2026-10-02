"""step.sh: run one Terminus-2 batch on a checkpoint with shell state carried in files."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field

from pivots.effect.terminus import Command, keystrokes_to_script

XP = "/var/lib/xp"  # plan §7.4 says /opt/xp; /var is writable on every backend we run (some hosts mount /opt read-only)
PROMPT_MARK = "\x1e"  # record separator: start of an echoed prompt line, stripped when rendering


def step_script(batch: list[Command], *, host: str = "modal", default_cwd: str = "/app",
                cmd_timeout: int = 60) -> tuple[str, list[str]]:
    """A bash program that replays `batch` with Terminus-2 shell semantics.

    Commands run in the driver's own shell (like the interactive shell Terminus
    types into), so cd, exports and functions persist across commands and, via
    /opt/xp, across checkpoints. A hung command is bounded by the step timeout.
    `cmd_timeout` is kept for callers that size the step timeout from it.
    """
    script, warnings = keystrokes_to_script(batch)
    b64 = base64.b64encode(script.encode()).decode()
    return f"""mkdir -p {XP}
[ -f {XP}/env.sh ] && . {XP}/env.sh 2>/dev/null
cd "$(cat {XP}/cwd 2>/dev/null || echo {default_cwd})" 2>/dev/null || cd {default_cwd} 2>/dev/null || cd /
printf %s {b64} | base64 -d > {XP}/.batch
__save() {{
  rm -f {XP}/.batch
  pwd > {XP}/cwd
  export -p | grep -v -E '^declare -[a-zA-Z]*r' | grep -v -E ' (PWD|OLDPWD|SHLVL|_|CLEAVE_[A-Z_]+)=' > {XP}/env.sh
  printf '{PROMPT_MARK}root@{host}:%s# \\n' "$PWD"
}}
trap __save EXIT
__rc=0
__buf=""
exec 9< {XP}/.batch
while IFS= read -r __line <&9 || [ -n "$__line" ]; do
  __buf="$__buf$__line"$'\\n'
  __chk=$(bash -n <<<"$__buf" 2>&1)
  if [ $? -eq 0 ] && [ -z "$__chk" ]; then  # complete command: parses with no unterminated heredoc warning
    printf '{PROMPT_MARK}root@{host}:%s# %s\\n' "$PWD" "${{__buf%$'\\n'}}"
    eval "$__buf" 2>&1 < /dev/null
    __rc=$?
    __buf=""
  fi
done
exit $__rc
""", warnings


@dataclass
class StepResult:
    index: int
    checkpoint: str
    rendered: str            # prompt-echoed output, recorded-observation shape
    exit_code: int
    duration_s: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReplayResult:
    base: str
    steps: list[StepResult]

    def anchor(self, turn: int) -> str:
        """Checkpoint before expert turn `turn` (0-based): the base for turn 0."""
        return self.base if turn == 0 else self.steps[turn - 1].checkpoint


def render(stdout: str) -> str:
    return stdout.replace(PROMPT_MARK, "")


def replay(world, batches: list[list[Command]], *, host: str = "modal", default_cwd: str = "/app",
           cmd_timeout: int = 60, network: bool = False) -> ReplayResult:
    """Run each batch as one kept step; return the checkpoint after every step.

    The world is advanced in place; checkpoints are what later forks start from.
    """
    base = world.checkpoint()
    steps = []
    for i, batch in enumerate(batches):
        prog, warns = step_script(batch, host=host, default_cwd=default_cwd, cmd_timeout=cmd_timeout)
        kw = {"network": network} if "network" in world.run.__code__.co_varnames else {}
        op = world.run(["bash", "-c", prog], cwd="/", timeout_s=cmd_timeout * max(1, len(batch)) + 60, **kw)
        steps.append(StepResult(i, world.checkpoint(), render(op.stdout), op.exit_code, op.duration_s, warns))
    return ReplayResult(base, steps)
