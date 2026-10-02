"""Local specimen tasks: small terminal tasks with a known environment, an expert
trajectory and a verifier, so every stage can be checked against ground truth.

specimens/<name>/
  task.json          {"name", "instruction", "host", "workdir"}
  setup.sh           builds the initial environment (network off, deterministic)
  trajectory.json    [Terminus-2 action, ...]; the last has task_complete: true
  tests/test.sh      verifier; writes 1 or 0 to /logs/verifier/reward.txt; offline

These play the role the 9 Nemotron-Terminal-Synthetic-Tasks play in the plan
(S2 positive control), for machines that cannot pull those images.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pivots.effect.terminus import Command, parse_action
from pivots.replay import ReplayResult, replay


@dataclass
class Specimen:
    name: str
    path: Path
    instruction: str
    host: str
    workdir: str
    setup: str
    actions: list[str]  # raw Terminus-2 JSON per expert turn

    def batches(self, upto: int | None = None) -> list[list[Command]]:
        return [parse_action(a).action.commands for a in self.actions[:upto]]


def load(path: str | Path) -> Specimen:
    p = Path(path)
    meta = json.loads((p / "task.json").read_text())
    traj = json.loads((p / "trajectory.json").read_text())
    return Specimen(meta["name"], p, meta["instruction"], meta.get("host", "modal"), meta.get("workdir", "/app"),
                    (p / "setup.sh").read_text(), [json.dumps(a, indent=2) for a in traj])


def build(world_factory, sp: Specimen):
    w = world_factory(workdir=sp.workdir)
    op = w.run(["bash", "-c", "set -e\n" + sp.setup], cwd="/")
    if not op.ok:
        raise RuntimeError(f"{sp.name}: setup failed rc={op.exit_code}\n{op.stdout[-800:]}\n{op.stderr[-800:]}")
    return w


def run_expert(world, sp: Specimen, upto: int | None = None) -> ReplayResult:
    return replay(world, sp.batches(upto), host=sp.host, default_cwd=sp.workdir)


def verify(world, checkpoint: str, sp: Specimen, timeout_s: float = 300) -> dict:
    """Run the verifier on a fork of `checkpoint`; the fork's writes are discarded."""
    f = world.fork(checkpoint)
    f.put(sp.path / "tests", "/tests")
    kw = {"keep": False} if "keep" in f.run.__code__.co_varnames else {}
    op = f.run(["bash", "-c", "mkdir -p /logs/verifier; bash /tests/test.sh > /logs/verifier/out.txt 2>&1;"
                " cat /logs/verifier/out.txt | tail -c 3000; echo; echo \"@@REWARD $(cat /logs/verifier/reward.txt"
                " 2>/dev/null)\"; echo \"@@CHECKS $(grep -c '^PASS ' /logs/verifier/out.txt)"
                " $(grep -c '^FAIL ' /logs/verifier/out.txt)\""], cwd=sp.workdir, timeout_s=timeout_s, **kw)
    f.close()
    m = re.search(r"@@REWARD (\d)", op.stdout)
    c = re.search(r"@@CHECKS (\d+) (\d+)", op.stdout)
    return {"reward": int(m.group(1)) if m else None, "tail": op.stdout[-1500:], "exit": op.exit_code,
            "checks_passed": int(c.group(1)) if c else None, "checks_failed": int(c.group(2)) if c else None}


def prompt_for(sp: Specimen, turn: int, observations: list[str], initial_screen: str) -> list[dict]:
    """The Terminus-2 message list a student sees at pivot `turn` (same shape as the released rows)."""
    from pivots.specimen_prompt import TERMINUS2_PREAMBLE
    first = (f"{TERMINUS2_PREAMBLE}\n\nTask Description:\n{sp.instruction}\n\n"
             f"Current terminal state:\nCurrent Terminal Screen:\n{initial_screen}\n")
    msgs = [{"role": "user", "content": first}]
    for i in range(turn):
        msgs.append({"role": "assistant", "content": sp.actions[i]})
        msgs.append({"role": "user", "content": "New Terminal Output:\n" + observations[i]})
    return msgs
