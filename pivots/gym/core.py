"""Framework-free core of the executed_pivot NeMo Gym resources server (plan §7.6, slice S10).

`verify(row, model_text)` returns the dict a Gym /verify response carries:
reward in {0, 1}, the continuous score, the failure kind, and mask_sample=True
when the environment failed rather than the policy (unknown anchor, sandbox
error, probe did not complete, the expert's own action not reproducible).
Keeping it framework-free lets it be tested without a Gym install and reused
by the audit.

anchors.jsonl maps each released row uuid to the checkpoint to fork:
    {"uuid": "t2_pre_rp_...", "anchor": "<checkpoint id>", "cwd": "/app", "next_answer": "<expert's next action>"}
(next_answer is optional; without it a cwd difference always draws the shell_state penalty.)

references.jsonl (H44, optional) lists earlier policy actions verified at a pivot, one per line:
    {"uuid": "t2_pre_rp_...", "text": "<Terminus-2 JSON action>", "E": 1, "P": 1, "source": "h31#12"}
Only E=1 and P=1 rows are accepted (pivots.effect.xreward.load_references). When the file is present (next to
anchors.jsonl, or at Config.references_path), X credits a candidate that matches the expert's step or any of the
pivot's references under the same rules; without it X compares with the expert's step alone.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from pivots.effect import EffectJudge, Pivot, string_reward
from pivots.effect.xreward import load_references

REFERENCES_FILE = "references.jsonl"  # looked for next to anchors.jsonl when Config.references_path is unset


@dataclass
class Config:
    threshold: float = 0.75
    min_self_agreement: float = 1.0  # mask pivots whose expert action is not reproducible
    cmd_timeout: int = 30
    also_report_j: bool = True
    references_path: str | None = None  # H44 reference file; None: references.jsonl next to the anchors, if present


class ExecutedPivotCore:
    def __init__(self, world, anchors: dict[str, dict], config: Config | None = None,
                 references: dict[str, list[str]] | None = None):
        self.world = world
        self.anchors = anchors
        self.cfg = config or Config()
        self.references = references or {}
        self.judge = EffectJudge(world, threshold=self.cfg.threshold, check_determinism=True,
                                 references=self.references, min_reference_agreement=self.cfg.min_self_agreement)

    @classmethod
    def from_files(cls, world, anchors_path: str, config: Config | None = None) -> "ExecutedPivotCore":
        with open(anchors_path) as fh:
            anchors = {r["uuid"]: r for r in map(json.loads, fh)}
        config = config or Config()
        path = config.references_path or os.path.join(os.path.dirname(os.path.abspath(anchors_path)), REFERENCES_FILE)
        if config.references_path and not os.path.exists(path):
            raise FileNotFoundError(f"executed_pivot: references file {path} not found")
        refs = load_references(path) if os.path.exists(path) else None
        return cls(world, anchors, config, refs)

    def verify(self, row: dict, model_text: str) -> dict:
        uuid = row.get("uuid")
        out = {"reward": 0.0, "score": 0.0, "mask_sample": False, "failure_kind": None, "uuid": uuid}
        if self.cfg.also_report_j:
            out["j_reward"] = string_reward(model_text, row["expected_answer"])["reward"]
        a = self.anchors.get(uuid)
        if a is None:
            return {**out, "mask_sample": True, "failure_kind": "no_anchor"}
        # next_answer: the expert's next action, when prepare.py recorded it (H39 needs it); rest_answers: all of
        # the expert's later actions (H49 checks an early task_complete against them)
        pv = Pivot(uuid, a["anchor"], a.get("cwd", "/app"), row["expected_answer"], cmd_timeout=self.cfg.cmd_timeout,
                   next_answer=a.get("next_answer"), rest_answers=tuple(a.get("rest_answers") or ()))
        try:
            prep = self.judge.prepare(pv)
        except Exception as e:  # sandbox down, checkpoint gone
            return {**out, "mask_sample": True, "failure_kind": f"prepare_error: {type(e).__name__}: {e}"}
        if prep["ref"].backend_error:
            return {**out, "mask_sample": True, "failure_kind": "expert_probe_failed"}
        if prep["self_agreement"] is not None and prep["self_agreement"] < self.cfg.min_self_agreement:
            return {**out, "mask_sample": True, "failure_kind": "expert_not_reproducible"}
        s = self.judge.score(pv, model_text)
        if s.failure == "backend_error":
            return {**out, "mask_sample": True, "failure_kind": "backend_error"}
        return {**out, "reward": s.binary, "score": round(s.reward, 4), "failure_kind": s.failure,
                "penalties": s.penalties}
