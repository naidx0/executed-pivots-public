"""Port of the upstream NeMo Gym terminus_judge string-only reward.

Source: NVIDIA-NeMo/Gym resources_servers/terminus_judge/app.py (main, 2026-09).
NVIDIA's terminal_pivot training profile uses it with
string_similarity_threshold=0.9 and enable_llm_judge=false. This is the
baseline an executed reward has to beat.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher

from .terminus import parse_action


def command_similarity(gt: dict, pred: dict) -> float:
    gt_ks = [c.get("keystrokes", "") for c in gt.get("commands", []) if "keystrokes" in c]
    pr_ks = [c.get("keystrokes", "") for c in pred.get("commands", []) if "keystrokes" in c]
    if not gt_ks and not pr_ks:
        return 1.0
    if not gt_ks or not pr_ks:
        return 0.0
    return SequenceMatcher(None, "".join(gt_ks), "".join(pr_ks)).ratio()


def check_task_complete(pred: dict, expected: dict) -> bool:
    if expected.get("task_complete"):
        return bool(pred.get("task_complete"))
    return True


def string_reward(model_text: str, expected_answer: str, threshold: float = 0.9,
                  lenient: bool = False) -> dict:
    """Return {"reward", "similarity", "failure"} exactly as upstream scores it.

    lenient=True accepts JSON wrapped in prose/<think> (upstream does not).
    """
    expected = json.loads(expected_answer)
    pr = parse_action(model_text)
    if pr.action is None or (not lenient and not pr.strict_ok):
        return {"reward": 0.0, "similarity": None, "failure": "model_output_invalid"}
    pred = pr.action.raw
    if not check_task_complete(pred, expected):
        return {"reward": 0.0, "similarity": None, "failure": "task_complete_check_failed"}
    sim = command_similarity(expected, pred)
    return {"reward": 1.0 if sim >= threshold else 0.0, "similarity": sim,
            "failure": None if sim >= threshold else "string_similarity_below_threshold"}
