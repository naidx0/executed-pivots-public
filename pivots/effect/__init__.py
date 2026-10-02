"""Executed pivots, horizon 0: parse Terminus-2 actions, run them in forks, compare effects.

J (jreward): NVIDIA's shipped string reward, ported exactly.
X (xreward): effect equivalence with the expert's batch, from one forked run.
"""
from .jreward import string_reward
from .terminus import parse_action
from .xreward import EffectJudge, ExecScore, Pivot

__all__ = ["EffectJudge", "ExecScore", "Pivot", "parse_action", "string_reward"]
