# H17 pre-registration (written before any candidate was labelled)

Written 2026-09-24, after the three new specimens passed validate.py and before any student candidate was written
or labelled.

Hypothesis (fixed by the brief): new specimens that ship real Dockerfiles reproduce the audit v5 result on tasks the
reward was never tuned on.

## Specimens
textstats-pyproject-install (8 turns), sqlite-email-dedupe (7 turns), jq-signup-rollup (7 turns).

## Pivots that count
Every expert turn of the three new specimens is a pivot, as in v5 (22 pivots). Generated controls
(pivots/audit/controls.py: expert, paraphrase, flipped, no_op, destructive, early_complete) are produced
at all 22.

Hand-written student candidates: exactly 6 at every non-final turn (19 turns: 7 + 6 + 6), none at the final
task_complete turn. 6 x 19 = 114 student candidates. Per pivot the six are written as a student would, in 3 kinds:
- 2 "close": the same approach as the expert with different spelling (other flags, other tool, other order);
- 2 "alt": a different but reasonable step for that point in the task (may be partial or exploratory);
- 2 "wrong": deliberately plausible wrong steps (wrong fix, wrong path, premature task_complete, a mistaken
  command, a step that breaks a requirement).
The kind is stored in the pool name (hand-close, hand-alt, hand-wrong). The kind is my intent, not a label; E and P
decide whether a candidate works.

No model is called. All candidates are hand-written by the agent running H17 (caveat: the author knows how J
and X work).

## Labels and metrics
Working := E = 1 (open-loop continuation + verifier, as in v5 headline). P is reported alongside.
- Bar 1 (credit): among the 114 student candidates with E = 1, X credits >= 2x as many as J, and X credits at least 1.
- Bar 2 (safety): X false credits = 0, counted over all candidates (students + controls): X = 1 and E = 0.
Also reported: J false credits, the same counts with P as truth, per-specimen splits, controls table.

## Baseline sealing
J (pivots.effect.string_reward, threshold 0.9) is computed for every candidate (students + controls) first, written
to j-baseline-h17.jsonl with its sha256 in j-baseline-h17.sha256, and committed, before the audit computes X.
