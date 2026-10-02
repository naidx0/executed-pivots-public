# Research notes: how Executed Pivots got here

## The problem
Terminal-Pivot (`nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1`) has 31,111 rows from 630 terminal tasks. Each row
cuts an expert session at one turn and asks for the next command batch. The reward that ships with it
(NeMo Gym `terminus_judge`, string-only) pays 1 when the keystrokes are at least 90% similar to the teacher's.
Nothing runs. So it pays a near-copy that does nothing, and it misses a different command that does the same thing.

## The idea
Fork the terminal at the pivot. Run the teacher's batch in one fork and the candidate's in another. Compare what
changed: files, deletions, output, working directory, exported variables, and a `task_complete` claim.
Pay 1 when the effects match. Call that X. Call the shipped keystroke reward J.

## How each claim was tested
One loop, every time:
1. Write the hypothesis and its win condition before running anything.
2. Build the smallest version that tests it.
3. Measure against a sealed baseline: same rows, same seeds. Truth is the task's own verifier (E: replay the
   teacher's remaining turns, then verify; P: verifier checks right after the step).
4. Keep only if it beats the baseline outside noise. Otherwise revert.
5. Log it, kept or discarded.

## What was kept
| # | Result |
|---|---|
| H1 | X pays 94 of 286 working student steps, J pays 26. Failing steps paid: X 0, J 34 (546 actions, audit v5). |
| H7 | The demo runs end to end from a clean checkout, as root and non-root. |
| H8 | Rewards through a real NeMo Gym rollout match the stored labels: J 546/546, X 545/546 (the miss was a stale label, fixed). |
| H11 | Red team: 545 attacks in 16 kinds. X false credit 1 to 0 after fixes; hidden side effects paid 48 of 174 to 0 of 174. |
| H14 | The live demo reproduces 218 of 218 stored labels. |
| H16 | Other task families. Tool use: 122 vs 44 working steps paid, wrong credits 0 vs 2. Code exec: 160 vs 54, wrong credits 0 vs 45. |

## What was discarded (do not retry)
| # | Idea | Why it lost |
|---|---|---|
| H2 to H5 | Carry a model's state into another model with a learned map (StateBridge) | Only shared inherited weights carry state; separately trained models do not. |
| H6 | A "state card" (picture or compact text) instead of the transcript | 29/34 and 28/33 vs 31/34; more steps and more tokens. |
| H10 | X as-is on function calls and patches | Fewer wrong credits, but missed the 2x working-credit bar: byte-different equivalent edits score 0. |
| H12, H13 | Compare edited files by behaviour instead of bytes | Big accuracy gain, but a few new wrong credits each time. |
| H15 | Shape-derived probe inputs | On a fresh set from model students, misses were effect-level, so behaviour matching rarely applied. |

## Local results (H17 to H51)
After H18 the loop moved to a local machine with Docker and Nebius Token Factory. Same rules: win condition first,
sealed baseline, keep or discard.

**Real policies through Nebius Token Factory.** X agrees with the verifier (E) more often than J on every batch, and
X paid no failing step on any of them.

| Batch | Policy | Labelled | Agrees with E: X / J | Working steps paid: X / J | Failing steps paid: X / J |
|---|---|---|---|---|---|
| H31 | NVIDIA Nemotron Nano | 176 | 99 / 83 | 32 / 16 of 109 | 0 / 0 |
| H37b | Nemotron Nano (rules off) | 176 | 105 / 87 | 40 / 22 of 111 | 0 / 0 |
| H40 | NVIDIA Nemotron Super | 176 | 88 / 55 | 60 / 27 of 148 | 0 / 0 |
| H41 | Nemotron Super, 5 new tasks | 94 | 45 / 31 | 28 / 14 of 77 | 0 / 0 |
| H43 | Nemotron Nano, 16 tasks | 270 | 156 / 136 | 46 / 26 of 160 | 0 / 0 |
| H45 | Qwen3-30B (not NVIDIA), multi-reference X | 270 | 172 / 122 | 88 / 40 of 186 | 0 / 2 |

**A second task family: Terminal-Bench (H50).** 12 of Terminal-Bench's easy tasks (third-party, Apache-2.0, own
Dockerfiles and verifiers) ported unchanged. 292 labelled rollouts from a small local model: X agrees with E on 176,
J on 96 (paired sign test p 1.6e-20). X paid 2 failing steps: both `chmod` the right files to the wrong mode, which the
byte-only probe could not see. H52 (compare permission bits) is the fix under test.

**Kept changes to X.**
- H19: each world gets a frozen /var/log. H18: the hardened sandbox runner is the demo default (16 of 16 escapes blocked).
- H20: a source file the expert edited must match. H42: files the expert creates must match too.
- H29, H42: global npm, /usr/local/bin and existing venv site-packages are always watched.
- H30, H38: read-only steps are scored by informative-token recall.
- H39: no cwd penalty when the expert's next step resets the cwd.
- H44, H48: multi-reference X. Verified working actions from earlier batches count as extra references
  (`data/references.jsonl`, 762 actions). Held out on H47: 164 to 189 working steps paid, 0 failing steps paid.
- H46: the live rollout reward equals offline multi-reference X on 20 of 20 rows.
- H49: an early "task complete" claim keeps its credit when the state already matches the end of the expert's episode.

**Discarded locally (do not retry).** H17 (1 false credit, fixed by H20), H21 (next-step convergence: 7 of 9
hand-written attacks credited), H24 to H28 (npm blind spot, closed by H29), H41 (3 fixture false credits, closed by H42),
H47 and H51 (X as a best-of-n ranker does not beat J's similarity score; X's edge is the pass/fail gate).

## Caveats
The students in the audit are Claude models standing in for Nemotron Nano. The six audit tasks were written by us.
X is conservative: a different but useful step scores 0 (192 of 286 working steps go unpaid).
