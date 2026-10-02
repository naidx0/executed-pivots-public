# H16: executed effect vs keystroke match on crucible tool_use and code_exec

**Verdict: KEPT.** Both families pass all three bars (B0 n ≥ 150, B1 X working credit ≥ 2 × J, B2 X false credits ≤ J). See the caveats before you rely on it. In code_exec, X matched the oracle on every candidate, which is close to true by construction.

Branch `cloud/h16-crucible` (worktree /home/claude/cleave-h16-crucible). Code: `pivots/xfamily/`. Tests: `tests/test_xfamily.py`.

## Result

| Family | n (pivots) | Oracle working | J credit (working / false) | X credit (working / false) | Agreement with oracle: J / X | X − J working [95% CI] | X − 2J working [95% CI] | X − J false [95% CI] | B0 | B1 | B2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tool_use | 377 (184) | 142 | 46 (44 / 2) | 122 (122 / 0) | 0.735 / 0.947 | +78 [+64, +92] | +34 [+15, +53] | −2 [−5, 0] | pass | pass (122 ≥ 88) | pass (0 ≤ 2) |
| code_exec | 368 (178) | 160 | 99 (54 / 45) | 160 (160 / 0) | 0.590 / 1.000 | +106 [+91, +121] | +52 [+30, +74] | −45 [−57, −34] | pass | pass (160 ≥ 108) | pass (0 ≤ 45) |

- The CIs come from a pivot-clustered bootstrap: 10,000 reps, seed 0, whole pivots resampled.
- The agreement difference, X minus J, is +0.21 [+0.18, +0.25] in tool_use and +0.41 [+0.36, +0.46] in code_exec.
- **Generated candidates only** (a sensitivity check, not part of the verdict):

  | Family | n | J working / false | X working / false | Verdict |
  |---|---|---|---|---|
  | tool_use | 316 | 44 / 0 | 110 / 0 | passes |
  | code_exec | 300 | 51 / 32 | 122 / 0 | passes |

- **Handwritten candidates only:**

  | Family | n | J working / false | X working / false |
  |---|---|---|---|
  | tool_use | 61 | 0 / 2 | 12 / 0 |
  | code_exec | 68 | 3 / 13 | 38 / 0 |

### By category (all candidates)

| Category | tool_use: n, working, J working/false, X working/false | code_exec: n, working, J working/false, X working/false |
|---|---|---|
| gold copy | 44, 44, 44/0, 44/0 | 42, 42, 42/0, 42/0 |
| equivalent rewrite | 98, 88, 0/0, 78/0 | 119, 118, 12/0, 118/0 |
| near-miss wrong | 127, 0, 0/2, 0/0 | 100, 0, 0/43, 0/0 |
| useless | 59, 0, 0/0, 0/0 | 55, 0, 0/2, 0/0 |
| early answer | 49, 10, 0/0, 0/0 | 52, 0, 0/0, 0/0 |

- **What X misses in tool_use:** 20 working steps. Ten are reordered independent lookups: X runs at horizon 0, and the candidate adds a different fact than the expert's next step. The other ten are right early answers, where the student does the calculator's arithmetic in its head, so the effect is an answer, not a calculator result.
- **What J gets wrong in tool_use:** two false credits, both handwritten: `!28+33+5` for `!28+33+53`, and `!91+-49`, which the calculator rejects. J credits no working rewrite at all, because in turns this short any edit falls below 0.9.
- **What J gets wrong in code_exec:** 45 false credits. Most are sign flips or one-off coefficients in long programs, such as `x*x-2` for `-x*x-2` or `2*x*x+6*x+5` for `2*x*x+7*x+5`. Two are useless edits that left the draft unchanged or reformatted it.

## Protocol (fixed before any label)

- **Commit 3a7b064** holds the code, the tests and the pre-registration, which is in the `pivots/xfamily/run.py` docstring. It covers the bars, the working definition, the X and J definitions, the candidate categories and weights, and the bootstrap. No candidate file existed at that commit.
- **Commit b283e11** seals the baseline:
  - It adds `docs/h16/candidates_*.jsonl` and `docs/h16/baseline_j_*.jsonl`.
  - It adds `baseline_j_sealed.json`, which holds the sha256 of each candidate file and each J-label file.
  - No oracle or X label existed at that commit.
  - The label stage re-checks the seal before it runs: oracle labels first, then X.
- **Sealed sha256s:**
  - candidates_tool_use: 0b37c88b…c19a
  - baseline_j_tool_use: 56efc541…04b8
  - candidates_code_exec: 0b5e8d39…9536
  - baseline_j_code_exec: 5b4da721…5363
- **Environments:** pure ports of the kit's `envs/tool_use.py`, `envs/code_exec.py` and `tasks._eval_expr`, in `pivots/xfamily/envs.py`, with the kit's sha256s recorded in its docstring. `tinylang.py` is copied verbatim, and a test checks that it matches the kit. Torch is not needed, and the kit was not edited.

### Definitions

- **J** is `SequenceMatcher(expert_step, candidate).ratio() >= 0.9` on the raw turn text (tool_use) or program text (code_exec).
- **X in tool_use:**
  - Fork the pre-state, which is the gold turns[:t]. Run the expert step and the candidate from it.
  - The effect is the fact the environment appended: `("read", key, value)`, `("calc", value)` or `("error",)`. For a final turn it is `("answer", int)` or invalid.
  - Credit = 1 when the two effects are equal and are not an error.
- **X in code_exec:**
  - The effect is the program's outputs on the probe inputs −10..10, compared with gold's outputs.
  - There are two pivot types: *final*, a program written from the prompt, and *edit*, a wrong draft revised into a full program.
- **Oracle and "working" in tool_use:**
  - Run the candidate. If it ends the episode, the oracle is `verify()`.
  - Otherwise a gold policy finishes the episode inside the kit's 5-turn budget and `verify()` scores it. The gold policy reads missing keys in prompt order, uses the calculator unless a calculator result already equals the needed value, then answers. A test shows it reproduces every gold trace.
  - A step is working when the episode succeeds **and** the gold policy's remaining-turn count went down.
- **Oracle and "working" in code_exec:**
  - Episodes are single-turn, so the gold policy has nothing left to do. Working means `verify(program) == 1` on the visible tests plus the hidden tests −3..9.
  - Every pre-state fails verify, so a passing step always counts as progress.

### Candidates

- **Generated:** each pivot gets 2 candidates from distinct categories, drawn with these weights: gold 0.15, equivalent 0.25, near-miss 0.30, useless 0.15, early 0.15.
  - tool_use: 45 problems, seed 16001, every turn is a pivot.
  - code_exec: 75 problems, seed 16002, each with a final pivot and an edit pivot.
- **Handwritten, marked `source: handwritten`:** 61 in tool_use and 68 in code_exec, on separate problems (seeds 16101 and 16102). I wrote them before any label existed, in `pivots/xfamily/handwritten.py`.
  - The tool_use ones are what a student model would write: keys inside the calculator, `!11+92=103`, `103.`, prose answers, upper-case keys, carry slips, `!91+-49`, repeated addition.
  - The code_exec ones are the same kind of thing: completing the square, Horner form, `2x` without the `*`, extended lookup tables, half-fixed constants, and patches like `...-16-2`.
  - One of my handwritten "equivalent" programs was actually wrong: `(x+10)*x-20` for `x*x+9*x-20`. The oracle labelled it not working, and both J and X gave it no credit.
- **Tooling:** no model or API was called. Everything ran on 2 cores (`taskset -c 0,1`).

## Caveats

1. **In code_exec, X is close to the oracle by construction.**
   - The probe range −10..10 covers the verify range −3..9. Any program that matches gold on the probes therefore passes verify, so X cannot give a false credit there, and it missed nothing on this set.
   - Exploratory and not pre-registered: with probes that avoid the test range (−10..−4 and 10..12), X still scores 160 working and 0 false.
   - With the visible inputs 0, 1, 2 alone, X would give 32 false credits, mostly the hard-coded lookup tables. That is still fewer than J's 45.
   - The family result therefore depends on probing off the visible examples.
2. **The candidate mix sets the size of the gap.**
   - J only credits exact copies in tool_use and near-exact copies in code_exec. So B1 mostly compares how many equivalent rewrites there are against how many gold copies there are.
   - That ratio was fixed at 0.25 : 0.15 before labelling. With far more gold copies in the mix, B1 could fail.
   - Some of the tool_use rewrites are only whitespace, such as `! ab` or ` 55`, which the tools strip.
3. **X in tool_use works at horizon 0.** It misses reordered lookups and right mental-arithmetic answers, 20 of the 142 working steps. Its agreement with the oracle is 0.947, not 1.
4. **I wrote the handwritten candidates knowing that J is weak on long near-identical strings.** The generated-only results, which pass on their own, are the cleaner evidence.

## Files (this folder)

- `candidates_tool_use.jsonl` and `candidates_code_exec.jsonl`: the candidates and pivots.
- `baseline_j_*.jsonl` and `baseline_j_sealed.json`: the sealed J baseline.
- `labels_*.jsonl`: the oracle label, working flag, X credit and effects for each candidate.
- `results-h16.json`: every count, CI and category breakdown.
