# program.md: improve the reward yourself

This is the research loop behind this repo, set up so an agent (or you) can run it. The pattern follows
Karpathy's autoresearch: one file you change, one fixed evaluation, one number, keep or revert.

## The rules
- **You change one file:** `pivots/effect/xreward.py`, the executed-effect reward X.
- **You never change:** the specimens, the verifiers, `pivots/audit/`, or the sealed label files. They are the exam.
- **The number:** working student steps X pays at each of the 45 pivots (E = 1), from `audit_scores.py`.
  Hard gate: wrong credits (paid while the verifier says the step failed) stay at 0.
- **Noise:** `arloop.py compare --paired` pairs pivots and keeps a change only if the gain is above 2 standard errors.

## The baseline
Audit v5 is the sealed baseline (546 actions). Checked on 2 Oct 2026 with these two files:
- X vs the shipped keystroke reward: 2.09 vs 0.58 working steps per pivot, t 5.33, **KEEP**. Wrong credits 0 vs 34.
- The red-team fixes (audit v4 to v5): +4 working steps, t 1.43, not decided on credit. They were kept for
  safety (hidden side effects paid 48 to 0), which is a separate bar.

## One round
1. Pre-register: `python arloop.py prereg --id R1 --hypothesis "..." --win "gain above 2 SE, wrong credits 0"`.
2. Edit `xreward.py`. Keep the change small.
3. Score it:
   ```bash
   python -m pivots.audit.run specimens/* --students audit-v5/students.jsonl --out runs/R1
   python audit_scores.py runs/R1/rows.jsonl --reward X > runs/R1/scores.json
   python arloop.py verdict --base base.json --new runs/R1/scores.json --paired --id R1
   ```
   `base.json` is `python audit_scores.py audit-v5/rows.jsonl --reward X`. Check `total_wrong` is 0.
4. KEEP: commit with the numbers in the message. Otherwise: `git checkout pivots/effect/xreward.py`.
   `arloop.py` writes the verdict to `research/ledger.jsonl` and `log.md` either way.

## Ideas not yet tried
- Partial credit for steps that make real progress but differ from the teacher (X pays 94 of 286 working steps now).
- Compare file permission bits (H52, built).
- Cache effects for repeated batches at the same pivot (H53, built).
- Run the loop on the H17 Dockerfile tasks, which X was never tuned on.

See RESEARCH.md for what has already been tried and lost.
