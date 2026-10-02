# Stage 1 (S1): materialise — corpus report

- tasks: **630**; trajectories: **2716**; parsed turns (incl. each trajectory's gold turn): **38,558**
- assistant turns that are not valid JSON (kept, flagged): 307
- expert turns after a trajectory's last released pivot (unrecoverable): 4,848
- E0 files recovered: **3,501** total; per task median 5, p10 0, p90 12, max 35
- partial files (head/tail/sed or truncated): 141 (4.0%)
- tasks with zero recovered files: **104**
- E0 content conflicts across trajectories of one task (same path, full reads, different bytes): 18
- heredoc writes captured from keystrokes (gold_delta with content): 6,778
- tasks flagged: network-fetching 197, service/background 256, sudo 28
- instruction variants per task: {1: 629, 2: 1}

Every count is over all 630 tasks. Files are written under `data/tasks/<task>/e0/` (gitignored); manifests are the record.

## How the E0 conflict count was driven down (measured before each repair)

| Run | Rule | Conflicts | Files |
|---|---|---|---|
| 1 | prompt split anchored at line start; all `cat` reads count | 3,293 | 6,393 |
| 2 | prompt split anywhere (a file without a trailing newline glues the next prompt onto its last byte); `/tmp`, `/logs`, `/var/log`, `/proc`, `/sys`, `/dev`, `/run` excluded as derived | 2,424 | 6,279 |
| 3 | a read counts toward E0 only before the trajectory's first non-read-only command; whitespace-insensitive comparison | **18** | 3,501 |

Run 3 is the rule. Sample diagnosis on 120 seeded tasks between runs 1 and 2: 431 whitespace-only, 288 near-identical, 571 substantive — the substantive ones were glued prompts and files the expert's own scripts had written. Between runs 2 and 3 the remaining substantive cases were files edited by a script or a relative-path `sed -i` before being read; the before-first-mutation rule removes them. The 18 that remain are listed in the run report (`conflict_paths`) for hand review in S7.

Late reads (after the first mutation) are kept in `trajectories.json` per turn: 14,266 across the corpus. They are evidence of the post-mutation state, not of E0.

Gate S1: 630 manifests written; corpus report above; parser behaviour checked on literal samples (glued prompt, heredoc, full read) and conflict classes inspected by hand on the 120-task sample. Exit code 0.
