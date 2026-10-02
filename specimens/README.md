# Specimen tasks

Six small terminal tasks with known ground truth, in the format `pivots/specimen.py` describes
(`task.json`, `setup.sh`, `trajectory.json`, `tests/test.sh`). They stand in for NVIDIA's ATCB
tasks on machines that cannot pull those images: each has a small project under `/app` (plus `/data`,
`/var/log` or `/etc` where the domain needs it), a realistic instruction, and a 7 to 9 turn expert
trajectory in the GLM-5.1 / Terminus-2 style. The expert explores, reads files, edits, runs the
project's tests or a script, checks the output, and ends with a `task_complete: true` turn that has no commands.

Every verifier runs offline with only the standard library (`tests/test.sh` calls `python3 /tests/test_outputs.py`)
and writes 1 or 0 to `/logs/verifier/reward.txt`. The verifiers check outcomes (files, contents, git
state, test results, hidden inputs), not the commands used. The alternative-approach checks below show that
different correct solutions pass and plausible wrong ones fail.

| name | domain | turns | what the verifier checks |
|---|---|---|---|
| `billing-invoice-bugfix` | Python bug fix (off-by-one pagination, `>` vs `>=` discount threshold) with failing unittests | 7 | The visible test file is byte-identical to a pristine copy, and the pristine visible suite passes against `/app/billing`. Also checks 12 hidden edge cases: last page, past the end, page 0 raises, default `per_page`, discount at 9/10/11 units, and invoice totals. |
| `sensor-site-report` | Data pipeline: implement `build_site_summary` and aggregate `/data/readings/*.json` into `/app/reports/site_summary.csv` | 7 | Recomputes the report from `/data`. Checks the exact header, one row per site in site order, the counts, and 2-decimal mean/max within 0.005. It also calls `build_site_summary` on a hidden record set that includes a maintenance sensor, an offline sensor and null values. |
| `backup-cron-repair` | Ops: broken bash backup script, broken config, broken cron snippet | 9 | The tarball members are relative to `/app/data`. `MANIFEST.txt` equals a freshly computed sha256 list. `backup.cron` and `/etc/cron.d/app-backup` both schedule `30 2 * * * root .../backup.sh`. After deleting `/var/backups/app`, the script runs from `/` with a minimal env (`PATH`, `HOME` only), exits 0 and recreates correct artifacts. |
| `ledger-git-revert` | Git: undo a bad commit on `main` without rewriting history, then branch `release/1.1` | 7 | `main` is checked out. All 5 original commit hashes are ancestors of `main`, a new commit exists, and the working tree is clean. `release/1.1` equals `main`. From `git archive main`, the committed test suite passes (6 tests), later commits' files are present, and the hidden tax values are right. |
| `access-log-summary` | Text processing (awk/sort/uniq) over two rotated access logs, `/healthz` excluded | 7 | Recomputes both reports from the logs. `traffic_summary.txt` is compared line by line: totals, unique IPs, status classes, and the top 3 IPs with a count tie broken byte-wise (`10.0.0.12` before `10.0.0.7`). `errors_by_path.csv` has query strings stripped and ties broken by path. |
| `statcli-make-fix` | C + Makefile: missing `-lm` on the link line, and a truncating `qsort` comparator that breaks the median | 8 | The test cases are byte-identical to pristine copies. A fresh copy of the project without `build/` and `bin/` builds with `make`, and `make test` passes. `bin/statcli` gives the right count, mean, median and stddev (3 dp) on 6 hidden inputs: even/odd counts, negatives, one value, and 101 values. |

Coverage of the trajectory features the pipeline cares about:

- **Directory changes (`cd`), which make cwd carry into later turns:** `sensor-site-report` (`/app/pipeline`), `backup-cron-repair` (`/app/ops`, then `/`), `ledger-git-revert` (`/app/ledger`), `access-log-summary` (`/var/log/webapp`, then `/app/reports`) and `statcli-make-fix` (`/app/statcli`).
- **Environment variables:**
  - `sensor-site-report` runs `export REPORT_DIR=/app/reports`. `run.py` defaults to a relative `out/`, so a student who forgets the export writes the report to the wrong place.
  - `access-log-summary` runs `export LC_ALL=C`.
  - `backup-cron-repair` sets `BACKUP_CONF=./backup.conf` inline and runs `env -i`.
- **Relative paths after a `cd`:** used in 5 of the 6 tasks.
- **Turns whose last command exits non-zero:** `billing` turn 2 (`unittest` fails) and `statcli` turns 1 and 3 (link error, `make test` fails).
- **Heredoc edits:** `cat > report.py <<'EOF'` in `sensor-site-report`, and `python3 - <<'EOF'` in `statcli-make-fix`.

## Validation (overlay backend, measured 2026-09-23)

Run `python3 specimens/validate.py` from the repo root to repeat it (about 90 s for all six).

- **gold:** the verifier on the final expert checkpoint, run 3 times.
- **nop:** the verifier on the freshly built environment.
- **partials:** the verifier on every step checkpoint before the first checkpoint that passes. All of them must be 0, and there must be at least 2.
- **replay:** two more builds from scratch and replays. The verdict vector (base plus every step) has to be identical across all three builds.

| name | gold x3 | nop | partials (step checkpoints before the solving step) | first passing step | replay x3 (verdict vectors) |
|---|---|---|---|---|---|
| `access-log-summary` | 1 1 1 | 0 | 0 0 0 0 (4) | after turn 4 (errors CSV written) | identical |
| `backup-cron-repair` | 1 1 1 | 0 | 0 0 0 0 0 0 (6) | after turn 6 (cron installed) | identical |
| `billing-invoice-bugfix` | 1 1 1 | 0 | 0 0 0 0 (4) | after turn 4 (second bug fixed) | identical |
| `ledger-git-revert` | 1 1 1 | 0 | 0 0 0 0 0 (5) | after turn 5 (branch created) | identical |
| `sensor-site-report` | 1 1 1 | 0 | 0 0 0 0 (4) | after turn 4 (`run.py` with `REPORT_DIR`) | identical |
| `statcli-make-fix` | 1 1 1 | 0 | 0 0 0 0 0 (5) | after turn 5 (comparator fixed) | identical |

The partial checkpoints include the realistic half-done states:
- `billing` after only the discount fix.
- `backup` with the script and config fixed and run, but the cron job not installed.
- `ledger` with the revert committed but no `release/1.1` branch.
- `access-log` with only the summary written.
- `statcli` with the link fixed but the median still wrong.
- `sensor` with `report.py` implemented but the pipeline not yet run.

Every step's rendered output was read against the next turn's `analysis`. The expert never claims to see
output that the environment does not produce. The numbers it quotes (for example 288 requests, the 57/57
tie, `median=3.550`, and exit 127 from `DEST_DIR: command not found`) come from the replayed observations.

**Alternative approaches.** Each was run from the base state as one batch; rerun with `python3 specimens/alternatives.py`. All 15 behaved as intended.

These different-but-correct solutions pass:
- billing: `per_page * (page - 1)`.
- sensor: `statistics.mean` with an inline `REPORT_DIR`.
- backup: config rewritten with `printf`, cron line `30 02 * * * root bash /app/ops/backup.sh`.
- ledger: manual fix plus `git commit` instead of `git revert`.
- logs: pure Python instead of awk.
- statcli: `-lm` appended directly to the recipe.

These plausible wrong solutions fail:
- billing: editing the tests to match the bug.
- sensor: not filtering on status.
- sensor: forgetting `REPORT_DIR`, so the report lands in `out/`.
- backup: working only when `BACKUP_CONF` is exported.
- ledger: `reset --hard` plus `cherry-pick`, which rewrites history.
- ledger: leaving `release/1.1` checked out.
- logs: sorting IPs numerically instead of byte-wise.
- statcli: `-lm` in `CFLAGS`, which sits before the objects so `--as-needed` drops it.
- statcli: fixing the link but not the comparator.

## Sandbox notes (overlay backend)

1. **`rm -rf` of a directory that exists in a lower layer fails with EIO.**
   - Example: `mkdir d && touch d/f` in one run, then `rm -rf d` in a later run. The files are removed, but `rmdir d` returns `Input/output error`, and later runs on that checkpoint can hit EIO too: in `statcli`, after an expert `make clean`, the verifier's `mkdir build` failed with EIO.
   - Deleting the file in one run and the directory in the next works. Directories created in the same run delete fine.
   - Cause: `OverlayWorld` mounts overlayfs inside a user namespace without the `userxattr` option. Removing a merged directory needs its upper copy marked opaque, and that xattr cannot be set, so the kernel returns EIO. The same two-mount sequence succeeds with `-o ...,userxattr` (reproduced with raw `unshare -rm` + `mount -t overlay`).
   - Workarounds in these specimens: the `statcli` verifier builds a fresh copy of the project instead of running `make clean`, and the expert uses `make -B` instead of `make clean`.
   - Any released trajectory that runs `rm -rf <dir from an earlier turn>` (for example `make clean` or `rm -rf build`) will diverge on this backend.
2. **The host's `/root/.gitconfig` leaks into every world.** The base image is the host root. On this host it sets `commit.gpgsign = true` with `gpg.ssh.program = /tmp/code-sign`, and `/tmp` is empty inside the world, so every `git commit` fails (`cannot exec '/tmp/code-sign'`, rc 128). `ledger-git-revert/setup.sh` writes its own `/root/.gitconfig`. The whole host tree (including `/home/claude/cleave`) is also visible inside worlds, so `find /` would see it.
3. **A step's exit code is the exit code of the batch's last command only.** Earlier failures in the same batch are not reflected in it (`step.sh` sets `__rc` after every command).
4. **Some observations are nondeterministic** because worlds use the real clock: `ls -la` mtimes, and the hash and `Date:` line printed by `git revert`. Verdicts are unaffected (replay x3 is identical), but byte-level observation fidelity is not.
