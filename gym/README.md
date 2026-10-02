# Real Gym rollouts from a local model (H31)

One command runs executed_pivot rollouts with a local model as the policy, under the shared GPU lock, and labels
them with X, J, E and P:

```bash
gym/run_local_policy.sh -n 2 --driver gym
```

Everything except the model has been run on CPU with a stand-in policy (numbers below). The first real run needs the
GPU and a served model.

## Pieces

| piece | what it does |
|---|---|
| `pivots/students/local_policy.py` | OpenAI-compatible proxy between Gym and llama-server or Ollama's `/v1`. It reads the endpoint from `LOCAL_POLICY_BASE_URL` and `LOCAL_POLICY_MODEL` (plus `LOCAL_POLICY_API_KEY` only if the server wants one). No key is in any file. It times out each attempt (`LOCAL_POLICY_TIMEOUT`, 300 s) and retries timeouts, dropped connections, 429 and 5xx (`LOCAL_POLICY_RETRIES`, 2). It then reduces the reply to the Terminus-2 action (see below). |
| `resources_servers/executed_pivot/configs/local_policy.yaml` | Gym's `policy_model` (the `inference_provider` server) pointed at the proxy (`LOCAL_POLICY_PROXY_URL`, default `http://127.0.0.1:8913/v1`). Sampling: temperature 1.0, top_p 0.95, max_tokens 4096. One request at a time. |
| `resources_servers/executed_pivot/scripts/prepare.py --set all` | Rows and anchors for all eleven specimens: 89 pivots. |
| `pivots/gym/rollouts.py` | The `direct` driver. It does simple_agent's job in one process without a Gym install: prompt, then proxy, then `ExecutedPivotCore.verify` (the function `/verify` calls). |
| `pivots/gym/labels.py` | Adds E and P to each unmasked rollout, and writes `labels.jsonl`, `summary.json` and `summary.md` with the H31 verdict. |
| `gym/gpu_lock.py` | The GPU lock. It uses the lock line and ledger format of ml-harness-stage `scripts/take_the_card.py` (copied there, not imported). The take is atomic, and a release removes only the holder's own lock. |
| `gym/gym-runner.Dockerfile` | `gym-runner:local`: h24-runner plus NeMo Gym 2921bf0 (the commit `requirements.txt` pins) on Python 3.13. `gym eval run` and the Gym-side tests run in it. |

**How a reply becomes an action.** The proxy handles four cases:
- `<think>` blocks and the `reasoning` and `reasoning_content` fields go to `reasoning_content`. Gym turns that into a reasoning item, which executed_pivot does not score.
- The action is the last valid Terminus-2 object in the content. Code fences and prose around it are dropped, as the Terminus-2 harness tolerates them.
- If the content holds no action but the reasoning does, the action comes from the reasoning. Thinking models served by Ollama or llama-server often put the whole answer there.
- If there is no valid action anywhere, the reply passes through unchanged. executed_pivot then pays 0 as `model_output_invalid`, which is not masked.

X and J both score the text the proxy passes on. Each rollout records where the action came from: `policy.source` in direct rollouts, and `policy.jsonl` for both drivers.

**Which specimens need which image.** The overlay backend runs `setup.sh` on the container's own root, so that root
needs each specimen's toolchain. `prepare.py` stops early and names the image when a tool is missing.

| specimens | smallest runner | why |
|---|---|---|
| billing-invoice-bugfix, sensor-site-report, access-log-summary, backup-cron-repair; textstats-pyproject-install, sqlite-email-dedupe, jq-signup-rollup | h17-runner:local | python3, jq, awk, gzip, tar |
| ledger-git-revert, statcli-make-fix | h20-runner:local | git, make, gcc |
| **csvsum-npm-package, tally-go-build** | **h24-runner:local** | nodejs and npm, and golang-go |

`h24-runner:local` has all eleven, and the script uses it, or `gym-runner:local` built on it, for everything.
backup-cron-repair writes `/etc`, so the containers run privileged with `/etc/resolv.conf`, `/etc/hosts` and
`/etc/hostname` unmounted, as in the H20 runner. The script pins the model host's IPv4 address before the unmount,
because the unmount also removes Docker's resolver.

## Commands

One-time setup, both done on this machine on 2026-09-24:

```bash
docker build -t gym-runner:local -f gym/gym-runner.Dockerfile gym   # about 70 s; needs h24-runner:local
docker volume create gym-store
```

Serve the model on the Windows host. The flags below are an example: pick the model, and keep the context at or
above 12k so the longest prompt (about 5k tokens) plus 4096 new tokens fit. Then run under the lock:

```bash
export LOCAL_POLICY_BASE_URL=http://127.0.0.1:8080/v1     # llama-server; Ollama: http://127.0.0.1:11434/v1
export LOCAL_POLICY_MODEL=<served model name>             # Ollama needs the tag; llama-server ignores it
# Pre-flight: 6 pivots x 1 rollout, about 5 min (estimate), to check parse rate and speed before the long run
gym/run_local_policy.sh --driver gym -n 1 --out out/gym/h31-preflight \
    --pivots billing-invoice-bugfix:3,ledger-git-revert:3,csvsum-npm-package:5,tally-go-build:4,jq-signup-rollup:2,textstats-pyproject-install:3 \
    --serve-cmd "llama-server -m <model.gguf> -c 12288 -ngl 99 --port 8080"
# H31: all 89 pivots x 2 rollouts
gym/run_local_policy.sh --driver gym -n 2 --out out/gym/h31 \
    --serve-cmd "llama-server -m <model.gguf> -c 12288 -ngl 99 --port 8080"
```

With `--serve-cmd`, the script starts the server after taking the lock. Before releasing the lock it stops the
server's whole process tree (`taskkill /T /F`). Without it, the endpoint must already be up. The script waits up to
`--serve-wait` seconds (default 600) for `/models` either way. Killing Ollama does not reap `llama-server.exe`,
so after a run check `tasklist | grep -i llama` for orphans holding VRAM.

What the script does and writes, all under `--out`:

| step | GPU lock | output |
|---|---|---|
| 1. dataset: `prepare.py --set all` in h24-runner, then the `--pivots` / `--limit` selection | not held | `data/train.jsonl`, `data/anchors.jsonl`, `data/selected.jsonl`, `prepare.log` |
| 2. take the lock (`$GPU_LOCK`, default `~/.gpu/gpu.lock`); exit 3 at once if another lane holds it, or poll with `--lock-wait S` | taken | a line in `gpu.lock` and `gpu.lock.log` |
| 3. start the model (optional), check the endpoint, run N rollouts per pivot through the proxy | held | `rollouts.jsonl` (direct) or `gym/rollouts.jsonl` plus Gym's aggregate metrics (gym), `policy.jsonl` (one line per policy request), `proxy.log`, `rollouts.log` |
| 4. stop the model (if started) and release the lock; a trap does both on any exit | released | a `released ... held=<s>` ledger line |
| 5. labels: E and P from the specimens' verifiers | not held | `labels.jsonl`, `summary.json`, `summary.md` |

`--driver direct` runs the same rollouts without Gym, in h24-runner. It is the fallback if the Gym image is
unavailable. `--lock-only` takes and releases the lock and does nothing else. `--skip-prepare` reuses `data/` in
the same `--out`, and `--skip-labels` stops after step 4.

## Verified on CPU (2026-09-24, no model)

The "local model" in these runs was `standin_policy.py --from-rows out/gym/data/train.jsonl --style reasoning
--pick expert`. It replays each pivot's expert action with an empty `content` and the JSON at the end of
`reasoning`, the shape a thinking model served by Ollama returns. The lock was a scratch file, not the machine's
lock.

| run | rollouts | under lock | result |
|---|---|---|---|
| `--driver direct`, 6 pivots across 6 specimens (incl. csvsum, tally, textstats, jq), n=2 | 12 | 44 s | reward 12/12, X=J=E=P=1 on all 12, masked 0; labels 68 s |
| `--driver gym` (`gym eval run`, Ray, three servers), same pivots | 12 | 79 s | mean/reward 1.0, mean/j_reward 1.0, mean/verify_s 5.56; X=J=E=P=1 on all 12 |
| `prepare.py --set all` in h24-runner | 89 pivots | n/a | 49 s; the same build in h20-runner stops on tally-go-build: "needs go" |
| failure path (`tests/test_gym_run_script.py`) | 0 | 1 s | endpoint down: exit 1, lock released by the trap; foreign lock: exit 3, lock untouched |

Tests (all CPU, no model):
- `tests/test_local_policy.py` (16): reasoning-only replies (`reasoning`, `reasoning_content`), think and fence
  wrapping, malformed JSON (truncated, schema, prose), timeouts retried then given up with the count, 5xx retried,
  4xx not, a refused connection, the key only in the header, and the proxy's 502 on upstream failure.
- `tests/test_gym_run_script.py` (8): the lock line in the protocol format, a held lock refused (negative
  control), eight racing takers with exactly one winner, and a release refused for a foreign lock. It also drives
  the script itself: `--lock-only` takes and releases; a held lock gives exit 3; a failed run is released by the trap.
- `tests/test_gym_labels.py` (4): both rollout shapes, the H31 verdict (win, loss on one false credit, loss on
  agreement, inconclusive), and the sign test.
- `tests/test_gym_local_rollouts.py` (2, Linux): a fake model drives the proxy and the direct driver through
  `ExecutedPivotCore.verify` at billing-invoice-bugfix:3 and ledger-git-revert:3. Results: a think-wrapped expert gets 1; a flipped
  sed gets X 0 and J 1; malformed JSON gets 0 as `model_output_invalid`, not masked; a reasoning-only expert gets 1;
  503 twice gives a masked `policy_error`. Then the real E/P labeller runs.
- `tests/test_gym_local_policy.py` (2, Gym): the config's `inference_provider` goes to the proxy, then the fake,
  then the app's `/verify`. A reasoning-only reply becomes a reasoning item plus reward 1, and malformed JSON gets
  `executed_pivot:model_output_invalid`, not masked.

The Gym-side and Linux tests run in the image:

```bash
docker run --rm --privileged -v gym-store:/tmp -v "$(pwd -W)":/mnt/src -w /mnt/src \
  -e PYTHONPATH=/mnt/src:/opt/Gym gym-runner:local bash -c \
  "umount /etc/resolv.conf /etc/hosts /etc/hostname; /opt/gym/bin/python -m pytest tests resources_servers/executed_pivot/tests -q"
```

## Expected cost of the first real run (estimates, not measured)

These estimates assume a 4B-class model at Q4 on the RTX 2060 Super (8 GB). The throughput figures are assumptions
until the pre-flight measures them: about 1,000 tokens/s of prompt processing and about 40 tokens/s of generation.
Prompt sizes are measured from the 89 rows: median 9.1k characters, max 17.6k, 825k in total. The token counts
assume about 3.5 characters per token.

| per rollout | thinking model | non-thinking model |
|---|---|---|
| prompt tokens | median about 2.6k, max about 5k | same |
| completion tokens | about 800 to 2,500 | about 150 to 400 |
| generation time | about 20 to 60 s | about 4 to 10 s |
| verify (measured on CPU) | warm about 0.4 s, cold 2 to 7 s (csvsum and tally builds); overlaps the next generation at concurrency 2 | same |

| whole run, 89 pivots x 2 = 178 rollouts | thinking | non-thinking |
|---|---|---|
| prompt tokens (llama-server's prompt cache makes the second repeat mostly free) | about 470k processed | about 470k |
| completion tokens | about 140k to 450k | about 27k to 70k |
| time under the GPU lock | about 1.5 to 3.5 h | about 20 to 40 min |
| dataset before the lock plus labels after it (CPU) | about 1 min plus about 10 to 20 min | same |

The pre-flight (6 pivots x 1) should take about 5 minutes under the lock with a thinking model. Scale the table
above by its measured `gen_s_median` and completion tokens (`summary.md`, "proxy log" line) before starting the
long run.

## What the first real run should measure

1. **Parse health.**
   - How often each action source occurs: `content`, `reasoning`, `invalid`, `empty`. Read it from `summary.md`'s proxy-log line and `policy.jsonl`.
   - `finish_reason: length`, which means max_tokens ran out mid-thought.
   - If `invalid` plus `empty` is above about 20% in the pre-flight, fix the serving (chat template, a reasoning-format flag, max_tokens) before the long run. Those rollouts get X = J = E = 0 and dilute H31 without testing it.
2. **Cost against the estimates.** Measure prompt and completion tokens per rollout, generation seconds, verify seconds, and the time the lock was held (the `held=` value in the ledger).
3. **H31** (below), with its paired table: only-X-right against only-J-right, and the sign test.
4. **Infrastructure.** Count masked rollouts by kind: `policy_error` (upstream failures after the retries) and `expert_not_reproducible`. Also count 30 s `cmd_timeout` hits, which on the stand-in came from students opening `vi`.
5. **Driver agreement, if both are run.** On the same pivots, the Gym driver's `reward` and `j_reward` should equal what the direct driver computes for the same texts (same core).

## H31 (pre-registered 2026-09-24, before any real policy rollout)

**Hypothesis.** On real policy rollouts, X agrees with E at least as well as J does, paired over the same
rollouts, and X gives no false credit.

- **Rollouts.** `gym/run_local_policy.sh --driver gym -n 2 --out out/gym/h31`: all 89 pivots of the eleven
  specimens, 2 rollouts per pivot, one local model through `configs/local_policy.yaml` (temperature 1.0,
  top_p 0.95, max_tokens 4096), executed_pivot overlay backend, threshold 0.75, cmd_timeout 30.
- **Labels, per unmasked rollout.**
  - X is Gym's `reward`.
  - J is Gym's `j_reward`.
  - E is the executed outcome: fork the pivot, run the action, splice the expert's remaining turns, run the verifier.
  - P (verified progress) is reported but does not decide.
- **Exclusions.** Masked rollouts (environment failures and `policy_error`) are excluded and reported by kind.
  Unparseable replies are included: they are a policy outcome.
- **Win.** At least 100 labelled rollouts, and #(X = E) ≥ #(J = E) over the same rollouts, and #(X = 1, E = 0) = 0.
- **Loss.** Either condition fails. Below 100 labelled rollouts the result is inconclusive: run more repeats, and do not reinterpret.
- **Reported, not deciding.** Credit on working steps (X = 1 and E = 1, against J), only-X-right against
  only-J-right with the exact sign test, the same tables against P, and all of it again on parseable rollouts.
- **Stated edge.** E splices the expert's remaining turns (open loop), so it can over-credit an action that a
  later expert turn repairs, and under-credit one that breaks a later expert command. One model is not a claim
  about policies in general. The model is whichever one is served, and the run records it in the lock line and in
  `endpoint.json`.
