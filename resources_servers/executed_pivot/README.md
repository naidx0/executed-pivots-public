# Description

`executed_pivot` is the NeMo Gym resources server of Executed Pivots. It is a drop-in replacement for `terminus_judge` with `enable_string_similarity: true` and
`enable_llm_judge: false` (the `terminus_judge_string_only` config) on Terminal-Pivot rows. Each row is one pivot: the
Terminus-2 conversation up to one expert turn, with the expert's next batch as `expected_answer`.

`terminus_judge` compares keystroke strings. `executed_pivot` instead runs the policy's batch and the expert's batch
in two forks of the pivot's rebuilt state, then compares what they did. The comparison covers file changes, deletions,
output, new errors, the cwd and exports left in the shell, and a premature `task_complete`. The reward is 1 when
the effect score is at least `threshold`.

- An effect-equivalent action gets reward 1, however it is spelled.
- An action one character away from the expert's but with the opposite effect gets reward 0.
- Environment failures set `mask_sample: true`, so the trainer drops the sample instead of learning from an
  infrastructure error. They include an unknown pivot, a sandbox error, a probe that did not finish, and an expert
  action that does not reproduce.

The scoring core is `pivots/gym/core.py` (`ExecutedPivotCore`). It has no Gym dependency and is shared with the audit.
This directory adds the Gym contract around it.

## Verify response

On top of `BaseVerifyResponse` (`reward`, `mask_sample`, `failure_kind`, `failure_reason`), the response carries:

| field | meaning |
|---|---|
| `score` | continuous effect score in [0, 1]; `reward = score >= threshold` |
| `j_reward` | what `terminus_judge_string_only` would have paid for the same text |
| `penalties` | multiplicative penalties applied: `side_effects`, `destructive`, `new_errors`, `timeout`, `shell_state`, `premature_complete` |
| `verify_s` | wall time of the executed comparison; Gym aggregates it like any numeric field (`mean/verify_s`) |

The `failure_kind` values are `executed_pivot:<kind>`, following Gym's `<server>:<kind>` convention. The masked ones
are `no_anchor`, `prepare_error`, `expert_probe_failed`, `expert_not_reproducible` and `backend_error`. Policy
failures such as `model_output_invalid` and `task_complete_check_failed` are not masked. `failure_reason` has the
detail.

## Data

Rows have the same shape as `terminus_judge` rows: `responses_create_params`, `uuid`, `expected_answer` (the
expert's Terminus-2 JSON), and `metadata.harness: terminus_2`. `data/anchors.jsonl` maps each `uuid` to the checkpoint
to fork: `{"uuid", "anchor", "cwd", "next_answer"}` (the expert's next action, optional; H39 uses it). Checkpoint ids belong to one world store, so `anchors.jsonl` is machine-local
and not committed.

`scripts/prepare.py` builds rows and anchors from the local specimens (`specimens/` at the repo root). It builds each
specimen's environment in an overlay store, replays the expert, and records one anchor per turn. `data/example.jsonl`
holds `billing-invoice-bugfix` turns 0 to 4.

### Reference file (multi-reference X, H44)

`data/references.jsonl`, next to the anchors (or the file named by `references_path`), is optional. It lists earlier
policy actions verified as working at a pivot, one JSON object per line:

```json
{"uuid": "billing-invoice-bugfix:0", "text": "<Terminus-2 JSON action>", "E": 1, "P": 1, "source": "h31#0"}
```

- `uuid`: the row's uuid (`<task>:<turn>`), the pivot the action was verified at.
- `text`: the action, as the policy wrote it.
- `E`, `P`: its executed outcome and verified progress labels (`pivots.gym.labels`). Both must be 1; any other row
  makes the server refuse to start (`ValueError`).
- `source`: free text, where the row came from.

When the file is present, each reference is run at its pivot like the expert's step (twice; one that does not
reproduce is not used), and a rollout the expert comparison does not credit gets reward 1 if it matches any reference
under the same X rules. Without the file, X compares with the expert's step alone.

The recommended file is `data/references.jsonl`: 762 actions over 124 pivots. It holds the H44 set plus the
reproducible E=1, P=1 actions of the H43 and H45 batches; H48 in RESEARCH.md has the measurement. On the held-out
H47 rollouts it credits +25 working actions over the H44 set, with 0 false credits. The earlier H44 set
(437 actions over 120 pivots from the H31, H37b, H40 and H41 batches) is a subset of it.

# Example usage

The commands below are for the local overlay backend (Linux user namespaces + overlayfs, no Docker, no sandbox
token). Run them from the repo root. The server needs NeMo Gym main, not the PyPI release (0.4.0 lacks
`ReverifyMode` and `mask_sample`), and Gym main needs Python >= 3.13.14. `requirements.txt` pins the Gym commit the
server was tested against, and `gym env test` builds the server's own venv from it with `uv`, which must be on PATH.

```bash
# 0. A Gym venv (about 2 min). uv fetches the Python if the system has none new enough.
pip install uv
uv python install 3.13.15
uv venv --seed --python 3.13.15 .venv-gym && . .venv-gym/bin/activate
(cd resources_servers/executed_pivot && pip install uv -r requirements.txt)  # -e ../.. is the repo root
```

```bash
# 1. Anchors + rows (about 2 s per specimen). The store must match overlay_store in the config.
python3 resources_servers/executed_pivot/scripts/prepare.py specimens/billing-invoice-bugfix --store /tmp/cleave-overlay-gym

# 2. The server's tests under Gym's harness
gym env test +entrypoint=resources_servers/executed_pivot

# 3. Serve it with an agent and a real policy model, and collect rollouts. Needs an OpenAI-compatible model
#    endpoint; without the three policy_* values Gym stops at "references 'policy_base_url', which is not set".
#    For a full rollout run with no model (a stand-in policy), see "Running a rollout" below.
gym env start --config resources_servers/executed_pivot/configs/executed_pivot_overlay.yaml \
    --config responses_api_models/openai_model/configs/openai_model.yaml \
    ++policy_base_url=<url>/v1 ++policy_api_key=<key> ++policy_model_name=<model>
gym eval run --no-serve +agent_name=executed_pivot_overlay_simple_agent \
    +input_jsonl_fpath=resources_servers/executed_pivot/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/executed_pivot/data/example_rollouts.jsonl
```

`configs/executed_pivot.yaml` is the production config. It uses the `contree` backend (Nebius Sandboxes) with anchors
from the rebuilt Terminal-Pivot rows, and needs `contree-client` and a sandbox token.

## Running a rollout

`gym eval run` sends each dataset row through the whole pipeline: the agent (`simple_agent`) calls the policy through
Gym's model server, posts the answer to this server's `/verify`, and aggregates the verify responses into metrics.
The policy can be any OpenAI-compatible endpoint. `scripts/standin_policy.py` is one that needs no model: it answers
`/v1/responses` and `/v1/chat/completions` with actions stored in an audit rows file (`pivots/audit` output, one row
per labelled action with `task`, `turn`, `action`, `text`, `J`, `X`), so the rewards Gym reports can be checked against
labels that are already known. It recovers the pivot from the prompt (the task instruction in the first message, the
turn from the number of assistant messages). `--pick` chooses the action: `expert`, `student` or `control` (random,
per `--seed`), `random`, `cycle` (the k-th request for a pivot gets its k-th stored action), or an action id such as
`ctrl:flipped`. Each reply records the replayed action in `response.metadata.standin_action`, and `--log` writes one
line per request.

These are the commands of the recorded run (all 45 specimen pivots, all 546 labelled actions, from the repo root):

```bash
export PYTHONPATH=$PWD  # this checkout's pivots/ and cleave/, whatever the venv's editable install points at
export AUDIT=out/audit    # a pivots.audit.run output dir with progress.jsonl; the recorded run used the full
                          # audit with the 360 student samples (the package's audit/)

# 1. Rows and anchors for all six specimens (45 pivots, about 15 s); --n-example 0 keeps data/example.jsonl
python resources_servers/executed_pivot/scripts/prepare.py specimens/*/ \
    --store /tmp/cleave-overlay-rollouts --n-example 0

# 2. The stand-in policy. cycle + 14 repeats replays every stored action (10 to 14 per pivot) at least once
python resources_servers/executed_pivot/scripts/standin_policy.py \
    --actions $AUDIT/rows.jsonl --pick cycle --port 8911 --log /tmp/ep-rollouts-standin.jsonl &

# 3. Gym starts the three servers (Ray), prepares the train split, collects 630 rollouts, aggregates metrics
gym eval run \
    --config resources_servers/executed_pivot/configs/executed_pivot_overlay.yaml \
    --config responses_api_models/openai_model/configs/openai_model.yaml \
    --agent executed_pivot_overlay_simple_agent --split train --num-repeats 14 --concurrency 3 \
    --model standin --model-url http://127.0.0.1:8911/v1 --model-api-key dummy \
    --output /tmp/ep-rollouts/rollouts.jsonl \
    ++executed_pivot_overlay_resources_server.resources_servers.executed_pivot.overlay_store=/tmp/cleave-overlay-rollouts
kill %1

# 4. Gym's rewards against the known labels (exit 1 on any reward != X or j_reward != J)
python resources_servers/executed_pivot/scripts/check_rollouts.py /tmp/ep-rollouts/rollouts.jsonl \
    --actions $AUDIT/rows.jsonl --progress $AUDIT/progress.jsonl
```

Gym writes `rollouts.jsonl`, `rollouts_aggregate_metrics.json` (the metrics table, per agent, per task and per
repeat), `rollouts_materialized_inputs.jsonl` and the prepared split next to `--output`.

- The chat-completions path, which is what a vLLM-served model uses: replace the second `--config` with
  `responses_api_models/vllm_model/configs/vllm_model.yaml`. Gym then drops the reply's metadata, so run one request
  per pivot (`--pick random --num-repeats 1`) and pass `--standin-log` to `check_rollouts.py`.
- A real policy: stop the stand-in and point `--model`, `--model-url` and `--model-api-key` at the endpoint.
- A Nebius Token Factory model (default Nemotron 3 Nano): start the budget proxy,
  `python -m pivots.students.tokenfactory --serve 8912 --max-requests <cap>` (it reads the key from
  `$NEBIUS_API_KEY` or `/root/.config/nebius/tf_key` and stops for good on 401/402/403 or any billing or quota
  error), and replace the second `--config` with `resources_servers/executed_pivot/configs/tokenfactory_policy.yaml`
  (Gym's `inference_provider` server, pointed at the proxy; no `--model-*` flags). Bound the spend with
  `--limit` and `--num-repeats`: one policy request per rollout. `tests/test_gym_tokenfactory_policy.py` runs one
  pivot through the config, the proxy and a mock endpoint.
- NeMo Gym's own pyproject wants Python >= 3.13.14 and builds a uv venv per server. On a machine with an older 3.13 and
  a relaxed Gym install (this lane's), add `+uv_venv_dir=<dir> +skip_venv_if_present=true`, where
  `<dir>/{resources_servers/executed_pivot,responses_api_agents/simple_agent,responses_api_models/openai_model}/.venv`
  are symlinks to the Gym venv.

`tests/test_gym_rollout_standin.py` (repo root) runs one pivot through the same pieces in process: the openai_model
server class calls the stand-in over HTTP, and its responses go to `/verify` and `/aggregate_metrics`. It skips
without NeMo Gym.

# Licensing information

Code: Apache 2.0<br>
Data: the specimens are original to this repo

## Dependencies

- nemo_gym: Apache 2.0
- Executed Pivots (this repo; Python packages `cleave` and `pivots`): the world backends and the scoring core
