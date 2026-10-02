# Stage 0 profile: nvidia/Nemotron-RL-Agentic-Terminal-Pivot-v1

Source file: `atcb_terminal_pivot_release_final_v2.jsonl` (1,372,462,504 bytes). Whole file read once; every count is to the end of the file.

## Counts (distinct)
- rows (pivots): **31,111**
- distinct task_name: **630**
- distinct source_trajectory_uid: **2,716**
- trajectories per task: min 1, median 5.0, max 5
- total_source_agent_turns per trajectory: min 3, median 14.0, p90 26, max 78
- pivot coverage (pivots / turns) per trajectory: median 0.77; trajectories with every turn present: 41 of 2,716

## Field values
- schema_version: {'terminus2_pre_rp_pivot_sample_v1': 31111}
- harness: {'terminus_2': 31111}
- teacher_model: {'zai-org/GLM-5.1-FP8': 20013, 'zai-org/GLM-5.1': 11098}
- tool_name: {'bash_command': 28360, 'mark_task_complete': 2751}
- message roles in history: {'user': 312729, 'assistant': 281618}
- expected_answer shape: {'json-ish': 31111}; chars median 936, p90 1492

## History size
- messages per row: median 15, p90 41, max 155
- history chars per row: median 29,588, p90 86,412, max 293,224

## Reconstruction risk signals (regex over the history text; a row counts once)
- rows whose history contains a network-fetching command (curl/wget/pip install/apt/git clone/npm i): **31,111** (100.0%)
- rows whose history contains a long-running service or daemon pattern: **12,331** (39.6%)
- rows containing sudo: **792** (2.5%)
- rows with a truncation marker in observations: **3,427** (11.0%)

These are upper bounds on affected rows, not on affected tasks; the same command appears in every later pivot of its trajectory.

## One row, abbreviated
- task_name: `telemetry-pipeline-audit-2`; tool_name: `bash_command`
- first messages:
  - **user**: `You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.  Format your response as JSON with the following structure:  {   "analysis": "Analyze the current state based on the terminal output provided. What do you `
- expected_answer (first 700 chars):
```
{
  "analysis": "I'm at the project root directory. I need to explore the project structure, understand the existing code, and then implement the missing pipeline logic.",
  "plan": "First, I'll explore the directory structure to understand what files exist, then read the key files to understand the current implementation and what needs to be done.",
  "commands": [
    {
      "keystrokes": "find /app/ingest_pipeline -type f | head -50\n",
      "duration": 1.0
    },
    {
      "keystrokes": "find /data -type f | head -20\n",
      "duration": 1.0
    },
    {
      "keystrokes": "ls -la /app/reports/ 2>/dev/null || echo 'reports dir does not exist'\n",
      "duration": 0.5
    }
  ],
  
```

## Largest tasks by pivot count
- `implement-unit-conversion-proof-checker`: 187 pivots, 5 trajectories
- `claims-notebook-state`: 185 pivots, 5 trajectories
- `strict-rbac-compliance`: 162 pivots, 5 trajectories
- `kiosk-evidence-applier`: 147 pivots, 5 trajectories
- `rail-yard-accumulation-coordinator`: 146 pivots, 5 trajectories
- `hft-dark-pool-settlement-pipeline`: 142 pivots, 5 trajectories
- `stale-cache-rejection`: 138 pivots, 5 trajectories
- `airgapped-evidence-fuzzer`: 135 pivots, 5 trajectories
- `broadcast-playout-reconciliation`: 123 pivots, 5 trajectories
- `rag-calibration-orchestrator`: 121 pivots, 3 trajectories

## Corrected instrument: risk signals over the expert's commands only (per task, not per row)

The first pass matched the system prompt itself (it mentions `pip install`), which made the row-level network figure read 100%. Re-measured over the `keystrokes` in `expected_answer` across all pivots of all trajectories of a task:

- tasks whose expert ever runs a network-fetching command (curl/wget/pip install/apt/git clone/npm i): **178 of 630 (28.3%)**
- tasks whose expert ever starts a service or background process (systemctl/docker/pg_ctl/nginx/uvicorn/nohup/trailing `&`): **238 of 630 (37.8%)** (upper bound; the trailing-`&` pattern is loose)
- tasks with sudo: **27 (4.3%)**
- expert commands per trajectory (from covered pivots): median 33, p90 68
- every `expected_answer` parses as JSON (0 failures); each is a Terminus-2 response: `analysis`, `plan`, `commands[].keystrokes`, `task_complete`

## What a trajectory looks like (one read end to end)

- Harness is **Terminus 2** (Terminal-Bench's agent); observations are raw terminal output with prompt `root@modal:/app/...#`, so the originals ran on Modal.
- Task files live under `/app` and `/data`; the expert's first moves are `find`, `ls`, `cat`, so **the contents of every file the expert read are in the history verbatim**. With five trajectories per task, the union of reads covers most of the initial filesystem.
- The first user message is the system prompt plus the task description; the task's own tests are never shown.
- Pivot coverage is median 0.77 of turns; 41 trajectories have every turn. The last pivot of a trajectory holds its whole history.

## The reward, pinned (all 31,111 rows, 2026-09-23)

- `agent_ref.name` = `terminus_judge_string_only_simple_agent` on **31,111 of 31,111** rows; `agent_ref.type` = `responses_api_agents`.
- Per NeMo Gym's `resources_servers/terminus_judge` (read by the protocol researcher at commit 5b0aae0): the candidate must parse as JSON in the Terminus-2 schema; if the gold has `task_complete=true` the candidate must too (one-sided: early completion is not penalised); then `SequenceMatcher` on the concatenated `keystrokes` must reach **≥ 0.9**; `enable_llm_judge: false`. The judge sees the gold JSON and the candidate JSON only — no terminal history, no command output.
- Rows carrying a per-row `threshold` override: **0**. Gold rows with `task_complete=true`: **2,751** (= the `mark_task_complete` rows).
- Consequence: the released reward is a string-similarity proxy. The audit question is whether keystroke similarity predicts executed outcome — "the keystroke is not the effect".
- Also from the card (per the protocol researcher): rows are *every valid assistant turn of a kept trajectory* (trajectories that passed their verifier, teacher GLM-5.1, ≤ 5 per task), not PivotRL's variance-filtered pivots. Lightning was RL-trained on these same rows, so its actions on these prompts are contaminated; the training arm needs an uncontaminated base, and the audit's student must be re-run on a sample with one.

## Model roles (decided 2026-09-23, on the evidence above)

- **Student for the audit and base for the RL arm: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`** — released 15 Dec 2025; its card lists no terminal-pivot data; the pivot set's teacher (GLM-5.1) is a 2026 model, so the data postdates Nano. A pre-RL Base checkpoint is linked (`...-Nano-30B-A3B-Base-BF16`).
- **Lightning (11 Aug 2026)** was trained with this data (the set ships as its RL data; its card names NeMo RL + NeMo Gym GRPO but no nodes/steps and no pre-RL checkpoint). It is the *contaminated comparator*: its audit row is reported separately, never pooled.
- **Rebuilder: Nemotron 3 Super** on Token Factory (reads the ~10k-token histories, writes workspace and verifier candidates).
- **Continuation teacher:** the original expert GLM-5.1 if servable (closed-loop, BF16), with **Nemotron 3 Ultra** as the NVIDIA-native comparator; both reported if both run.
- **Teacher confirmed servable (2026-09-23):** `zai-org/GLM-5.1` is in the Token Factory catalog (with 5.2, 5.3, 5.3-Flash). Primary continuation teacher = GLM-5.1 closed-loop; `nvidia/Nemotron-3-Ultra-550b-a55b` is the NVIDIA-native comparator. `Qwen/Qwen3-30B-A3B-Instruct-2507` is served (no Thinking variant), an option for a second uncontaminated student.
