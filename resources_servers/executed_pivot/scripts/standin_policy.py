"""Stand-in policy for executed_pivot rollouts: an OpenAI-compatible server that replays stored actions.

NeMo Gym's model servers (responses_api_models/openai_model, vllm_model) forward each policy call to
`<policy_base_url>/responses` or `<policy_base_url>/chat/completions`. This server answers both with a
Terminus-2 JSON action taken from an audit rows file (pivots/audit output: one row per labelled action,
with `task`, `turn`, `action`, `text` and the labels J, X), so the rewards Gym reports can be checked
against labels that are already known. No model, no GPU, no network beyond localhost.

The pivot is recovered from the prompt itself: the task is the specimen whose instruction appears in the
first user message, and the turn is the number of assistant messages already in the conversation (a
pivot at turn t carries 1 + 2t messages). Unrecognised prompts get a no-op action.

Which stored action answers a pivot (--pick):
    expert              the expert's own batch (ctrl:expert)
    student | control   a random student sample | a random non-expert control, per --seed
    random              any stored action for the pivot, per --seed
    cycle               the k-th request for a pivot gets its k-th stored action (sorted by id, wrapping);
                        with Gym's num_repeats >= the largest pool, every stored action is replayed
    <action id>         that action (e.g. ctrl:flipped, student:opus:3); falls back to the expert if absent

Every response names what it replayed: `metadata.standin_pivot` / `metadata.standin_action` on a
Responses API reply (Gym keeps it in the rollout's `response`), and one line per request in --log.

    python resources_servers/executed_pivot/scripts/standin_policy.py \
        --actions out/audit/rows.jsonl --pick cycle --port 8911 --log standin.jsonl

then point Gym at it: policy_base_url=http://127.0.0.1:8911/v1 policy_api_key=dummy policy_model_name=standin.

--from-rows <train.jsonl> takes the expert actions from dataset rows (prepare.py output) instead of an audit
rows file. --style shapes a chat-completions reply the way local thinking models do: `content` (the bare JSON),
`reasoning` (content empty, the JSON at the end of message.reasoning, as Ollama returns thinking models) or
`think` (a <think> block, prose and a ```json fence in content). gym uses these as a CPU stand-in for a
local model when smoke-testing gym/run_local_policy.sh.
Standard library only, so it runs under any Python 3.10+.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import threading
import time
import uuid as uuidlib
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PICKS = ("expert", "student", "control", "random", "cycle")
NOOP_ACTION = json.dumps(
    {"analysis": "stand-in: pivot not recognised", "plan": "do nothing", "commands": [], "task_complete": False}
)


def load_instructions(specimens_dir: str | Path) -> dict[str, str]:
    """task name -> instruction, from specimens/*/task.json."""
    out = {}
    for p in sorted(Path(specimens_dir).glob("*/task.json")):
        meta = json.loads(p.read_text())
        out[meta["name"]] = meta["instruction"]
    return out


def load_actions(path: str | Path) -> dict[str, dict[str, dict]]:
    """'task:turn' -> {action id -> row} from an audit rows file."""
    pools: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                pools[f"{r['task']}:{int(r['turn'])}"][r["action"]] = r
    return dict(pools)


def actions_from_rows(path: str | Path) -> dict[str, dict[str, dict]]:
    """'task:turn' -> {"ctrl:expert": row} from dataset rows (expected_answer is the expert's action)."""
    pools: dict[str, dict[str, dict]] = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                md = r["metadata"]
                pools[f"{md['task']}:{int(md['turn'])}"] = {"ctrl:expert": {
                    "task": md["task"], "turn": md["turn"], "action": "ctrl:expert", "text": r["expected_answer"]}}
    return pools


STYLES = ("content", "reasoning", "think")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


def messages_of(body: dict) -> list[tuple[str, str]]:
    """(role, text) for the conversation of a Responses API or Chat Completions request body."""
    items = body.get("input", body.get("messages")) or []
    if isinstance(items, str):
        return [("user", items)]
    out = []
    for m in items:
        if isinstance(m, dict) and m.get("role") and m.get("type", "message") == "message":
            out.append((m["role"], _content_text(m.get("content"))))
    return out


class ReplayPolicy:
    def __init__(self, actions: dict[str, dict[str, dict]], instructions: dict[str, str], pick: str = "expert",
                 seed: int = 0):
        if pick not in PICKS and ":" not in pick:
            raise ValueError(f"--pick must be one of {PICKS} or an action id like ctrl:flipped, got {pick!r}")
        self.actions, self.instructions, self.pick, self.seed = actions, instructions, pick, seed
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def pivot_of(self, msgs: list[tuple[str, str]]) -> str | None:
        first_user = next((t for r, t in msgs if r == "user"), "")
        # Longest instruction first, so an instruction that prefixes another cannot shadow it.
        task = next((n for n, ins in sorted(self.instructions.items(), key=lambda kv: -len(kv[1]))
                     if ins and ins in first_user), None)
        if task is None:
            return None
        return f"{task}:{sum(1 for r, _ in msgs if r == 'assistant')}"

    def choose(self, pivot: str | None) -> tuple[str, str]:
        """(action id, Terminus-2 text) for this request."""
        pool = self.actions.get(pivot or "", {})
        if not pool:
            return "unmatched", NOOP_ACTION
        with self._lock:
            k = self._counts[pivot]
            self._counts[pivot] += 1
        ids = sorted(pool)
        rng = random.Random(int(hashlib.sha256(f"{self.seed}:{pivot}:{k}".encode()).hexdigest()[:16], 16))
        if self.pick == "cycle":
            aid = ids[k % len(ids)]
        elif self.pick == "random":
            aid = rng.choice(ids)
        elif self.pick in ("student", "control"):
            prefix = "student:" if self.pick == "student" else "ctrl:"
            cands = [a for a in ids if a.startswith(prefix) and a != "ctrl:expert"]
            aid = rng.choice(cands) if cands else "ctrl:expert"
        else:  # expert or an exact action id
            aid = "ctrl:expert" if self.pick == "expert" else self.pick
        if aid not in pool:
            aid = "ctrl:expert"
        return aid, pool[aid]["text"]


def _responses_reply(body: dict, text: str, pivot: str | None, aid: str) -> dict:
    return {
        "id": f"resp_{uuidlib.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "model": body.get("model") or "standin",
        "status": "completed",
        "output": [{
            "id": f"msg_{uuidlib.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "metadata": {"standin_pivot": pivot or "", "standin_action": aid},
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                  "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
    }


def _chat_reply(body: dict, text: str, pivot: str | None, aid: str, style: str = "content") -> dict:
    message = {"role": "assistant", "content": text}
    if style == "reasoning":
        message = {"role": "assistant", "content": "", "reasoning": f"Replaying the stored action.\n{text}"}
    elif style == "think":
        message["content"] = f"<think>Replaying the stored action.</think>\nMy next step:\n```json\n{text}\n```"
    return {
        "id": f"chatcmpl-{uuidlib.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model") or "standin",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        # Not part of the OpenAI schema; Gym's chat path drops it, so --log is the record there.
        "standin": {"pivot": pivot or "", "action": aid},
    }


def make_server(policy: ReplayPolicy, host: str = "127.0.0.1", port: int = 0, log_path: str | None = None,
                quiet: bool = True, style: str = "content") -> ThreadingHTTPServer:
    log_lock = threading.Lock()
    log_fh = open(log_path, "a") if log_path else None

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: D401 - silence per-request stderr lines
            if not quiet:
                super().log_message(fmt, *args)

        def _send(self, status: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._send(200, {"object": "list", "data": [{"id": "standin", "object": "model", "owned_by": "local"}]})
            elif self.path in ("/", "/health"):
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self):
            t0 = time.time()
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError as e:
                self._send(400, {"error": {"message": f"bad json: {e}"}})
                return
            route = self.path.rstrip("/")
            if route.endswith("/responses"):
                build = _responses_reply
            elif route.endswith("/chat/completions"):
                def build(b, t, p, a):
                    return _chat_reply(b, t, p, a, style)
            else:
                self._send(404, {"error": {"message": f"no route {self.path}"}})
                return
            pivot = policy.pivot_of(messages_of(body))
            aid, text = policy.choose(pivot)
            self._send(200, build(body, text, pivot, aid))
            if log_fh:
                with log_lock:
                    log_fh.write(json.dumps({"t": t0, "route": route, "pivot": pivot, "action": aid}) + "\n")
                    log_fh.flush()

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--actions", help="audit rows.jsonl (task, turn, action, text, labels)")
    src.add_argument("--from-rows", help="dataset rows (prepare.py train.jsonl): replay the expert's actions")
    ap.add_argument("--style", choices=STYLES, default="content", help="chat-completions reply shape (see above)")
    ap.add_argument("--specimens", default=str(REPO_ROOT / "specimens"), help="dir of specimen task.json files")
    ap.add_argument("--pick", default="expert", help=f"one of {', '.join(PICKS)}, or an action id")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8911)
    ap.add_argument("--log", help="append one JSON line per request (pivot, action) here")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    actions = load_actions(a.actions) if a.actions else actions_from_rows(a.from_rows)
    policy = ReplayPolicy(actions, load_instructions(a.specimens), a.pick, a.seed)
    srv = make_server(policy, a.host, a.port, a.log, quiet=not a.verbose, style=a.style)
    n = sum(len(p) for p in policy.actions.values())
    print(f"stand-in policy on http://{a.host}:{srv.server_port}/v1 (pick={a.pick}, seed={a.seed}; "
          f"{n} actions over {len(policy.actions)} pivots)", file=sys.stderr, flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
