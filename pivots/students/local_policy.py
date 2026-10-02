"""A local model (llama-server, or Ollama's /v1) as the policy of executed_pivot rollouts.

NeMo Gym's policy_model talks OpenAI chat completions to `<base_url>/chat/completions`. This module sits
between Gym and a local OpenAI-compatible server and makes a small local model's replies usable as
Terminus-2 actions:

  endpoint  the upstream base URL and model come from the environment, never from a file:
                LOCAL_POLICY_BASE_URL   e.g. http://127.0.0.1:8080/v1 (llama-server) or
                                        http://127.0.0.1:11434/v1 (Ollama)
                LOCAL_POLICY_MODEL      the served model name (llama-server ignores it; Ollama needs it)
                LOCAL_POLICY_API_KEY    optional; sent as a Bearer token only when set
  timeout   every upstream attempt has a wall-clock timeout (LOCAL_POLICY_TIMEOUT, default 300 s).
  retry     a timeout, a dropped connection, 429 or 5xx is retried (LOCAL_POLICY_RETRIES, default 2 extra
            attempts) with exponential backoff; any other 4xx fails that request at once.
  action    the reply is reduced to the Terminus-2 JSON the executed_pivot server scores:
                - <think>...</think> blocks and the reasoning / reasoning_content fields become reasoning;
                - the action is the last valid Terminus-2 object in the content (code fences and prose
                  around it are tolerated, as the Terminus-2 harness tolerates them);
                - a thinking model that put its answer only in `reasoning` (content empty or not an
                  action) has the action taken from the reasoning instead;
                - a reply with no valid action anywhere is passed through unchanged, so executed_pivot
                  pays it 0 as model_output_invalid (a policy failure, not masked).
            J and X both see the text this module passes on.

    python -m pivots.students.local_policy --check            # GET <base>/models, 1 request
    python -m pivots.students.local_policy --smoke            # one tiny chat completion
    python -m pivots.students.local_policy --serve 8913 --log out/gym/policy.jsonl

Gym's config for the proxy is resources_servers/executed_pivot/configs/local_policy.yaml. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from pivots.effect.terminus import parse_action
from pivots.students.tokenfactory import split_reasoning

ENV_BASE_URL, ENV_MODEL, ENV_KEY = "LOCAL_POLICY_BASE_URL", "LOCAL_POLICY_MODEL", "LOCAL_POLICY_API_KEY"
ENV_TIMEOUT, ENV_RETRIES, ENV_MAX_TOKENS = "LOCAL_POLICY_TIMEOUT", "LOCAL_POLICY_RETRIES", "LOCAL_POLICY_MAX_TOKENS"
DEFAULT_PORT = 8913
# Sampling as the plan's audit students (temperature 1.0, top_p 0.95); max_tokens leaves room for thinking
# on an 8 GB card's context. A request's own values win.
DEFAULT_PARAMS = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 4096}
RETRY_STATUSES = (429, 500, 502, 503, 504)


class LocalPolicyError(RuntimeError):
    """The upstream failed this request (after the retries, or at once for a 4xx other than 429)."""


@dataclass
class Extracted:
    text: str  # what the policy "said": the Terminus-2 JSON when one was found, else the raw content
    reasoning: str
    source: str  # content | reasoning | invalid | empty
    error: str | None = None


def extract_action(message: dict) -> Extracted:
    """The Terminus-2 action in one chat message, wherever a local (thinking) model put it."""
    content, reasoning = split_reasoning(message or {})
    if content:
        pa = parse_action(content)
        if pa.action is not None:
            text = content if pa.strict_ok else json.dumps(pa.action.raw)
            return Extracted(text, reasoning, "content")
    if reasoning:
        pr = parse_action(reasoning)
        if pr.action is not None:
            return Extracted(json.dumps(pr.action.raw), reasoning, "reasoning")
    if not content:
        return Extracted("", reasoning, "empty", "no content and no action in the reasoning")
    return Extracted(content, reasoning, "invalid", parse_action(content).error)


def settings_from_env(env: dict | None = None) -> dict:
    """Upstream settings from the environment. The key is only ever read here, never written anywhere."""
    env = os.environ if env is None else env
    base = env.get(ENV_BASE_URL, "").strip()
    if not base:
        raise LocalPolicyError(f"set {ENV_BASE_URL} (e.g. http://127.0.0.1:8080/v1 for llama-server, "
                               "http://127.0.0.1:11434/v1 for Ollama)")
    return {"base_url": base, "model": env.get(ENV_MODEL, "").strip() or "local",
            "api_key": env.get(ENV_KEY, "").strip() or None,
            "timeout_s": float(env.get(ENV_TIMEOUT) or 300), "retries": int(env.get(ENV_RETRIES) or 2),
            "max_tokens": int(env.get(ENV_MAX_TOKENS) or DEFAULT_PARAMS["max_tokens"])}


class LocalPolicy:
    def __init__(self, base_url: str, model: str = "local", api_key: str | None = None, timeout_s: float = 300,
                 retries: int = 2, backoff_s: float = 1.0, max_tokens: int | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.base_url, self.model = base_url.rstrip("/"), model
        self._key = api_key
        self.timeout_s, self.retries, self.backoff_s, self.sleep = timeout_s, retries, backoff_s, sleep
        self.params = {**DEFAULT_PARAMS, **({"max_tokens": max_tokens} if max_tokens else {})}
        self.attempts = 0
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, env: dict | None = None, **kw) -> "LocalPolicy":
        return cls(**{**settings_from_env(env), **kw})

    def redact(self, text: str) -> str:
        return text.replace(self._key, "[redacted]") if self._key else text

    def _request(self, method: str, route: str, body: dict | None = None) -> tuple[dict, int]:
        """(JSON reply, attempts used)."""
        url = f"{self.base_url}{route}"
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        last = ""
        for attempt in range(self.retries + 1):
            with self._lock:
                self.attempts += 1
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8", "replace")
                try:
                    return json.loads(raw), attempt + 1
                except json.JSONDecodeError:
                    raise LocalPolicyError(f"{route}: reply is not JSON: {self.redact(raw[:300])}") from None
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace") if e.fp else ""
                last = f"HTTP {e.code}: {text[:300]}"
                if e.code not in RETRY_STATUSES:
                    raise LocalPolicyError(self.redact(f"{method} {route}: {last}")) from None
            except urllib.error.URLError as e:  # refused, reset, or a timeout wrapped by urllib
                last = f"{type(e.reason).__name__}: {e.reason}"
            except (TimeoutError, ConnectionError, OSError) as e:  # socket timeout while reading
                last = f"{type(e).__name__}: {e}"
            if attempt < self.retries:
                self.sleep(min(self.backoff_s * (2 ** attempt) * (0.5 + random.random()), 30.0))
        raise LocalPolicyError(self.redact(f"{method} {route}: {last} (gave up after {self.retries + 1} attempts, "
                                           f"timeout {self.timeout_s:g} s each)"))

    def models(self) -> list[str]:
        out, _ = self._request("GET", "/models")
        return sorted(m.get("id", "") for m in out.get("data", []) if isinstance(m, dict))

    def chat(self, messages: list[dict], **params) -> dict:
        """One completion, reduced: {text, reasoning, source, error, finish_reason, usage, attempts, gen_s, raw}."""
        body = {**self.params, **{k: v for k, v in params.items() if v is not None},
                "model": self.model, "messages": messages, "stream": False}
        t0 = time.perf_counter()
        out, attempts = self._request("POST", "/chat/completions", body)
        gen_s = round(time.perf_counter() - t0, 3)
        choices = out.get("choices") or []
        if not choices:
            raise LocalPolicyError(f"no choices in the reply: {self.redact(json.dumps(out)[:300])}")
        msg = choices[0].get("message") or {}
        ex = extract_action(msg)
        return {"text": ex.text, "reasoning": ex.reasoning, "source": ex.source, "error": ex.error,
                "finish_reason": choices[0].get("finish_reason"), "usage": out.get("usage") or {},
                "attempts": attempts, "gen_s": gen_s, "raw": out}


def prompt_hash(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


# --- OpenAI-compatible proxy for NeMo Gym ------------------------------------------------------------
def make_proxy(policy: LocalPolicy, host: str = "127.0.0.1", port: int = DEFAULT_PORT, log_path: str | None = None,
               quiet: bool = True) -> ThreadingHTTPServer:
    """Serve /v1/chat/completions and /v1/models, forwarding to the local model through `policy`.

    The reply's message carries the extracted action as `content` and the model's thinking as
    `reasoning_content` (Gym's uses_reasoning_parser turns it into a reasoning item, which executed_pivot
    does not score). An upstream failure after the retries is a 502 with error.code "upstream_error", so
    Gym records an infrastructure error instead of paying the policy 0. One JSON line per request goes to
    `log_path`: time, prompt hash, attempts, generation time, token usage, where the action came from.
    """
    log_lock = threading.Lock()
    log_fh = open(log_path, "a", encoding="utf-8") if log_path else None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def log(row: dict) -> None:
        if log_fh:
            with log_lock:
                log_fh.write(json.dumps(row) + "\n")
                log_fh.flush()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if not quiet:
                sys.stderr.write(policy.redact(fmt % args) + "\n")

        def _send(self, status: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                try:
                    self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in policy.models()]})
                except LocalPolicyError as e:
                    self._send(502, {"error": {"code": "upstream_error", "message": str(e)}})
            elif self.path in ("/", "/health"):
                self._send(200, {"status": "ok", "model": policy.model, "attempts": policy.attempts})
            else:
                self._send(404, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send(404, {"error": {"message": f"no route {self.path} (chat completions only)"}})
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except json.JSONDecodeError as e:
                self._send(400, {"error": {"message": f"bad json: {e}"}})
                return
            messages = body.pop("messages", [])
            for k in ("model", "stream", "stream_options"):
                body.pop(k, None)
            t0 = time.time()
            h = prompt_hash(messages)
            try:
                r = policy.chat(messages, **body)
            except LocalPolicyError as e:
                log({"t": round(t0, 3), "prompt": h, "status": "upstream_error", "error": str(e)})
                self._send(502, {"error": {"code": "upstream_error", "message": str(e)}})
                return
            raw = r["raw"]
            message = {"role": "assistant", "content": r["text"]}
            if r["reasoning"]:
                message["reasoning_content"] = r["reasoning"]
            self._send(200, {
                "id": raw.get("id") or f"chatcmpl-local-{h}",
                "object": "chat.completion",
                "created": raw.get("created") or int(t0),
                "model": raw.get("model") or policy.model,
                "choices": [{"index": 0, "finish_reason": r["finish_reason"] or "stop", "message": message}],
                "usage": r["usage"] or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                # Not part of the OpenAI schema (Gym drops it); pivots.gym.rollouts records it per rollout.
                "local_policy": {"source": r["source"], "error": r["error"], "attempts": r["attempts"],
                                 "gen_s": r["gen_s"], "reasoning_chars": len(r["reasoning"])},
            })
            log({"t": round(t0, 3), "prompt": h, "status": "ok", "source": r["source"], "error": r["error"],
                 "attempts": r["attempts"], "gen_s": r["gen_s"], "finish_reason": r["finish_reason"],
                 "usage": r["usage"], "reasoning_chars": len(r["reasoning"])})

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.log_fh = log_fh  # closed by the caller with the server
    return srv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="GET <base>/models (1 request)")
    g.add_argument("--smoke", action="store_true", help="one short chat completion")
    g.add_argument("--serve", type=int, metavar="PORT", help=f"proxy for NeMo Gym (default port {DEFAULT_PORT})")
    ap.add_argument("--host", default="127.0.0.1", help="bind address for --serve (0.0.0.0 inside a container)")
    ap.add_argument("--log", help="--serve: append one JSON line per request here")
    a = ap.parse_args(argv)
    try:
        policy = LocalPolicy.from_env()
        if a.check:
            print(json.dumps({"base_url": policy.base_url, "model": policy.model, "models": policy.models()}))
        elif a.smoke:
            msgs = [{"role": "user", "content": 'Reply with exactly this JSON and nothing else: {"analysis": "ok", '
                                                '"plan": "ok", "commands": [], "task_complete": false}'}]
            r = policy.chat(msgs, max_tokens=1024)
            print(json.dumps({k: r[k] for k in ("text", "source", "error", "finish_reason", "usage", "attempts",
                                                  "gen_s")} | {"reasoning_chars": len(r["reasoning"])}))
        else:
            srv = make_proxy(policy, a.host, a.serve, a.log)
            print(f"local policy proxy on http://{a.host}:{srv.server_port}/v1 -> {policy.base_url} "
                  f"(model {policy.model}, timeout {policy.timeout_s:g} s, {policy.retries} retries)",
                  file=sys.stderr, flush=True)
            try:
                srv.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                srv.server_close()
    except LocalPolicyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
