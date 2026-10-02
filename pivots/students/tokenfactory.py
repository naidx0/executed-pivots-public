"""Nebius Token Factory as a student: sample Terminus-2 actions from a Nebius-served model.

Token Factory speaks the OpenAI chat-completions API at https://api.tokenfactory.nebius.com/v1.
This client is standard library only (urllib), so it runs under the audit's interpreter with no
extra install. It is built to spend as little as possible and never by surprise:

  budget    every HTTP attempt (retries included, cache hits not) counts against --max-requests
            (default 400); the next attempt past it raises BudgetExhausted.
  stop      401, 402, 403, a billing or quota error on any status, or the egress proxy refusing
            the tunnel stops the client for good: it raises TokenFactoryStop, and every later call
            raises the same error without touching the network. These are never retried.
  retry     429 (plain rate limit) and 5xx, and dropped connections, back off exponentially
            (Retry-After honoured, capped at 60 s), at most --max-retries times per request.
  cache     each response is kept on disk under sha256(model, messages, params, sample index), so a
            rerun of the same sampling costs nothing. A ledger (requests.jsonl) records every
            attempt: time, route, status, prompt hash. Never the key.

The key is read at runtime from $NEBIUS_API_KEY, else from /root/.config/nebius/tf_key. It is only
ever placed in the Authorization header; errors and logs pass through redact() first.

    python -m pivots.students.tokenfactory --list-models
    python -m pivots.students.tokenfactory --smoke --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
    python -m pivots.students.tokenfactory --serve 8912    # OpenAI-compatible proxy for NeMo Gym

The audit uses it through `python -m pivots.audit.run ... --student tokenfactory:<model>`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

BASE_URL = "https://api.tokenfactory.nebius.com/v1"
# The plan's student: uncontaminated by the Terminal-Pivot data, and the base of the training arm.
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
KEY_FILE = "/root/.config/nebius/tf_key"
DEFAULT_CACHE = os.environ.get("TOKENFACTORY_CACHE", os.path.expanduser("~/.cache/cleave/tokenfactory"))
# Plan §5 (audit sampling): temperature 1.0, top_p 0.95. max_tokens leaves room for reasoning.
DEFAULT_PARAMS = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 8192}

STOP_STATUSES = (401, 402, 403)
# Billing / quota wording, matched case-insensitively on any error body. The key has no card, so a
# quota message is treated as final even when it arrives as a 429.
BILLING_RE = re.compile(r"billing|quota|insufficient|credit|balance|payment|budget|exceeded your|spend limit|"
                        r"subscription|top[ -]?up", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class TokenFactoryError(RuntimeError):
    """A request failed and was not retried (e.g. 400 or 404). The client can still be used."""


class TokenFactoryStop(TokenFactoryError):
    """Final: auth, billing, quota, egress or budget. Every later call raises this again."""

    def __init__(self, kind: str, message: str):
        super().__init__(f"Token Factory stopped ({kind}): {message}")
        self.kind = kind


class BudgetExhausted(TokenFactoryStop):
    def __init__(self, limit: int):
        super().__init__("budget", f"the request budget of {limit} is spent (raise --max-requests to allow more)")


def load_key(path: str = KEY_FILE) -> str:
    key = os.environ.get("NEBIUS_API_KEY", "").strip()
    if not key and os.path.exists(path):
        key = Path(path).read_text().strip()
    if not key:
        raise TokenFactoryStop("no_key", f"set NEBIUS_API_KEY or put the key in {path}")
    return key


def prompt_hash(model: str, messages: list[dict], params: dict, sample: int) -> str:
    blob = json.dumps({"model": model, "messages": messages, "params": params, "sample": sample},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def classify(status: int, body: str) -> str:
    """'stop' (never retry), 'retry' (back off and try again) or 'fail' (this request only)."""
    if status in STOP_STATUSES or (status >= 400 and BILLING_RE.search(body or "")):
        return "stop"
    if status == 429 or status >= 500:
        return "retry"
    return "fail"


def split_reasoning(message: dict) -> tuple[str, str]:
    """(action text, reasoning) from a chat message: reasoning_content / reasoning fields, or <think> blocks."""
    content = message.get("content") or ""
    if isinstance(content, list):
        content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    thought = _THINK_RE.findall(content)
    if thought:
        reasoning = "\n".join([reasoning, *thought]).strip()
        content = _THINK_RE.sub("", content)
    elif "</think>" in content:  # template opened <think> in the prompt; only the close is in the output
        head, content = content.split("</think>", 1)
        reasoning = "\n".join([reasoning, head]).strip()
    return content.strip(), reasoning


class TokenFactory:
    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = BASE_URL, api_key: str | None = None,
                 max_requests: int = 400, max_retries: int = 5, backoff_s: float = 2.0, timeout_s: float = 300,
                 cache_dir: str | None = DEFAULT_CACHE, key_file: str = KEY_FILE,
                 sleep: Callable[[float], None] = time.sleep):
        self.model, self.base_url = model, base_url.rstrip("/")
        self._key = api_key
        self._key_file = key_file
        self.max_requests, self.max_retries, self.backoff_s, self.timeout_s = max_requests, max_retries, backoff_s, timeout_s
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.sleep = sleep
        self.requests = 0  # HTTP attempts made by this client
        self.cache_hits = 0
        self._stopped: TokenFactoryStop | None = None
        self._lock = threading.Lock()
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # --- key handling -------------------------------------------------------------------------
    @property
    def key(self) -> str:
        if self._key is None:
            self._key = load_key(self._key_file)
        return self._key

    def redact(self, text: str) -> str:
        if self._key:
            text = text.replace(self._key, "[redacted]")
        return re.sub(r"(Bearer\s+)\S+", r"\1[redacted]", text)

    # --- transport ----------------------------------------------------------------------------
    def _ledger(self, route: str, status: Any, h: str | None = None) -> None:
        if not self.cache_dir:
            return
        with self._lock, open(self.cache_dir / "requests.jsonl", "a") as fh:
            fh.write(json.dumps({"t": round(time.time(), 3), "route": route, "status": status, "model": self.model,
                                 "prompt": h}) + "\n")

    def _take_budget(self) -> None:
        with self._lock:
            if self._stopped:
                raise self._stopped
            if self.requests >= self.max_requests:
                self._stopped = BudgetExhausted(self.max_requests)
                raise self._stopped
            self.requests += 1

    def _stop(self, kind: str, message: str) -> TokenFactoryStop:
        with self._lock:
            if not self._stopped:
                self._stopped = TokenFactoryStop(kind, self.redact(message))
            return self._stopped

    def _request(self, method: str, route: str, body: dict | None = None, h: str | None = None) -> dict:
        url = f"{self.base_url}{route}"
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(self.max_retries + 1):
            self._take_budget()
            req = urllib.request.Request(url, data=data, method=method, headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json", "Accept": "application/json"})
            wait = None
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8", "replace")
                self._ledger(route, 200, h)
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    raise TokenFactoryError(f"{route}: response is not JSON: {self.redact(raw[:300])}") from None
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace") if e.fp else ""
                self._ledger(route, e.code, h)
                verdict = classify(e.code, text)
                msg = f"HTTP {e.code} on {method} {route}: {text[:500]}"
                if verdict == "stop":
                    raise self._stop("billing_or_quota" if BILLING_RE.search(text) else f"http_{e.code}", msg) from None
                if verdict == "fail" or attempt == self.max_retries:
                    raise TokenFactoryError(self.redact(msg + ("" if verdict == "fail" else
                                                              f" (gave up after {attempt + 1} attempts)"))) from None
                ra = e.headers.get("Retry-After") if e.headers else None
                wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else None
            except urllib.error.URLError as e:
                reason = str(e.reason)
                self._ledger(route, f"urlerror:{reason[:80]}", h)
                if "Tunnel connection failed" in reason or "403" in reason:
                    raise self._stop("egress_blocked", f"the network path to {self.base_url} is refused ({reason}); "
                                     "this container's egress policy must allow the host") from None
                if attempt == self.max_retries:
                    raise TokenFactoryError(self.redact(f"{route}: {reason} after {attempt + 1} attempts")) from None
            except (TimeoutError, ConnectionError) as e:
                self._ledger(route, f"conn:{type(e).__name__}", h)
                if attempt == self.max_retries:
                    raise TokenFactoryError(f"{route}: {type(e).__name__} after {attempt + 1} attempts") from None
            delay = wait if wait is not None else self.backoff_s * (2 ** attempt) * (0.5 + random.random())
            self.sleep(min(delay, 60.0))
        raise AssertionError("unreachable")

    # --- API ------------------------------------------------------------------------------------
    def list_models(self) -> list[str]:
        out = self._request("GET", "/models")
        return sorted(m.get("id", "") for m in out.get("data", []))

    def chat(self, messages: list[dict], sample: int = 0, **params) -> dict:
        """One chat completion (the raw response JSON, plus `_cached`). Cached per (prompt, params, sample)."""
        params = {**DEFAULT_PARAMS, **params}
        h = prompt_hash(self.model, messages, params, sample)
        path = self.cache_dir / f"{h}.json" if self.cache_dir else None
        if path and path.exists():
            with self._lock:
                self.cache_hits += 1
            return {**json.loads(path.read_text()), "_cached": True, "_prompt_hash": h}
        out = self._request("POST", "/chat/completions", {"model": self.model, "messages": messages, **params}, h)
        if not out.get("choices"):
            raise TokenFactoryError(f"no choices in the response: {self.redact(json.dumps(out)[:300])}")
        if path:
            tmp = path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
            tmp.write_text(json.dumps(out))
            tmp.replace(path)
        return {**out, "_cached": False, "_prompt_hash": h}

    def sample(self, messages: list[dict], n: int, **params) -> list[dict]:
        """n independent samples for one prompt: [{text, reasoning, finish_reason, usage, cached, prompt_hash}]."""
        rows = []
        for k in range(n):
            out = self.chat(messages, sample=k, **params)
            choice = out["choices"][0]
            text, reasoning = split_reasoning(choice.get("message") or {})
            rows.append({"k": k, "text": text, "reasoning": reasoning, "finish_reason": choice.get("finish_reason"),
                         "usage": out.get("usage"), "cached": out["_cached"], "prompt_hash": out["_prompt_hash"]})
        return rows

    def stats(self) -> dict:
        return {"model": self.model, "requests": self.requests, "cache_hits": self.cache_hits,
                "max_requests": self.max_requests, "stopped": str(self._stopped) if self._stopped else None}


def pool_name(model: str) -> str:
    """A pool id for audit action names (student:<pool>:<i>): no ':' allowed."""
    return "tf-" + re.sub(r"[^a-z0-9.]+", "-", model.rsplit("/", 1)[-1].lower()).strip("-")


# --- OpenAI-compatible proxy for NeMo Gym ------------------------------------------------------------
def make_proxy(client: TokenFactory, host: str = "127.0.0.1", port: int = 8912, quiet: bool = True) -> ThreadingHTTPServer:
    """Serve /v1/chat/completions and /v1/models, forwarding to Token Factory through `client`.

    Gym's model server points here with any api_key: the real key stays in this process. The k-th identical
    request is sample k, so num_repeats > 1 draws distinct samples and a rerun replays them from the cache.
    Once the client stops (billing, auth, budget), every call gets a 429 with error.code "budget_exceeded",
    which Gym's HTTP client treats as a spent key and stops calling.
    """
    seen: dict[str, int] = defaultdict(int)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if not quiet:
                sys.stderr.write(client.redact(fmt % args) + "\n")

        def _send(self, status: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, e: Exception) -> None:
            if isinstance(e, TokenFactoryStop):
                self._send(429, {"error": {"code": "budget_exceeded", "type": e.kind, "message": str(e)}})
            else:
                self._send(502, {"error": {"code": "upstream_error", "message": client.redact(str(e))}})

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                try:
                    self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in client.list_models()]})
                except TokenFactoryError as e:
                    self._error(e)
            elif self.path in ("/", "/health"):
                self._send(200, {"status": "ok", **client.stats()})
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
            params = {**DEFAULT_PARAMS, **body}
            base = prompt_hash(client.model, messages, params, -1)
            with lock:
                k = seen[base]
                seen[base] += 1
            try:
                out = client.chat(messages, sample=k, **body)
            except TokenFactoryError as e:
                self._error(e)
                return
            out.pop("_cached", None)
            out.pop("_prompt_hash", None)
            self._send(200, out)

    return ThreadingHTTPServer((host, port), Handler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=os.environ.get("TOKENFACTORY_BASE_URL", BASE_URL))
    ap.add_argument("--max-requests", type=int, default=400, help="hard cap on HTTP attempts (retries count)")
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--key-file", default=KEY_FILE)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list-models", action="store_true", help="GET /models (1 request)")
    g.add_argument("--smoke", action="store_true", help="one short chat completion (1 request, or 0 if cached)")
    g.add_argument("--serve", type=int, metavar="PORT", help="OpenAI-compatible proxy for NeMo Gym on 127.0.0.1:PORT")
    ap.add_argument("--out", help="write the result (models list or smoke reply) here as JSON")
    a = ap.parse_args(argv)
    client = TokenFactory(a.model, a.base_url, max_requests=a.max_requests, max_retries=a.max_retries,
                          cache_dir=a.cache_dir, key_file=a.key_file)
    try:
        if a.list_models:
            models = client.list_models()
            result = {"models": models}
            print("\n".join(models))
            nem = [m for m in models if "nemotron" in m.lower()]
            print(f"# {len(models)} models, {len(nem)} Nemotron; default student {a.model} "
                  f"{'is' if a.model in models else 'is NOT'} listed", file=sys.stderr)
        elif a.smoke:
            msgs = [{"role": "user", "content": 'Reply with exactly this JSON and nothing else: {"ok": true}'}]
            out = client.chat(msgs, max_tokens=512, temperature=0.0)
            text, reasoning = split_reasoning(out["choices"][0]["message"])
            result = {"model": a.model, "text": text, "reasoning_chars": len(reasoning), "usage": out.get("usage"),
                      "cached": out["_cached"]}
            print(json.dumps(result))
        else:
            srv = make_proxy(client, port=a.serve)
            print(f"Token Factory proxy for {a.model} on http://127.0.0.1:{srv.server_port}/v1 "
                  f"(budget {a.max_requests} requests)", file=sys.stderr, flush=True)
            try:
                srv.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                srv.server_close()
            result = None
    except TokenFactoryStop as e:
        print(f"STOP: {e}", file=sys.stderr)
        print(json.dumps(client.stats()), file=sys.stderr)
        return 3
    except TokenFactoryError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if a.out and result is not None:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({**result, "stats": client.stats()}, indent=1))
    print(json.dumps(client.stats()), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
