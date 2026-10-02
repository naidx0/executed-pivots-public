"""A tiny OpenAI-compatible server on localhost that stands in for Token Factory in tests and dry runs.

    python -m pivots.students.mock_server --port 8913 [--script 429,200,402]

It answers GET /v1/models and POST /v1/chat/completions. `script` is a list of HTTP statuses (or
(status, error body) pairs) for the successive chat requests (then 200 for ever): 429 and 5xx carry a Retry-After of 0, 402 carries a
billing error body. Replies are Terminus-2 actions, cycled from a small fixed set, with a
reasoning_content field the way Nemotron reasoning models return it. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _act(analysis: str, *cmds: str, done: bool = False) -> str:
    return json.dumps({"analysis": analysis, "plan": "mock plan",
                       "commands": [{"keystrokes": c, "duration": 0.1} for c in cmds], "task_complete": done})


ACTIONS = [
    _act("Look around first.", "ls -la\n", "pwd\n"),
    _act("Check the environment.", "env | sort | head -n 20\n"),
    _act("Nothing to do yet."),
    _act("Show the directory tree.", "find . -maxdepth 2 -type f | head -n 30\n"),
]
MODELS = ["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", "nvidia/nemotron-3-super-120b-a12b", "zai-org/GLM-5.1"]
ERROR_BODIES = {
    401: {"error": {"code": "invalid_api_key", "message": "Invalid API key"}},
    402: {"error": {"code": "insufficient_quota", "message": "Insufficient balance: add a payment method"}},
    403: {"error": {"code": "forbidden", "message": "Forbidden"}},
    429: {"error": {"code": "rate_limit", "message": "Too many requests, slow down"}},
    500: {"error": {"code": "server_error", "message": "Internal error"}},
    503: {"error": {"code": "unavailable", "message": "Service unavailable"}},
}


class MockTokenFactory:
    def __init__(self, script: list[int] | None = None, port: int = 0, content_prefix: str = ""):
        self.script = list(script or [])
        self.content_prefix = content_prefix
        self.calls: list[dict] = []  # {path, status, auth} per request, kept in memory only
        self.chat_n = 0
        self._lock = threading.Lock()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), self._handler())

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def start(self) -> "MockTokenFactory":
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def _handler(self):
        mock = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, status, obj, headers=()):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for k, v in headers:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                with mock._lock:
                    mock.calls.append({"path": self.path, "status": 200, "auth": self.headers.get("Authorization")})
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in MODELS]})
                else:
                    self._send(404, {"error": {"message": "no route"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                with mock._lock:
                    status = mock.script.pop(0) if mock.script else 200
                    status, err = status if isinstance(status, tuple) else (status, None)
                    mock.calls.append({"path": self.path, "status": status, "auth": self.headers.get("Authorization"),
                                       "model": body.get("model"), "n_messages": len(body.get("messages", []))})
                    n = mock.chat_n
                    if status == 200:
                        mock.chat_n += 1
                if status != 200:
                    self._send(status, err or ERROR_BODIES.get(status, {"error": {"message": f"HTTP {status}"}}),
                               headers=[("Retry-After", "0")] if status in (429, 500, 502, 503, 504) else [])
                    return
                self._send(200, {
                    "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
                    "model": body.get("model"),
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": mock.content_prefix + ACTIONS[n % len(ACTIONS)],
                        "reasoning_content": f"mock reasoning {n}"}}],
                    "usage": {"prompt_tokens": 100 * len(body.get("messages", [])), "completion_tokens": 50,
                              "total_tokens": 100 * len(body.get("messages", [])) + 50}})

        return H


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8913)
    ap.add_argument("--script", default="", help="comma-separated statuses for successive chat requests")
    a = ap.parse_args(argv)
    m = MockTokenFactory([int(s) for s in a.script.split(",") if s], port=a.port)
    print(f"mock Token Factory on {m.url}", flush=True)
    try:
        m.httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
