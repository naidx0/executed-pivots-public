"""A scripted fake of a local OpenAI-compatible model server (llama-server / Ollama /v1) for tests.

Each POST /v1/chat/completions takes the next step of `script` (then `default` for ever). A step is a tuple:
    ("content", text)            message.content = text
    ("reasoning", text)          content "" and message.reasoning = text (Ollama's field for thinking models)
    ("reasoning_content", text)  content "" and message.reasoning_content = text (llama-server --reasoning-format)
    ("think", text)              content = "<think>...</think>" + prose + a ```json fence around text
    ("status", code)             that HTTP status with an error body
    ("sleep", seconds, text)     wait, then answer with content = text (a hung model, for timeouts)
Standard library only; binds 127.0.0.1 on a free port.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOpenAI:
    def __init__(self, script=None, default=("content", "{}"), host: str = "127.0.0.1"):
        self.script = list(script or [])
        self.default = default
        self.calls: list[dict] = []
        self._lock = threading.Lock()
        self.httpd = ThreadingHTTPServer((host, 0), self._handler())
        self.httpd.daemon_threads = True

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}/v1"

    def start(self) -> "FakeOpenAI":
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def _next(self):
        with self._lock:
            return self.script.pop(0) if self.script else self.default

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, status, obj):
                data = json.dumps(obj).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # the client timed out and hung up

            def do_GET(self):
                self._send(200, {"object": "list", "data": [{"id": "fake-local", "object": "model"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                step = fake._next()
                with fake._lock:
                    fake.calls.append({"auth": self.headers.get("Authorization"), "body": body, "step": step[0]})
                kind = step[0]
                msg = {"role": "assistant", "content": ""}
                if kind == "status":
                    self._send(step[1], {"error": {"message": f"fake {step[1]}"}})
                    return
                if kind == "sleep":
                    time.sleep(step[1])
                    msg["content"] = step[2]
                elif kind == "content":
                    msg["content"] = step[1]
                elif kind in ("reasoning", "reasoning_content"):
                    msg[kind] = step[1]
                elif kind == "think":
                    msg["content"] = f"<think>I should look first.</think>\nHere is my action:\n```json\n{step[1]}\n```\n"
                self._send(200, {"id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()),
                                 "model": body.get("model"), "choices": [{"index": 0, "finish_reason": "stop",
                                                                          "message": msg}],
                                 "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})

        return H
