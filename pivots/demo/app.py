"""FastAPI app for the live demo: one static page and a small JSON API.

    GET  /                    the page (pivots/demo/static/index.html)
    GET  /api/pivots          decisive pivots and the one-click examples
    GET  /api/pivot/{id}      terminal so far, the expert's batch, stored students and controls
    POST /api/run             {pivot, source | keystrokes (+task_complete) | action, verify}
    GET  /api/health          readiness, limits, sandbox load

Limits on untrusted input: request body size, keystroke and action length, a per-batch timeout inside
the sandbox, a hard cap on every sandbox run, and a concurrency limit with a short queue (429 beyond it).
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from pivots.demo.core import EXAMPLES, BadRequest, Busy, DemoCore

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 64 * 1024


class RunRequest(BaseModel):
    pivot: str = Field(max_length=120)
    source: str | None = Field(default=None, max_length=120)
    keystrokes: str | None = None
    task_complete: bool = False
    action: str | None = None
    verify: bool = True


def create_app(core: DemoCore, *, warm: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        if warm:  # build the six worlds and prepare the teacher runs in the background
            threading.Thread(target=core.warm, name="warm", daemon=True).start()
        yield

    app = FastAPI(title="Executed Pivots: try a step", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    app.state.core = core

    @app.middleware("http")
    async def limit_body(request: Request, call_next):
        if request.method == "POST":
            n = request.headers.get("content-length")
            if n is None or not n.isdigit():
                return JSONResponse({"error": "a Content-Length header is required"}, status_code=411)
            if int(n) > MAX_BODY:
                return JSONResponse({"error": f"request body over {MAX_BODY} bytes"}, status_code=413)
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", media_type="text/html; charset=utf-8",
                            headers={"Cache-Control": "no-cache"})

    @app.get("/api/health")
    def health():
        return core.status()

    @app.get("/api/pivots")
    def pivots():
        return {"pivots": core.pivot_list(), "examples": EXAMPLES, "source": core.data.get("source"),
                "limits": core.status()["limits"]}

    @app.get("/api/pivot/{key}")
    async def pivot(key: str):
        try:
            return await run_in_threadpool(core.pivot_detail, key)
        except BadRequest as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        except Busy as e:
            return JSONResponse({"error": str(e)}, status_code=429, headers={"Retry-After": "5"})
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    @app.post("/api/run")
    async def run(req: RunRequest):
        try:
            text = core.action_text(req.pivot, source=req.source, keystrokes=req.keystrokes,
                                    task_complete=req.task_complete, action=req.action)
            return await run_in_threadpool(core.run, req.pivot, text, with_p=req.verify)
        except BadRequest as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Busy as e:
            return JSONResponse({"error": str(e)}, status_code=429, headers={"Retry-After": "5"})
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=503)

    return app


def app_from_env() -> FastAPI:
    """Uvicorn factory: `uvicorn --factory pivots.demo.app:app_from_env`. Settings come from XP_DEMO_* env."""
    from pivots.demo.sandbox import Limits
    from pivots.demo.core import InputLimits

    store = os.environ.get("XP_DEMO_STORE", "/tmp/xp-demo-store")
    limits = Limits(batch_timeout_s=int(os.environ.get("XP_DEMO_TIMEOUT", "10")))
    core = DemoCore(store, limits=limits, inputs=InputLimits(),
                    max_concurrent=int(os.environ.get("XP_DEMO_CONCURRENCY", "2")),
                    max_queue=int(os.environ.get("XP_DEMO_QUEUE", "6")),
                    hardened=os.environ.get("XP_DEMO_HARDENED", "1") == "1",  # H18/H19: hardened by default
                    pivots=os.environ.get("XP_DEMO_PIVOTS", "decisive"))  # "all": every turn can be opened
    return create_app(core, warm=os.environ.get("XP_DEMO_WARM", "1") == "1")
