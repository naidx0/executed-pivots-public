"""Collect executed_pivot rollouts from a policy endpoint without a Gym install (the `direct` driver).

It does what Gym's simple_agent does for one executed_pivot row, in one process: send the row's Terminus-2
prompt to the policy's /v1/chat/completions, and score the reply with ExecutedPivotCore.verify, the same
function the resources server's /verify calls. Each rollout line carries Gym's verify fields (reward = X,
j_reward = J, score, mask_sample, failure_kind, penalties, verify_s) plus the policy's side (text, token
usage, generation time, where the action came from). `gym eval run` with configs/local_policy.yaml is the
Gym-served equivalent; pivots.gym.labels reads either file.

    python -m pivots.gym.rollouts --rows resources_servers/executed_pivot/data/train.jsonl \
        --anchors resources_servers/executed_pivot/data/anchors.jsonl --store /tmp/gym-rollouts \
        --policy-url http://127.0.0.1:8913/v1 --n 2 --out out/gym/run/rollouts.jsonl

A policy failure after the proxy's retries (HTTP 502) is recorded as masked with failure_kind
policy_error and is not verified: the environment did not get an answer to score.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from pivots.audit.run import parse_pivots


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


def failure_kind(detail: str | None) -> str | None:
    """Core failure label -> '<server>:<kind>', as the resources server's /verify reports it."""
    if not detail:
        return None
    kind = re.sub(r"[^a-z0-9_]+", "_", detail.split(":", 1)[0].strip().lower()).strip("_")
    return f"executed_pivot:{kind or 'unknown'}"


def chat_messages(row: dict) -> list[dict]:
    """The row's Responses-API input as chat messages ({role, content})."""
    items = row["responses_create_params"]["input"]
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    return [{"role": m["role"], "content": _text(m.get("content"))} for m in items
            if isinstance(m, dict) and m.get("role") and m.get("type", "message") == "message"]


def select_rows(rows: list[dict], pivots: str | None = None, limit: int | None = None) -> list[dict]:
    only = parse_pivots(pivots)
    if only is not None:
        rows = [r for r in rows if int(r["metadata"]["turn"]) in only.get(r["metadata"]["task"], set())]
    return rows[:limit] if limit else rows


def call_policy(url: str, messages: list[dict], timeout_s: float = 900, **params) -> dict:
    """POST one chat completion; {text, usage, finish_reason, local_policy} or {error}."""
    body = json.dumps({"model": "local", "messages": messages, **params}).encode()
    req = urllib.request.Request(f"{url.rstrip('/')}/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {(e.read() if e.fp else b'').decode('utf-8', 'replace')[:300]}"}
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    choice = (out.get("choices") or [{}])[0]
    return {"text": _text((choice.get("message") or {}).get("content")) or "", "usage": out.get("usage") or {},
            "finish_reason": choice.get("finish_reason"), "local_policy": out.get("local_policy") or {}}


def wait_for_policy(url: str, timeout_s: float = 60) -> bool:
    """True once GET <url>/models answers 200 (through the proxy, that also reaches the model server)."""
    deadline = time.time() + timeout_s
    while True:
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/models", timeout=10) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        if time.time() >= deadline:
            return False
        time.sleep(1)


def collect(rows: list[dict], n: int, policy_url: str, verify: Callable[[dict, str], dict], out_path: str | Path,
            concurrency: int = 2, params: dict | None = None, policy_timeout_s: float = 900) -> list[dict]:
    """n rollouts per row, written to out_path as they finish (a crash keeps what was collected)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("")
    lock = threading.Lock()
    params = params or {}

    def one(job: tuple[dict, int]) -> dict:
        row, k = job
        meta = row.get("metadata") or {}
        t0 = time.perf_counter()
        pol = call_policy(policy_url, chat_messages(row), policy_timeout_s, **params)
        wall_gen = round(time.perf_counter() - t0, 3)
        rec = {"uuid": row["uuid"], "task": meta.get("task"), "turn": meta.get("turn"), "k": k,
               "driver": "direct", "gen_wall_s": wall_gen}
        if "error" in pol:
            rec.update({"text": None, "reward": 0.0, "score": 0.0, "j_reward": None, "mask_sample": True,
                        "failure_kind": "executed_pivot:policy_error", "failure_reason": pol["error"]})
        else:
            t1 = time.perf_counter()
            res = verify({"uuid": row["uuid"], "expected_answer": str(row["expected_answer"])}, pol["text"])
            rec.update({"text": pol["text"], "usage": pol["usage"], "finish_reason": pol["finish_reason"],
                        "policy": pol["local_policy"], "reward": float(res["reward"]), "score": float(res["score"]),
                        "j_reward": res.get("j_reward"), "mask_sample": bool(res["mask_sample"]),
                        "failure_kind": failure_kind(res.get("failure_kind")),
                        "failure_reason": res.get("failure_kind"), "penalties": res.get("penalties"),
                        "verify_s": round(time.perf_counter() - t1, 3)})
        with lock, open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    jobs = [(row, k) for row in rows for k in range(n)]
    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        return list(ex.map(one, jobs))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="train.jsonl from resources_servers/executed_pivot/scripts/prepare.py")
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--store", required=True, help="the overlay store prepare.py built the anchors in")
    ap.add_argument("--policy-url", default=os.environ.get("LOCAL_POLICY_PROXY_URL", "http://127.0.0.1:8913/v1"))
    ap.add_argument("--n", type=int, default=2, help="rollouts per pivot")
    ap.add_argument("--pivots", help="only these pivots: task:turn,task:turn")
    ap.add_argument("--limit", type=int, help="only the first N selected pivots")
    ap.add_argument("--concurrency", type=int, default=2,
                    help="rollouts in flight (2 lets one verify overlap the next generation)")
    ap.add_argument("--policy-timeout", type=float, default=900, help="seconds per policy call (the proxy retries)")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--cmd-timeout", type=int, default=30)
    ap.add_argument("--wait-policy", type=float, default=60, help="seconds to wait for the policy endpoint")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from cleave.world.overlay import OverlayWorld
    from pivots.gym.core import Config, ExecutedPivotCore

    rows = select_rows([json.loads(x) for x in open(a.rows, encoding="utf-8") if x.strip()], a.pivots, a.limit)
    if not rows:
        print("no pivots selected", file=sys.stderr)
        return 2
    if not wait_for_policy(a.policy_url, a.wait_policy):
        print(f"policy endpoint {a.policy_url} did not answer /models within {a.wait_policy:g}s", file=sys.stderr)
        return 3
    core = ExecutedPivotCore.from_files(OverlayWorld(store=a.store, workdir=None), a.anchors,
                                        Config(threshold=a.threshold, cmd_timeout=a.cmd_timeout))
    t0 = time.time()
    recs = collect(rows, a.n, a.policy_url, core.verify, a.out, a.concurrency, policy_timeout_s=a.policy_timeout)
    masked = sum(r["mask_sample"] for r in recs)
    print(f"{len(recs)} rollouts over {len(rows)} pivots in {time.time() - t0:.0f}s: reward {sum(r['reward'] for r in recs):g}, "
          f"masked {masked} ({sum(r['failure_kind'] == 'executed_pivot:policy_error' for r in recs)} policy errors) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
