"""Live student sampling for the audit: `--student tokenfactory:<model>` -> N actions per pivot.

The sampler receives the exact Terminus-2 prompt at each pivot (pivots.specimen.prompt_for, the one the
Claude stand-in pools were given) and returns (pool, action text) pairs, which the audit labels J/X/E
like any other student. Every sample, with its reasoning, finish reason, token usage and cache flag, is
also appended to `<out>/students.jsonl` in the audit's --students format ({task, turn, pool, text}), so
a run can be relabelled later without calling the model again.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pivots.students.tokenfactory import (
    BASE_URL,
    DEFAULT_CACHE,
    DEFAULT_MODEL,
    TokenFactory,
    TokenFactoryError,
    TokenFactoryStop,
    pool_name,
    split_reasoning,
)

PROVIDERS = ("tokenfactory",)


def parse_spec(spec: str) -> tuple[str, str]:
    """'tokenfactory:<model>' (or bare 'tokenfactory') -> (provider, model)."""
    provider, _, model = spec.partition(":")
    if provider not in PROVIDERS:
        raise ValueError(f"--student {spec!r}: provider must be one of {PROVIDERS}, as in tokenfactory:<model id>")
    return provider, model or DEFAULT_MODEL


class LiveSampler:
    def __init__(self, spec: str, n: int, out_path: str | None = None, concurrency: int = 4, base_url: str = BASE_URL,
                 max_requests: int = 400, cache_dir: str | None = DEFAULT_CACHE, pool: str | None = None,
                 client: TokenFactory | None = None, **params):
        _, model = parse_spec(spec)
        self.client = client or TokenFactory(model, base_url, max_requests=max_requests, cache_dir=cache_dir)
        self.pool = pool or pool_name(self.client.model)
        if ":" in self.pool:
            raise ValueError("a pool name cannot contain ':'")
        self.n, self.concurrency, self.params = n, concurrency, params
        self.out_path = Path(out_path) if out_path else None
        self.errors: list[dict] = []
        self.samples = 0
        self._lock = threading.Lock()
        if self.out_path:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            self.out_path.write_text("")

    def expected_requests(self, n_pivots: int) -> int:
        return n_pivots * self.n

    def _one(self, task: str, turn: int, messages: list[dict], k: int) -> dict | None:
        try:
            out = self.client.chat(messages, sample=k, **self.params)
        except TokenFactoryStop:
            raise
        except TokenFactoryError as e:  # this sample only (e.g. a 400); the run goes on
            with self._lock:
                self.errors.append({"task": task, "turn": turn, "k": k, "error": str(e)})
            return None
        choice = out["choices"][0]
        text, reasoning = split_reasoning(choice.get("message") or {})
        row = {"task": task, "turn": turn, "pool": self.pool, "k": k, "text": text, "model": self.client.model,
               "reasoning": reasoning, "finish_reason": choice.get("finish_reason"), "usage": out.get("usage"),
               "cached": out["_cached"], "prompt_hash": out["_prompt_hash"]}
        with self._lock:
            self.samples += 1
            if self.out_path:  # written as it arrives, so a stop mid-run keeps what was paid for
                with open(self.out_path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
        return row

    def __call__(self, sp, pivots: list[tuple[int, list[dict]]]) -> dict[int, list[tuple[str, str]]]:
        jobs = [(sp.name, t, msgs, k) for t, msgs in pivots for k in range(self.n)]
        with ThreadPoolExecutor(self.concurrency) as ex:
            futs = [ex.submit(self._one, *j) for j in jobs]
            rows = []
            try:
                for f in futs:
                    rows.append(f.result())
            except TokenFactoryStop:
                for f in futs:
                    f.cancel()
                raise
        got: dict[int, list[tuple[str, str]]] = {t: [] for t, _ in pivots}
        for r in rows:
            if r is None:
                continue
            got[r["turn"]].append((self.pool, r["text"]))
        return got

    def stats(self) -> dict:
        return {**self.client.stats(), "pool": self.pool, "n_per_pivot": self.n, "samples": self.samples,
                "errors": self.errors}
