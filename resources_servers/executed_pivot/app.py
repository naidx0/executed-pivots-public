"""NeMo Gym resources server: executed_pivot (drop-in for terminus_judge_string_only).

Same request shape as resources_servers/terminus_judge (responses_create_params,
expected_answer, uuid, metadata). The reward comes from executing the policy's
batch in a fork of the pivot's anchor instead of comparing keystroke strings.
Environment failures set mask_sample=True so GRPO drops the sample instead of
learning from an infrastructure error.

The scoring logic lives in pivots/gym/core.py (framework-free, tested without
Gym); this module is the Gym contract around it. `pivots` and `cleave` must be
importable: install the repo into the server venv (requirements.txt does) or run
Gym with the repo root as a search root (`--search-dir` / NEMO_GYM_EXTRA_ROOTS).
"""

import asyncio
import os
import re
import time
from typing import Any, ClassVar, Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from pivots.gym.core import Config, ExecutedPivotCore


class ExecutedPivotResourcesServerConfig(BaseResourcesServerConfig):
    # Verify forks immutable checkpoints and keeps no per-rollout state.
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    backend: str = "contree"  # "contree" (Nebius Sandboxes) or "overlay" (local, no Docker)
    anchors_path: str = "data/anchors.jsonl"  # relative to this server's directory
    # H44: verified earlier actions per pivot; unset -> references.jsonl next to the anchors, when present
    references_path: Optional[str] = None
    overlay_store: Optional[str] = None  # overlay only; must live under /tmp
    threshold: float = 0.75
    min_self_agreement: float = 1.0
    cmd_timeout: int = 30
    max_concurrency: int = 48  # Sandboxes beta: 50 concurrent operations per token


class ExecutedPivotRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    expected_answer: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class ExecutedPivotVerifyRequest(ExecutedPivotRunRequest, BaseVerifyRequest):
    pass


class ExecutedPivotVerifyResponse(BaseVerifyResponse):
    uuid: Optional[str | int] = None
    expected_answer: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    score: float = 0.0  # continuous effect score; reward is score >= threshold
    j_reward: Optional[float] = None  # what the string reward (terminus_judge_string_only) would have paid
    penalties: Optional[dict[str, float]] = None
    verify_s: Optional[float] = None  # wall time of the executed comparison (Gym reports mean/verify_s)


def _last_assistant_text(body: BaseVerifyRequest) -> str:
    """Concatenated assistant text, same as terminus_judge's _extract_last_assistant_text."""
    texts: list[str] = []
    for o in body.response.output:
        if getattr(o, "type", None) == "message" and getattr(o, "role", None) == "assistant":
            content = getattr(o, "content", None)
            if isinstance(content, list):
                texts += [c.text for c in content if isinstance(getattr(c, "text", None), str)]
            elif isinstance(content, str):
                texts.append(content)
    text = "\n".join(texts).strip()
    return text.split("</think>")[-1].strip() if "</think>" in text else text


def _failure_kind(detail: Optional[str]) -> Optional[str]:
    """Core failure label -> Gym's low-cardinality '<server>:<kind>' name (detail goes to failure_reason)."""
    if not detail:
        return None
    kind = re.sub(r"[^a-z0-9_]+", "_", detail.split(":", 1)[0].strip().lower()).strip("_")
    return f"executed_pivot:{kind or 'unknown'}"


def _make_world(config: ExecutedPivotResourcesServerConfig):
    if config.backend == "overlay":
        from cleave.world.overlay import OverlayWorld

        return OverlayWorld(store=config.overlay_store, workdir=None)
    if config.backend == "contree":
        from cleave.world.contree import ContreeWorld

        return ContreeWorld(_fresh=False)
    raise ValueError(f"executed_pivot: unknown backend {config.backend!r}; use 'contree' or 'overlay'")


class ExecutedPivotResourcesServer(SimpleResourcesServer):
    config: ExecutedPivotResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if not os.path.exists(self.config.anchors_path):
            raise FileNotFoundError(
                f"executed_pivot: anchors file {os.path.abspath(self.config.anchors_path)} not found; "
                "build it with scripts/prepare.py (overlay) or point anchors_path at the rebuilt pivots' anchors"
            )
        core_config = Config(
            threshold=self.config.threshold,
            min_self_agreement=self.config.min_self_agreement,
            cmd_timeout=self.config.cmd_timeout,
            references_path=self.config.references_path,
        )
        self._core = ExecutedPivotCore.from_files(_make_world(self.config), self.config.anchors_path, core_config)
        self._sem = asyncio.Semaphore(self.config.max_concurrency)

    async def verify(self, body: ExecutedPivotVerifyRequest) -> ExecutedPivotVerifyResponse:
        text = _last_assistant_text(body)
        expected = body.expected_answer or (body.metadata or {}).get("expected_answer") or ""
        row = {"uuid": body.uuid, "expected_answer": str(expected)}
        async with self._sem:
            t0 = time.perf_counter()
            res = await asyncio.to_thread(self._core.verify, row, text)
            elapsed = time.perf_counter() - t0
        detail = res.get("failure_kind")
        return ExecutedPivotVerifyResponse(
            **body.model_dump(exclude={"uuid", "expected_answer", "metadata"}),
            uuid=body.uuid,
            expected_answer=row["expected_answer"],
            metadata=body.metadata,
            reward=float(res["reward"]),
            score=float(res["score"]),
            j_reward=res.get("j_reward"),
            penalties=res.get("penalties"),
            verify_s=round(elapsed, 3),
            mask_sample=bool(res["mask_sample"]),
            failure_kind=_failure_kind(detail),
            failure_reason=detail,
        )


if __name__ == "__main__":
    ExecutedPivotResourcesServer.run_webserver()
