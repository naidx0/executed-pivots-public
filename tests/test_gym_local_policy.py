"""configs/local_policy.yaml under NeMo Gym: executed_pivot rollouts whose policy is a local model.

Gym's inference_provider server, built from the config file itself, calls the local-policy proxy
(pivots.students.local_policy), which forwards to a scripted fake of llama-server / Ollama. Each reply goes to
the executed_pivot server's /verify. Checks: the config's fields are what Gym's server class accepts, a
thinking model's reasoning-only answer is lifted into the action and carried as a reasoning item, a wrapped
(think + fenced) expert action earns reward 1, and malformed JSON is paid 0 as model_output_invalid without
being masked. NeMo Gym needs Python >= 3.13 and the overlay backend needs Linux, so elsewhere this skips;
it runs in gym/gym-runner.Dockerfile (see gym/README.md):

    /opt/gym/bin/python -m pytest tests/test_gym_local_policy.py -q
"""
import asyncio
import json
import os
import shutil
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("nemo_gym", reason="NeMo Gym is not installed (needs Python >= 3.13)")
ip_app = pytest.importorskip("responses_api_models.inference_provider.app",
                             reason="Gym's inference_provider server is not importable")

from fastapi.testclient import TestClient  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from _fake_openai import FakeOpenAI  # noqa: E402
from cleave.world.overlay import available  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402
from pivots.students import local_policy as lp  # noqa: E402
from resources_servers.executed_pivot.app import (  # noqa: E402
    ExecutedPivotResourcesServer,
    ExecutedPivotResourcesServerConfig,
)
from resources_servers.executed_pivot.scripts.prepare import pivot_rows  # noqa: E402

pytestmark = pytest.mark.skipif(not available(), reason="needs user namespaces + overlayfs")

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "resources_servers/executed_pivot/configs/local_policy.yaml"
TASK, TURN = "billing-invoice-bugfix", 3


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("gym_local")
    store = f"/tmp/cleave-gym-local-test-{os.getpid()}"
    rows, anchors = pivot_rows(REPO_ROOT / "specimens" / TASK, store, turns=[TURN])
    row = rows[0]
    (tmp / "anchors.jsonl").write_text("".join(json.dumps(a) + "\n" for a in anchors))
    expert = row["expected_answer"]
    fake = FakeOpenAI([("reasoning", f"The fix is the threshold comparison.\n{expert}"), ("think", expert),
                       ("content", expert[:-10])]).start()
    policy = lp.LocalPolicy.from_env({lp.ENV_BASE_URL: fake.url, lp.ENV_MODEL: "qwen3:4b"}, sleep=lambda s: None)
    proxy = lp.make_proxy(policy, port=0, log_path=str(tmp / "proxy.jsonl"))
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    os.environ["LOCAL_POLICY_PROXY_URL"] = f"http://127.0.0.1:{proxy.server_port}/v1"
    try:
        cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    finally:
        os.environ.pop("LOCAL_POLICY_PROXY_URL")
    fields = cfg["policy_model"]["responses_api_models"]["inference_provider"]
    model = ip_app.InferenceProvider(
        config=ip_app.InferenceProviderConfig(host="127.0.0.1", port=0, name="policy_model", **fields),
        server_client=MagicMock(spec=ServerClient))
    resources = ExecutedPivotResourcesServer(
        config=ExecutedPivotResourcesServerConfig(
            host="127.0.0.1", port=0, entrypoint="", name="executed_pivot", backend="overlay", overlay_store=store,
            anchors_path=str(tmp / "anchors.jsonl"), max_concurrency=2),
        server_client=MagicMock(spec=ServerClient))
    yield {"row": row, "model": model, "client": TestClient(resources.setup_webserver()), "fake": fake,
           "fields": fields, "tmp": tmp}
    proxy.shutdown()
    fake.stop()
    shutil.rmtree(store, ignore_errors=True)


async def _policy_calls(model, body, n: int) -> list:
    from nemo_gym import server_utils

    session = server_utils.set_global_aiohttp_client(server_utils.GlobalAIOHTTPAsyncClientConfig())
    try:
        return [await model.responses(MagicMock(), body) for _ in range(n)]
    finally:
        await session.close()
        server_utils._GLOBAL_AIOHTTP_CLIENT = None


def test_config_points_at_the_proxy_with_no_key(setup) -> None:
    f = setup["fields"]
    assert f["base_url"].startswith("http://127.0.0.1:") and f["uses_reasoning_parser"] is True
    assert f["api_key"] == "not-needed-local-proxy"


def test_local_policy_rollouts_through_verify(setup) -> None:
    row = setup["row"]
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    responses = asyncio.run(_policy_calls(setup["model"], body, 3))
    assert [c["body"]["model"] for c in setup["fake"].calls] == ["qwen3:4b"] * 3
    outs = []
    for response in responses:
        r = setup["client"].post("/verify", json={**row, "response": response.model_dump(mode="json")})
        assert r.status_code == 200, r.text
        outs.append(r.json())
    reasoning_only, wrapped, malformed = outs
    types = [o["type"] for o in reasoning_only["response"]["output"]]
    assert "reasoning" in types  # the thinking became a reasoning item, not part of the action
    for out in (reasoning_only, wrapped):  # the expert's own action, however the model wrapped it
        assert out["mask_sample"] is False and out["reward"] == 1.0 and out["j_reward"] == 1.0, out["failure_reason"]
    assert malformed["reward"] == 0.0 and malformed["mask_sample"] is False
    assert malformed["failure_kind"] == "executed_pivot:model_output_invalid"
    log = [json.loads(x) for x in (setup["tmp"] / "proxy.jsonl").read_text().splitlines()]
    assert [x["source"] for x in log] == ["reasoning", "content", "invalid"]
