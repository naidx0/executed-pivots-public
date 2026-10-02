"""configs/tokenfactory_policy.yaml under NeMo Gym: an executed_pivot rollout whose policy is a Token Factory model.

Gym's inference_provider server, built from the config file itself, calls the budget proxy
(pivots.students.tokenfactory --serve), which forwards to a mock OpenAI-compatible server standing in for
api.tokenfactory.nebius.com. The reply goes to the executed_pivot server's /verify. Checks: the config's
fields are what Gym's server class accepts, the proxy substitutes the real key for Gym's placeholder,
reasoning is carried separately from the action, and the verify response is a real (unmasked) score.
NeMo Gym needs Python >= 3.13, so under the system interpreter this module skips:

    /tmp/claude-0/gymenv/bin/python -m pytest tests/test_gym_tokenfactory_policy.py -q
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

from cleave.world.overlay import available  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402
from pivots.students import tokenfactory as tf  # noqa: E402
from pivots.students.mock_server import ACTIONS, MockTokenFactory  # noqa: E402
from resources_servers.executed_pivot.app import (  # noqa: E402
    ExecutedPivotResourcesServer,
    ExecutedPivotResourcesServerConfig,
)
from resources_servers.executed_pivot.scripts.prepare import pivot_rows  # noqa: E402

pytestmark = pytest.mark.skipif(not available(), reason="needs user namespaces + overlayfs")

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "resources_servers/executed_pivot/configs/tokenfactory_policy.yaml"
TASK, TURN = "billing-invoice-bugfix", 3
KEY = "sk-test-SENTINEL-gym"


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("gym_tf")
    store = f"/tmp/cleave-gym-tf-test-{os.getpid()}"
    rows, anchors = pivot_rows(REPO_ROOT / "specimens" / TASK, store, turns=[TURN])
    (tmp / "anchors.jsonl").write_text("".join(json.dumps(a) + "\n" for a in anchors))
    mock = MockTokenFactory().start()
    client = tf.TokenFactory(tf.DEFAULT_MODEL, mock.url, api_key=KEY, cache_dir=str(tmp / "cache"), max_requests=5)
    proxy = tf.make_proxy(client, port=0)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    os.environ["TOKENFACTORY_PROXY_URL"] = f"http://127.0.0.1:{proxy.server_port}/v1"
    try:
        cfg = OmegaConf.to_container(OmegaConf.load(CONFIG), resolve=True)
    finally:
        os.environ.pop("TOKENFACTORY_PROXY_URL")
    fields = cfg["policy_model"]["responses_api_models"]["inference_provider"]
    model = ip_app.InferenceProvider(
        config=ip_app.InferenceProviderConfig(host="127.0.0.1", port=0, name="policy_model", **fields),
        server_client=MagicMock(spec=ServerClient))
    resources = ExecutedPivotResourcesServer(
        config=ExecutedPivotResourcesServerConfig(
            host="127.0.0.1", port=0, entrypoint="", name="executed_pivot", backend="overlay", overlay_store=store,
            anchors_path=str(tmp / "anchors.jsonl"), max_concurrency=2),
        server_client=MagicMock(spec=ServerClient))
    yield {"row": rows[0], "model": model, "client": TestClient(resources.setup_webserver()), "mock": mock,
           "tf": client, "fields": fields, "tmp": tmp}
    proxy.shutdown()
    mock.stop()
    shutil.rmtree(store, ignore_errors=True)


async def _policy_calls(model, body, n: int) -> list:
    from nemo_gym import server_utils

    session = server_utils.set_global_aiohttp_client(server_utils.GlobalAIOHTTPAsyncClientConfig())
    try:
        return [await model.responses(MagicMock(), body) for _ in range(n)]
    finally:
        await session.close()
        server_utils._GLOBAL_AIOHTTP_CLIENT = None


def test_config_points_at_the_proxy_without_the_key(setup) -> None:
    f = setup["fields"]
    assert f["base_url"].startswith("http://127.0.0.1:") and f["model"] == tf.DEFAULT_MODEL
    assert KEY not in json.dumps(f) and f["uses_reasoning_parser"] is True


def test_rollout_through_token_factory_policy(setup) -> None:
    row, mock = setup["row"], setup["mock"]
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    responses = asyncio.run(_policy_calls(setup["model"], body, 2))
    assert [c["auth"] for c in mock.calls] == [f"Bearer {KEY}"] * 2  # Gym sent the placeholder; the proxy the key
    assert all(c["model"] == tf.DEFAULT_MODEL for c in mock.calls)
    seen = set()
    for response in responses:
        r = setup["client"].post("/verify", json={**row, "response": response.model_dump(mode="json")})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["mask_sample"] is False and out["j_reward"] == 0 and out["reward"] in (0, 1)
        types = [o["type"] for o in out["response"]["output"]]
        assert "reasoning" in types  # reasoning_content became a reasoning item, not part of the action
        text = [c["text"] for o in out["response"]["output"] if o["type"] == "message" for c in o["content"]]
        seen.add("".join(text).strip())
    assert seen == set(ACTIONS[:2])  # two distinct samples: the proxy numbers identical prompts
    assert setup["tf"].requests == 2
    for p in setup["tmp"].rglob("*"):
        if p.is_file():
            assert KEY not in p.read_text(errors="replace")
