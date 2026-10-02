"""One executed_pivot rollout through NeMo Gym's own pieces, with the replaying stand-in as the policy.

A dataset row (billing-invoice-bugfix turn 3) goes to Gym's openai_model server class, which calls the
stand-in over HTTP at /v1/responses exactly as it would call a real endpoint; the NeMoGymResponse it returns
goes to the executed_pivot server's /verify, and the verify responses go to /aggregate_metrics. Each stored
action has a known label (J = string reward, X = executed reward), and Gym must report exactly those.

The full Ray-served path (`gym eval run`) is in resources_servers/executed_pivot/README.md, "Running a
rollout". NeMo Gym needs Python >= 3.13, so under the system interpreter this module skips:

    /tmp/claude-0/gymenv/bin/python -m pytest tests/test_gym_rollout_standin.py -q
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
openai_model_app = pytest.importorskip("responses_api_models.openai_model.app",
                                       reason="Gym's openai_model server is not importable")

from fastapi.testclient import TestClient  # noqa: E402

from cleave.world.overlay import available  # noqa: E402
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming  # noqa: E402
from nemo_gym.server_utils import ServerClient  # noqa: E402
from resources_servers.executed_pivot.app import (  # noqa: E402
    ExecutedPivotResourcesServer,
    ExecutedPivotResourcesServerConfig,
)
from resources_servers.executed_pivot.scripts.prepare import pivot_rows  # noqa: E402
from resources_servers.executed_pivot.scripts.standin_policy import (  # noqa: E402
    ReplayPolicy,
    load_actions,
    load_instructions,
    make_server,
)

pytestmark = pytest.mark.skipif(not available(), reason="needs user namespaces + overlayfs")

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK, TURN = "billing-invoice-bugfix", 3


def _action(*cmds: str) -> str:
    return json.dumps({"analysis": "a", "plan": "p", "task_complete": False,
                       "commands": [{"keystrokes": c, "duration": 0.1} for c in cmds]})


TAIL = ("grep -n BULK_THRESHOLD billing/invoice.py\n", "python3 -m unittest discover -s tests 2>&1 | tail -n 5\n")
# action id -> (text or None for the expert's own, J, X), labels as the audit assigns them
ACTIONS = {
    "ctrl:expert": (None, 1, 1),
    "ctrl:paraphrase": (_action("python3 -c \"import pathlib; p = pathlib.Path('billing/invoice.py'); "
                                "p.write_text(p.read_text().replace('qty > BULK_THRESHOLD', 'qty >= BULK_THRESHOLD'))\"\n",
                                *TAIL), 0, 1),
    "ctrl:flipped": (_action("sed -i 's/if qty > BULK_THRESHOLD:/if qty < BULK_THRESHOLD:/' billing/invoice.py\n",
                             *TAIL), 1, 0),
}


@pytest.fixture(scope="module")
def setup(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("gym_rollout")
    store = f"/tmp/cleave-gym-rollout-test-{os.getpid()}"
    rows, anchors = pivot_rows(REPO_ROOT / "specimens" / TASK, store, turns=[TURN])
    row = rows[0]
    (tmp / "anchors.jsonl").write_text("".join(json.dumps(a) + "\n" for a in anchors))
    (tmp / "actions.jsonl").write_text("".join(
        json.dumps({"task": TASK, "turn": TURN, "action": aid, "text": text or row["expected_answer"], "J": j, "X": x}) + "\n"
        for aid, (text, j, x) in ACTIONS.items()))
    policy = ReplayPolicy(load_actions(tmp / "actions.jsonl"), load_instructions(REPO_ROOT / "specimens"), pick="cycle")
    httpd = make_server(policy, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    model = openai_model_app.SimpleModelServer(
        config=openai_model_app.SimpleModelServerConfig(
            host="127.0.0.1", port=0, entrypoint="", name="policy_model",
            openai_base_url=f"http://127.0.0.1:{httpd.server_port}/v1", openai_api_key="dummy", openai_model="standin"),
        server_client=MagicMock(spec=ServerClient))
    resources = ExecutedPivotResourcesServer(
        config=ExecutedPivotResourcesServerConfig(
            host="127.0.0.1", port=0, entrypoint="", name="executed_pivot", backend="overlay", overlay_store=store,
            anchors_path=str(tmp / "anchors.jsonl"), max_concurrency=2),
        server_client=MagicMock(spec=ServerClient))
    yield {"row": row, "model": model, "client": TestClient(resources.setup_webserver())}
    httpd.shutdown()
    shutil.rmtree(store, ignore_errors=True)


def test_standin_recognises_the_pivot_from_the_prompt(setup) -> None:
    from resources_servers.executed_pivot.scripts.standin_policy import messages_of

    policy = ReplayPolicy({}, load_instructions(REPO_ROOT / "specimens"))
    assert policy.pivot_of(messages_of(setup["row"]["responses_create_params"])) == f"{TASK}:{TURN}"
    assert policy.choose("unknown:0")[0] == "unmatched"


async def _policy_calls(model, body, n: int) -> list:
    """n calls through the model server, on the aiohttp session Gym's servers share (made in this loop)."""
    from nemo_gym import server_utils

    session = server_utils.set_global_aiohttp_client(server_utils.GlobalAIOHTTPAsyncClientConfig())
    try:
        return [await model.responses(body) for _ in range(n)]
    finally:
        await session.close()
        server_utils._GLOBAL_AIOHTTP_CLIENT = None


def test_rollouts_report_the_known_labels(setup) -> None:
    row, client = setup["row"], setup["client"]
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(row["responses_create_params"])
    responses = asyncio.run(_policy_calls(setup["model"], body, len(ACTIONS)))  # pick=cycle: each action once
    verified = []
    for i, response in enumerate(responses):
        aid = response.metadata["standin_action"]
        r = client.post("/verify", json={**row, "response": response.model_dump(mode="json")})
        assert r.status_code == 200, r.text
        out = r.json()
        _, want_j, want_x = ACTIONS[aid]
        assert (out["j_reward"], out["reward"]) == (want_j, want_x), (aid, out)
        assert out["mask_sample"] is False and out["verify_s"] > 0
        verified.append({**out, "_ng_task_index": 0, "_ng_rollout_index": i})
    assert {v["response"]["metadata"]["standin_action"] for v in verified} == set(ACTIONS)

    agg = client.post("/aggregate_metrics", json={"verify_responses": verified})
    assert agg.status_code == 200, agg.text
    metrics = agg.json()["agent_metrics"]
    assert metrics["mean/reward"] == pytest.approx(2 / 3) and metrics["mean/j_reward"] == pytest.approx(2 / 3)
