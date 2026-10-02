"""executed_pivot through NeMo Gym's own server stack, on a real specimen pivot.

The pivot is billing-invoice-bugfix turn 3: the expert fixes the bulk-discount
threshold with sed, greps the line and reruns the tests. The request goes through
the FastAPI app Gym serves (POST /verify), so the request/response models, the
failsafe wrapper and serialization are exercised, not only the verify coroutine.

Needs user namespaces + overlayfs (the overlay backend); skipped otherwise.
"""

import json
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from cleave.world.overlay import available
from nemo_gym.server_utils import ServerClient

from ..app import ExecutedPivotResourcesServer, ExecutedPivotResourcesServerConfig, _failure_kind
from ..scripts.prepare import pivot_rows


REPO_ROOT = Path(__file__).resolve().parents[3]
SPECIMEN = REPO_ROOT / "specimens" / "billing-invoice-bugfix"
TURN = 3

pytestmark = pytest.mark.skipif(not available(), reason="needs user namespaces + overlayfs")


def _action(*cmds: str) -> str:
    return json.dumps(
        {
            "analysis": "The bulk discount check uses > where the spec says 10 units or more.",
            "plan": "Fix the comparison, show the line, rerun the tests.",
            "commands": [{"keystrokes": c, "duration": 0.1} for c in cmds],
            "task_complete": False,
        }
    )


# Same effect as the expert's sed, spelled differently enough that the string reward pays 0.
PARAPHRASE = _action(
    "python3 -c \"import pathlib; p = pathlib.Path('billing/invoice.py'); "
    "p.write_text(p.read_text().replace('qty > BULK_THRESHOLD', 'qty >= BULK_THRESHOLD'))\"\n",
    "grep -n BULK_THRESHOLD billing/invoice.py\n",
    "python3 -m unittest discover -s tests 2>&1 | tail -n 5\n",
)
# One character away from the expert's batch, opposite effect: the string reward pays 1.
FLIPPED = _action(
    "sed -i 's/if qty > BULK_THRESHOLD:/if qty < BULK_THRESHOLD:/' billing/invoice.py\n",
    "grep -n 'BULK_THRESHOLD' billing/invoice.py\n",
    "python3 -m unittest discover -s tests 2>&1 | tail -n 5\n",
)


@pytest.fixture(scope="module")
def pivot(tmp_path_factory: pytest.TempPathFactory):
    store = f"/tmp/cleave-gym-test-{os.getpid()}"
    rows, anchors = pivot_rows(SPECIMEN, store, turns=[TURN])
    anchors_path = tmp_path_factory.mktemp("executed_pivot") / "anchors.jsonl"
    anchors_path.write_text("".join(json.dumps(a) + "\n" for a in anchors))
    yield {"row": rows[0], "store": store, "anchors_path": str(anchors_path)}
    shutil.rmtree(store, ignore_errors=True)


@pytest.fixture(scope="module")
def client(pivot) -> TestClient:
    config = ExecutedPivotResourcesServerConfig(
        host="127.0.0.1",
        port=20099,
        entrypoint="",
        name="executed_pivot_test_server",
        backend="overlay",
        overlay_store=pivot["store"],
        anchors_path=pivot["anchors_path"],
        max_concurrency=4,
    )
    server = ExecutedPivotResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return TestClient(server.setup_webserver())


def _request(row: dict, model_output: str, **overrides) -> dict:
    return {
        **row,
        "response": {
            "id": "resp_test",
            "created_at": 0,
            "model": "test_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": model_output, "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
        **overrides,
    }


def _verify(client: TestClient, body: dict) -> dict:
    r = client.post("/verify", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_row_is_a_terminus_judge_row(pivot) -> None:
    row = pivot["row"]
    assert row["uuid"] == f"billing-invoice-bugfix:{TURN}"
    assert row["metadata"]["harness"] == "terminus_2"
    msgs = row["responses_create_params"]["input"]
    assert msgs[0]["role"] == "user" and len(msgs) == 1 + 2 * TURN
    assert "if qty > BULK_THRESHOLD:" in json.loads(row["expected_answer"])["commands"][0]["keystrokes"]


def test_expert_action_is_rewarded(client, pivot) -> None:
    out = _verify(client, _request(pivot["row"], pivot["row"]["expected_answer"]))
    assert out["reward"] == 1.0 and out["j_reward"] == 1.0
    assert out["mask_sample"] is False and out["failure_kind"] is None


def test_effect_equivalent_paraphrase_gets_reward_1(client, pivot) -> None:
    out = _verify(client, _request(pivot["row"], PARAPHRASE))
    assert out["reward"] == 1.0, out
    assert out["j_reward"] == 0.0  # the string reward would have punished it
    assert out["mask_sample"] is False
    assert out["uuid"] == pivot["row"]["uuid"]


def test_flipped_action_gets_reward_0(client, pivot) -> None:
    out = _verify(client, _request(pivot["row"], FLIPPED))
    assert out["reward"] == 0.0 and out["score"] < 0.75, out
    assert out["j_reward"] == 1.0  # the string reward would have paid it
    assert out["mask_sample"] is False


def test_unknown_pivot_is_masked(client, pivot) -> None:
    out = _verify(client, _request(pivot["row"], pivot["row"]["expected_answer"], uuid="no-such-pivot:0"))
    assert out["mask_sample"] is True and out["reward"] == 0.0
    assert out["failure_kind"] == "executed_pivot:no_anchor"


def test_think_tags_are_stripped(client, pivot) -> None:
    out = _verify(client, _request(pivot["row"], "<think>fix the threshold</think>" + PARAPHRASE))
    assert out["reward"] == 1.0


def test_failure_kind_is_namespaced_and_low_cardinality() -> None:
    assert _failure_kind(None) is None
    assert _failure_kind("no_anchor") == "executed_pivot:no_anchor"
    assert _failure_kind("prepare_error: OverlayError: unknown checkpoint x") == "executed_pivot:prepare_error"
