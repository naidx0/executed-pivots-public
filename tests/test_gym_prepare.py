"""executed_pivot's prepare script leaves the committed example rows alone when the pivots are the same."""
import json
from pathlib import Path

import pytest

from resources_servers.executed_pivot.scripts.prepare import write_example


def _rows(*uuids, note=""):
    return [{"uuid": u, "responses_create_params": {"input": [{"role": "user", "content": note}]}} for u in uuids]


def test_same_pivots_keep_the_committed_file(tmp_path: Path):
    p = tmp_path / "example.jsonl"
    assert write_example(p, _rows("t:0", "t:1", note="Ran 7 tests in 0.001s"))
    before = p.read_bytes()
    assert not write_example(p, _rows("t:0", "t:1", note="Ran 7 tests in 0.002s"))
    assert p.read_bytes() == before


def test_different_pivots_are_written(tmp_path: Path):
    p = tmp_path / "example.jsonl"
    write_example(p, _rows("t:0"))
    assert write_example(p, _rows("t:0", "t:1"))
    assert [json.loads(line)["uuid"] for line in p.read_text().splitlines()] == ["t:0", "t:1"]


# --- the specimen sets: the original six plus the H17/H24 and H41 Dockerfile specimens ------------------------------
from resources_servers.executed_pivot.scripts import prepare as P  # noqa: E402

SPECIMENS = Path(__file__).resolve().parents[1] / "specimens"


def _names(pred):
    return {p.parent.name for p in SPECIMENS.glob("*/task.json") if pred(p.parent)}


def test_sets_are_derived_from_the_specimen_dirs():
    everything = _names(lambda d: True)
    tb = _names(lambda d: d.name.startswith("tb-"))
    assert set(P.TB_TWELVE) == tb and len(P.TB_TWELVE) == 12
    everything -= tb
    with_dockerfile = _names(lambda d: (d / "Dockerfile").exists()) - tb
    assert set(P.DOCKERFILE_FIVE) | set(P.H41_FIVE) == with_dockerfile and len(P.DOCKERFILE_FIVE) == 5
    assert len(P.H41_FIVE) == 5 and not set(P.H41_FIVE) & set(P.DOCKERFILE_FIVE)
    assert set(P.ORIGINAL_SIX) == everything - with_dockerfile and len(P.ORIGINAL_SIX) == 6
    assert set(P.RUNNER) == everything | tb and [p.name for p in P.specimen_set("all")] == list(P.SETS["all"])


def test_node_and_go_specimens_name_the_h24_runner():
    for name in P.DOCKERFILE_FIVE + P.H41_FIVE:
        dockerfile = (SPECIMENS / name / "Dockerfile").read_text()
        needs_h24 = "nodejs" in dockerfile or "golang" in dockerfile
        assert (P.RUNNER[name][0] == "h24-runner:local") == needs_h24, name
    assert {n for n, (img, _) in P.RUNNER.items() if img == "h24-runner:local"} == {"csvsum-npm-package", "tally-go-build"}


def test_a_missing_toolchain_stops_the_build_naming_the_runner(monkeypatch, tmp_path: Path):
    assert P.missing_tools("tally-go-build", which=lambda t: f"/usr/bin/{t}") == []  # positive control
    assert P.missing_tools("tally-go-build", which=lambda t: None if t == "go" else "/x") == ["go"]
    monkeypatch.setattr(P.shutil, "which", lambda t: None if t in ("node", "npm") else "/x")
    with pytest.raises(SystemExit, match=r"csvsum-npm-package needs node, npm .*h24-runner:local"):
        P.pivot_rows(SPECIMENS / "csvsum-npm-package", str(tmp_path / "store"))
