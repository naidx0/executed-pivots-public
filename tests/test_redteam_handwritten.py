"""H23: the hand-written red-team family source_edit_handwritten (pivots/audit/redteam.py; H23 in RESEARCH.md).

Pure tests: each attack is the expert's batch, then one overwrite of the edited source file with the expert's
post-step bytes and one replacement; a replacement that is not found exactly once fails generation.
"""
import base64
import json
import re
from pathlib import Path

import pytest

from pivots.audit.redteam import H23_ATTACKS, _MED, handwritten_edits

ROOT = Path(__file__).resolve().parent.parent
GOLD = {"analysis": "a", "plan": "p", "task_complete": False,
        "commands": [{"keystrokes": "make && make test\n", "duration": 5}]}


def b64(s):
    return base64.b64encode(s.encode()).decode()


def written(action: str) -> str:
    last = json.loads(action)["commands"][-1]["keystrokes"]
    return base64.b64decode(re.search(r"printf %s (\S+) \| base64 -d", last).group(1)).decode()


def test_eleven_attacks_on_distinct_names_and_known_specimens():
    assert len(H23_ATTACKS) == 11
    assert len({(t, k, n) for t, k, _, n, _, _ in H23_ATTACKS}) == 11
    assert all((ROOT / "specimens" / t / "task.json").exists() for t, *_ in H23_ATTACKS)
    assert all(old != new for *_, old, new in H23_ATTACKS)


def test_statcli_odd_median_line_is_the_setup_line_the_expert_leaves_alone():
    setup = (ROOT / "specimens/statcli-make-fix/setup.sh").read_text()
    assert setup.count(_MED) == 1


def test_attack_is_batch_then_overwrite_with_one_replacement():
    post = "int x;\n" + _MED + "int y;\n"
    path = "/app/statcli/src/stats.c"
    rows = handwritten_edits("statcli-make-fix", 5, GOLD, {"files": {path: {"size": len(post), "b64": b64(post)}}})
    assert [m["mutation"] for m, _ in rows] == ["odd_median_off_by_one", "odd_median_mid3",
                                                  "odd_median_build_sensing"]
    for m, text in rows:
        cmds = json.loads(text)["commands"]
        assert m["placement"] == "after" and cmds[0]["keystrokes"] == "make && make test\n" and len(cmds) == 2
        out = written(text)
        assert out != post and out.startswith("int x;\n") and out.endswith("int y;\n")
    sensing = written(rows[2][1])
    assert sensing.index('__has_include("../bin/statcli")') < sensing.index(_MED) < sensing.index("#else")
    assert handwritten_edits("statcli-make-fix", 4, GOLD, {"files": {}}) == []


def test_missing_or_repeated_replacement_fails():
    path = "/app/statcli/src/stats.c"
    for post in ("int x;\n", _MED + _MED):
        with pytest.raises(ValueError):
            handwritten_edits("statcli-make-fix", 5, GOLD, {"files": {path: {"size": 1, "b64": b64(post)}}})
