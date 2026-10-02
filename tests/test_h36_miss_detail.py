"""H36 script: the row selection and the cost cap (no sandbox, no credentials)."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h36", REPO / "scripts" / "h36_miss_detail.py")
h36 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h36)


@pytest.mark.skipif(not h36.LABELS.exists(), reason="needs the H31 labels, which the public repo does not ship")
def test_selects_the_68_content_misses():
    rows = h36.select_rows()
    assert len(rows) == 68
    assert all(r["E"] == 1 and r["X"] == 0 and r["source"] == "content" for r in rows)
    assert len({r["i"] for r in rows}) == 68


def test_selection_rule_on_a_planted_file(tmp_path):
    rows = [{"i": 0, "E": 1, "X": 0, "source": "content"}, {"i": 1, "E": 1, "X": 0, "source": "reasoning"},
            {"i": 2, "E": 0, "X": 0, "source": "content"}, {"i": 3, "E": 1, "X": 1, "source": "content"}]
    p = tmp_path / "labels.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert [r["i"] for r in h36.select_rows(p)] == [0]


def test_cost_cap_stops_and_stays_stopped():
    g = h36.CostGuard(max_steps=10, max_ops=100, max_cost=0.5)
    g.before_op()
    g.after_op(cost=0.3)
    g.before_op()  # 0.3 < 0.5: allowed
    g.after_op(cost=0.3)
    with pytest.raises(h36.SpendAbort, match="max_cost"):
        g.before_op()
    with pytest.raises(h36.SpendAbort):
        g.before_op()
    assert g.ops == 2
