"""pivots.audit.run's terminal summary and stale-label cleanup (no worlds needed)."""
import json
from pathlib import Path

from pivots.audit.run import clear_stale, report_lines, summarize


def _row(task, turn, action, J, X, E):
    return {"task": task, "turn": turn, "action": action, "J": J, "X": X, "E": E, "gold_passes": True}


ROWS = [
    _row("t", 0, "ctrl:expert", 1, 1, 1),
    _row("t", 0, "ctrl:flipped", 1, 0, 0),
    _row("t", 0, "ctrl:no_op", 0, 0, 0),
    _row("t", 0, "student:opus:0", 0, 1, 1),
    _row("t", 0, "student:opus:1", 0, 0, 0),
]


def test_report_is_a_short_table_not_the_summary_json():
    summ = summarize(ROWS)
    summ["seconds"] = 1.5
    lines = report_lines(summ, "out/audit")
    text = "\n".join(lines)
    assert len(lines) < 15 and "{" not in text
    assert "flipped" in text and text.index("expert") < text.index("flipped") < text.index("no_op")
    assert "students opus: 1 of 2 work (E=1); credited J 0, X 1; failing ones credited J 0, X 0" in text
    assert "decisive pivots 1; gold passes on 1 of 1 tasks" in text
    assert lines[-1] == "wrote out/audit/rows.jsonl, pivots.json, summary.json (1.5 s)"


def test_progress_labels_from_a_previous_run_are_removed(tmp_path: Path):
    (tmp_path / "progress.jsonl").write_text(json.dumps({"task": "t"}) + "\n")
    (tmp_path / "progress.summary.json").write_text("{}")
    (tmp_path / "rows.jsonl").write_text("")
    assert clear_stale(str(tmp_path)) == ["progress.jsonl", "progress.summary.json"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rows.jsonl"]
    assert clear_stale(str(tmp_path)) == []
    summ = summarize(ROWS)
    assert "rerun pivots.audit.progress" in report_lines(summ, str(tmp_path), ["progress.jsonl"])[-1]
