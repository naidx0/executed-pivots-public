"""Pivot Inspector build: a standalone UTF-8 page with the P label joined in (no browser needed)."""
import json
import re
from pathlib import Path

from pivots.report.inspector import TEMPLATE, build

ROW = {"task": "t", "turn": 0, "turns": 1, "action": "ctrl:flipped", "J": 1, "J_sim": 0.99, "X": 0, "X_score": 0.04,
       "E": 0, "text": "{}", "self_agreement": 1.0}


def _audit(tmp_path: Path, progress: bool = True) -> Path:
    d = tmp_path / "audit"
    d.mkdir()
    (d / "rows.jsonl").write_text(json.dumps(ROW) + "\n")
    (d / "pivots.json").write_text(json.dumps([{"task": "t", "turn": 0, "gold": "{}", "before": "$ ", "instruction": "fix ✓"}]))
    (d / "summary.json").write_text(json.dumps({"tasks": ["t"], "controls": {}, "ctrl": {}}))
    if progress:
        (d / "progress.jsonl").write_text(json.dumps({"task": "t", "turn": 0, "action": "ctrl:flipped", "P": 0,
                                                      "checks_before": 6, "checks_expert": 9, "checks_after": 6}) + "\n")
    return d


def _data(html: str) -> dict:
    return json.loads(re.search(r"const DATA = (.*?);\nconst \$", html, re.S).group(1).replace("<\\/", "</"))


def test_page_is_a_standalone_utf8_document(tmp_path: Path):
    out = tmp_path / "inspector.html"
    build(str(_audit(tmp_path)), str(out), "a note")
    raw = out.read_bytes()
    html = raw.decode("utf-8")
    assert html.startswith("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">")
    assert '<meta name="viewport"' in html
    head, body = html.split("</head>")
    assert "<style>" in head and "</style>" in head and 'id="pane"' in body and "<script>" in body
    assert "<title>Executed Pivots · Pivot Inspector</title>" in head and "<h1>Executed Pivots</h1>" in body
    assert html.rstrip().endswith("</html>") and html.count("<body>") == 1
    assert "✓".encode() in raw  # the template's own check marks, written as UTF-8
    assert _data(html)["note"] == "a note"


def test_progress_label_and_check_counts_are_joined(tmp_path: Path):
    out = tmp_path / "inspector.html"
    build(str(_audit(tmp_path)), str(out))
    row = _data(out.read_text(encoding="utf-8"))["rows"][0]
    assert row["P"] == 0 and row["checks"] == [6, 9, 6] and "self_agreement" not in row
    out2 = tmp_path / "no-progress.html"
    (tmp_path / "audit" / "progress.jsonl").unlink()
    build(str(tmp_path / "audit"), str(out2))
    assert "P" not in _data(out2.read_text(encoding="utf-8"))["rows"][0]


def test_filter_buttons_do_not_reset_the_scope_buttons():
    js = TEMPLATE.read_text(encoding="utf-8")
    assert 'querySelectorAll(".fbtn")' not in js  # both button rows use .fbtn; filters must touch only #filters
