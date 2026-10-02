"""Sampling from the pre-built pivot prompts (the Windows-friendly half of the live audit), against the mock."""
import json

import pytest

from pivots.students import sample_prompts as spm
from pivots.students.mock_server import MockTokenFactory

DECISIVE = "access-log-summary:4,backup-cron-repair:3,billing-invoice-bugfix:3,ledger-git-revert:3,sensor-site-report:4,statcli-make-fix:5"


def test_prompts_cover_every_pivot_and_the_decisive_six():
    prompts = spm.load_prompts()
    assert len(prompts) == 45
    for key in spm.parse_pivots(DECISIVE):
        msgs = prompts[key]
        assert msgs[0]["role"] == "user" and "Task Description:" in msgs[0]["content"]
        assert len(msgs) == 1 + 2 * key[1]  # first prompt, then (assistant, terminal output) per earlier turn


def test_sample_writes_students_rows(tmp_path, monkeypatch):
    m = MockTokenFactory().start()
    try:
        monkeypatch.setenv("NEBIUS_API_KEY", "sk-test-not-a-key")
        out = tmp_path / "students.jsonl"
        rc = spm.main(["--pivots", "ledger-git-revert:3,billing-invoice-bugfix:3", "--n", "2", "--out", str(out),
                       "--base-url", m.url, "--cache-dir", str(tmp_path / "cache"), "--max-requests", "6"])
        assert rc == 0
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(rows) == 4 and {(r["task"], r["turn"]) for r in rows} == {("ledger-git-revert", 3), ("billing-invoice-bugfix", 3)}
        assert all(r["pool"].startswith("tf-") and r["text"] for r in rows)
        assert "sk-test-not-a-key" not in out.read_text()
    finally:
        m.stop()


def test_unknown_pivot_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        spm.main(["--pivots", "no-such-task:1", "--out", str(tmp_path / "s.jsonl")])
