"""The served demo runs steps in the hardened runner unless XP_DEMO_HARDENED=0 (H18 kept after H19)."""
import pytest

pytest.importorskip("fastapi")
import pivots.demo.app as app  # noqa: E402


def _captured(monkeypatch, env):
    seen = {}

    class FakeCore:
        def __init__(self, store, **kw):
            seen.update(kw)

    monkeypatch.setattr(app, "DemoCore", FakeCore)
    monkeypatch.setattr(app, "create_app", lambda core, warm: core)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    app.app_from_env()
    return seen


def test_hardened_is_the_default(monkeypatch):
    monkeypatch.delenv("XP_DEMO_HARDENED", raising=False)
    assert _captured(monkeypatch, {})["hardened"] is True


def test_opt_out(monkeypatch):
    assert _captured(monkeypatch, {"XP_DEMO_HARDENED": "0"})["hardened"] is False
