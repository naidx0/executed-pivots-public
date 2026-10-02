"""The demo opens only the 17 decisive pivots unless XP_DEMO_PIVOTS=all (or --pivots all) opts in to every turn."""
import pytest

from pivots.demo.core import BadRequest, DemoCore


def _core(tmp_path, **kw):
    return DemoCore(str(tmp_path / "store"), **kw)


def test_default_scope_is_the_decisive_pivots(tmp_path):
    core = _core(tmp_path)
    ids = [p["id"] for p in core.pivot_list()]
    assert ids == core.decisive and len(ids) == 17
    with pytest.raises(BadRequest):
        core._split("sensor-site-report:5")  # a real turn, but not decisive


def test_all_scope_opens_every_turn_of_every_specimen(tmp_path):
    core = _core(tmp_path, pivots="all")
    ids = [p["id"] for p in core.pivot_list()]
    want = [f"{name}:{t}" for name, st in core.specs.items() for t in range(len(st.sp.actions))]
    assert ids == want and len(set(ids)) == len(ids)
    assert set(core.decisive) < set(ids)
    st, turn = core._split("sensor-site-report:5")
    assert (st.sp.name, turn) == ("sensor-site-report", 5)
    with pytest.raises(BadRequest):
        core._split(f"sensor-site-report:{len(core.specs['sensor-site-report'].sp.actions)}")


def test_unknown_scope_is_refused(tmp_path):
    with pytest.raises(ValueError):
        _core(tmp_path, pivots="some")


def test_env_selects_the_scope(monkeypatch):
    pytest.importorskip("fastapi")
    import pivots.demo.app as app

    seen = {}

    class FakeCore:
        def __init__(self, store, **kw):
            seen.update(kw)

    monkeypatch.setattr(app, "DemoCore", FakeCore)
    monkeypatch.setattr(app, "create_app", lambda core, warm: core)
    monkeypatch.delenv("XP_DEMO_PIVOTS", raising=False)
    app.app_from_env()
    assert seen["pivots"] == "decisive"
    monkeypatch.setenv("XP_DEMO_PIVOTS", "all")
    app.app_from_env()
    assert seen["pivots"] == "all"
