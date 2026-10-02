"""scripts/h32_contree_rescore.py against a fake contree_sdk and a stub demo core (no key, no network).

The stub core stands in for pivots.demo.core.DemoCore: it builds a NebiusWorld through the world factory
the script hands it and runs sandbox operations for every step, so the credential check, the spend guard,
the per-step log and the comparison with the reference labels all run as they would for real.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import _fake_contree_sdk as fake

REPO = Path(__file__).resolve().parents[1]
KEY = "nbk-PLANTED-FAKE-KEY-2b81e0"
PROJECT = "project-e00planted0fake2"

REF = [  # (pivot, action, reference J X P)
    ("a:1", "ctrl:expert", [1, 1, 1]),
    ("a:1", "ctrl:no_op", [0, 0, 1]),
    ("a:1", "student:x:0", [0, 1, 1]),
    ("b:2", "ctrl:expert", [1, 1, 1]),
    ("b:2", "ctrl:flipped", [1, 0, 0]),
    ("b:2", "student:y:0", [0, 0, 1]),
    ("b:2", "student:y:1", [0, 1, 1]),
]


def load_script():
    spec = importlib.util.spec_from_file_location("h32_contree_rescore", REPO / "scripts" / "h32_contree_rescore.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubCore:
    """live labels = reference labels, except where `live` overrides; `billing` makes the platform refuse."""

    def __init__(self, factory, live=None, billing=()):
        self.factory = factory
        self.live = live or {}
        self.billing = set(billing)
        self.decisive = sorted({p for p, _, _ in REF})
        self.ran = []

    def candidates(self, key):
        return [{"id": a, "text": f"text:{a}"} for p, a, _ in REF if p == key]

    def run(self, key, text, with_p=True):
        aid = text.split(":", 1)[1]
        self.ran.append((key, aid))
        w = self.factory(workdir="/app")
        if (key, aid) in self.billing:
            w.run(["billing"])
        w.run(["set", "k", aid])
        j, x, p = self.live.get((key, aid), dict((((q, b), r) for q, b, r in REF))[(key, aid)])
        return {"J": {"reward": j}, "X": {"reward": x}, "P": {"P": p}}


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("NEBIUS_API_KEY", KEY)
    monkeypatch.setenv("NEBIUS_PROJECT_ID", PROJECT)
    monkeypatch.setitem(sys.modules, "contree_sdk", fake.module())
    ref = tmp_path / "ref.jsonl"
    ref.write_text("".join(json.dumps({"pivot": p, "action": a, "live_JXP": r, "stored_JXP": r, "match": True})
                           + "\n" for p, a, r in REF))
    return tmp_path, ref


def run_script(tmp, ref, monkeypatch, *extra, core_kw=None):
    mod = load_script()
    made = {}

    def build_core(a, factory):
        made["core"] = StubCore(factory, **(core_kw or {}))
        return made["core"]

    monkeypatch.setattr(mod, "build_core", build_core)
    out, log = tmp / "out" / "rescore.jsonl", tmp / "out" / "steps.jsonl"
    code = mod.main(["run", "--ref", str(ref), "--out", str(out), "--log", str(log), *extra])
    return code, out, log, made.get("core")


def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def all_written(tmp):
    return "".join(f.read_text(errors="replace") for f in tmp.rglob("*") if f.is_file() and f.name != "ref.jsonl")


def test_success_path_default_cap_is_five_steps(env, monkeypatch, capsys):
    tmp, ref = env
    code, out, log, core = run_script(tmp, ref, monkeypatch,
                                      core_kw={"live": {("a:1", "student:x:0"): (0, 0, 1)}})
    assert code == 0
    got = rows(out)
    assert [(r["pivot"], r["action"]) for r in got] == [(p, a) for p, a, _ in REF[:5]]
    assert len(core.ran) == 5
    assert sum(not r["match"] for r in got) == 1
    bad = next(r for r in got if not r["match"])
    assert bad["live_JXP"] == [0, 0, 1] and bad["ref_JXP"] == [0, 1, 1]
    steps = [r for r in rows(log) if "step" in r]
    assert [s["step"] for s in steps] == [1, 2, 3, 4, 5]
    assert all(s["status"] == "ok" and isinstance(s["wall_s"], float) and s["ops"] >= 2 for s in steps)
    summary = json.loads((tmp / "out" / "rescore.summary.json").read_text())
    assert summary["n"] == 5 and summary["identical"] == 4 and summary["x_false_credit"] == 0
    assert summary["aborted"] is None
    assert "5 steps" in capsys.readouterr().out


def test_x_false_credit_is_counted(env, monkeypatch):
    tmp, ref = env
    code, out, _, _ = run_script(tmp, ref, monkeypatch, "--max-steps", "7",
                                 core_kw={"live": {("b:2", "ctrl:flipped"): (1, 1, 0)}})
    assert code == 0 and len(rows(out)) == 7
    summary = json.loads((tmp / "out" / "rescore.summary.json").read_text())
    assert summary["x_false_credit"] == 1 and summary["identical"] == 6


def test_max_steps_and_start(env, monkeypatch):
    tmp, ref = env
    code, out, _, core = run_script(tmp, ref, monkeypatch, "--max-steps", "2", "--start", "3")
    assert code == 0 and core.ran == [("b:2", "ctrl:expert"), ("b:2", "ctrl:flipped")]


def test_missing_env_var_exits_before_any_client(env, monkeypatch, capsys):
    tmp, ref = env
    monkeypatch.delenv("NEBIUS_PROJECT_ID")
    code, out, log, core = run_script(tmp, ref, monkeypatch)
    assert code == 2 and core is None and fake.ContreeSync.instances == []
    assert "NEBIUS_PROJECT_ID" in capsys.readouterr().err
    assert not out.exists() and not log.exists()


def test_billing_error_aborts_the_run(env, monkeypatch, capsys):
    tmp, ref = env
    code, out, log, core = run_script(tmp, ref, monkeypatch, core_kw={"billing": [("a:1", "student:x:0")]})
    assert code == 3
    assert core.ran == [("a:1", "ctrl:expert"), ("a:1", "ctrl:no_op"), ("a:1", "student:x:0")]
    assert len(rows(out)) == 2  # the aborted step has no labels
    last = [r for r in rows(log) if "step" in r][-1]
    assert last["step"] == 3 and last["status"] == "abort"
    summary = json.loads((tmp / "out" / "rescore.summary.json").read_text())
    assert summary["aborted"] and "billing" in summary["aborted"]
    text = all_written(tmp) + "".join(capsys.readouterr())
    assert KEY not in text and PROJECT not in text and "[redacted]" in text


def test_preflight_refuses_an_image_with_egress_or_missing_tools(env, monkeypatch):
    tmp, ref = env
    mod = load_script()
    for report, want in [("EGRESS open\n", 4), ("MISSING git\nEGRESS closed\n", 4), ("", 4), ("EGRESS closed\n", 0)]:
        monkeypatch.setitem(sys.modules, "contree_sdk", fake.module())
        orig = fake.Sandbox.__init__

        def init(self, *a, _r=report, **k):
            orig(self, *a, **k)
            self.bash_stdout = _r
        monkeypatch.setattr(fake.Sandbox, "__init__", init)
        monkeypatch.setattr(mod, "build_core", lambda a, f: StubCore(f))
        code = mod.main(["run", "--preflight", "--max-steps", "1", "--ref", str(ref),
                         "--out", str(tmp / "pf" / "o.jsonl"), "--log", str(tmp / "pf" / "s.jsonl")])
        assert code == want, report
        monkeypatch.setattr(fake.Sandbox, "__init__", orig)


def test_no_credential_in_any_written_file_or_output(env, monkeypatch, capsys, caplog):
    tmp, ref = env
    run_script(tmp, ref, monkeypatch, "--max-steps", "7")
    text = all_written(tmp) + "".join(capsys.readouterr()) + caplog.text
    assert KEY not in text and PROJECT not in text


def test_prepare_image_imports_installs_and_tags(env, monkeypatch):
    tmp, _ = env
    mod = load_script()
    code = mod.main(["prepare-image", "--base", "ubuntu:24.04", "--tag", "xp-h32-base",
                     "--log", str(tmp / "out" / "steps.jsonl")])
    assert code == 0
    calls = fake.ContreeSync.instances[-1].box.calls
    assert [c["command"] for c in calls] == ["oci", "bash", "tag_as"]
    assert calls[1]["disposable"] is False and "apt-get install" in calls[1]["args"][1]
    assert calls[2]["tag"] == "xp-h32-base"


def test_demo_core_builds_specimens_through_the_world_factory(env, monkeypatch):
    # the real DemoCore with a NebiusWorld factory: setup.sh and the expert replay go to the sandbox,
    # the probe's world is the NebiusWorld itself (no local jail), and the overlay gc never runs
    import functools

    from cleave.world import overlay
    from cleave.world.nebius import NebiusWorld, SpendGuard, make_client
    mod = load_script()
    client, guard = make_client(), SpendGuard(max_steps=5)
    factory = functools.partial(NebiusWorld, "xp-h32-base", client=client, guard=guard, run_cap_s=60)
    core = mod.build_core(None, factory)
    assert core.runner == "nebius"
    monkeypatch.setattr(overlay, "gc", lambda *a, **k: pytest.fail("overlay gc ran for a remote runner"))
    st = core.specs["billing-invoice-bugfix"]
    core._acquire()
    try:
        core._ready(st)
    finally:
        core._release()
    assert isinstance(st.jw, NebiusWorld)
    calls = client.box.calls
    assert calls[0]["image"] == "tag:xp-h32-base" and calls[0]["command"] == "mkdir"
    assert calls[1]["command"] == "bash" and st.sp.setup in calls[1]["args"][1]
    assert len(st.rr.steps) == len(st.sp.actions)  # one kept run per expert turn
    assert all(c["disposable"] is False for c in calls[1:])
