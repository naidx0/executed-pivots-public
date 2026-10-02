"""NebiusWorld (contree-sdk) and its spend guard, against a fake contree_sdk module (no key, no network).

The fake is injected through sys.modules, so these tests never reach Nebius even though the real
contree-sdk may be installed. A planted fake key and project id stand in for the real ones.
"""
import json
import sys

import pytest

import _fake_contree_sdk as fake
from cleave.world import nebius
from cleave.world.nebius import (MissingCredentials, NebiusWorld, SpendAbort, SpendGuard, StepCapReached,
                                 looks_like_billing, make_client, redact, require_env)

KEY = "nbk-PLANTED-FAKE-KEY-7f3a9c"
PROJECT = "project-e00planted0fake"


@pytest.fixture
def sdk(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", KEY)
    monkeypatch.setenv("NEBIUS_PROJECT_ID", PROJECT)
    mod = fake.module()
    monkeypatch.setitem(sys.modules, "contree_sdk", mod)
    return mod


def world(sdk, guard=None, **kw):
    client = make_client()
    return NebiusWorld("xp-h32-base", client=client, guard=guard or SpendGuard(max_steps=5), **kw), client


# -- credentials -------------------------------------------------------------

@pytest.mark.parametrize("missing", ["NEBIUS_API_KEY", "NEBIUS_PROJECT_ID"])
def test_missing_env_var_fails_fast_before_any_client(monkeypatch, missing):
    monkeypatch.setenv("NEBIUS_API_KEY", KEY)
    monkeypatch.setenv("NEBIUS_PROJECT_ID", PROJECT)
    monkeypatch.delenv(missing)
    mod = fake.module()
    monkeypatch.setitem(sys.modules, "contree_sdk", mod)
    with pytest.raises(MissingCredentials) as e:
        make_client()
    assert missing in str(e.value)
    assert KEY not in str(e.value) and PROJECT not in str(e.value)
    assert fake.ContreeSync.instances == []  # no client was built, so no request could be made


def test_empty_env_var_counts_as_missing(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "  ")
    monkeypatch.setenv("NEBIUS_PROJECT_ID", PROJECT)
    with pytest.raises(MissingCredentials, match="NEBIUS_API_KEY"):
        require_env()


# -- the world ---------------------------------------------------------------

def test_success_path_checkpoint_fork_and_disposable(sdk):
    w, client = world(sdk)
    box = client.box
    assert box.calls[0]["command"] == "mkdir" and box.calls[0]["image"] == "tag:xp-h32-base"
    base = w.checkpoint()
    r = w.run(["set", "a", "1"])
    assert r.ok and w.checkpoint() != base
    after = w.checkpoint()
    assert w.run(["cat", "a"], keep=False).stdout == "1"
    assert w.checkpoint() == after  # a disposable run does not advance the world
    assert box.calls[-1]["disposable"] is True and box.calls[-2]["disposable"] is False
    assert box.calls[-1]["args"] == ["a"]  # argv is passed as command + args, not a shell string

    f = w.fork(base)  # a fork starts from the older image: it has no `a`
    assert f.run(["cat", "a"]).stdout == ""
    assert box.calls[-1]["image"] == base
    assert w.run(["exit", "3"]).exit_code == 3


def test_timeout_reads_as_124_like_the_local_runner(sdk):
    w, _ = world(sdk)
    r = w.run(["sleep"], timeout_s=5)
    assert r.exit_code == 124 and "timeout after" in r.stderr


def test_run_cap_bounds_every_timeout(sdk):
    w, client = world(sdk, run_cap_s=60)
    w.run(["cat", "x"], timeout_s=600)
    assert client.box.calls[-1]["timeout"] == 60


def test_put_and_get(sdk, tmp_path):
    w, _ = world(sdk)
    d = tmp_path / "tests"
    d.mkdir()
    (d / "test.sh").write_text("echo PASS\n")
    w.put(d, "/tests")
    got = tmp_path / "out.sh"
    w.get("/tests/test.sh", got)
    assert got.read_text() == "echo PASS\n"


def test_plain_api_error_is_an_op_failure_not_an_abort(sdk):
    g = SpendGuard(max_steps=5)
    w, _ = world(sdk, guard=g)
    r = w.run(["boom"])
    assert r.exit_code == 125 and "node lost" in r.stderr and g.tripped is None
    assert w.run(["cat", "a"]).ok  # the world is still usable


def test_consecutive_api_errors_abort(sdk):
    g = SpendGuard(max_steps=5, max_consecutive_errors=3)
    w, _ = world(sdk, guard=g)
    w.run(["boom"])
    w.run(["boom"])
    with pytest.raises(SpendAbort):
        w.run(["boom"])


# -- spend guard -------------------------------------------------------------

@pytest.mark.parametrize("cmd", ["billing", "quota429"])
def test_billing_error_aborts_and_blocks_every_later_op(sdk, cmd, tmp_path):
    log = tmp_path / "steps.jsonl"
    g = SpendGuard(max_steps=5, log_path=log)
    w, client = world(sdk, guard=g)
    with pytest.raises(SpendAbort) as e:
        with g.step(pivot="p:1", action="ctrl:expert"):
            w.run([cmd])
    assert g.tripped and KEY not in str(e.value) and PROJECT not in str(e.value)
    n = len(client.box.calls)
    with pytest.raises(SpendAbort):
        w.run(["cat", "a"])  # tripped: nothing more reaches the platform
    assert len(client.box.calls) == n
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert rows[-1]["status"] == "abort" and rows[-1]["pivot"] == "p:1"


def test_spend_abort_is_not_swallowed_by_except_exception(sdk):
    # the probe and the demo core catch Exception; the abort must pass through them
    g = SpendGuard(max_steps=5)
    w, _ = world(sdk, guard=g)
    with pytest.raises(SpendAbort):
        try:
            w.run(["billing"])
        except Exception:  # noqa: BLE001
            pytest.fail("SpendAbort was caught by `except Exception`")


def test_max_steps_cap(tmp_path):
    log = tmp_path / "steps.jsonl"
    g = SpendGuard(max_steps=2, log_path=log)
    for i in range(2):
        with g.step(pivot=f"p:{i}", action="a"):
            pass
    with pytest.raises(StepCapReached):
        with g.step(pivot="p:2", action="a"):
            pytest.fail("a third step ran under max_steps=2")
    rows = [json.loads(x) for x in log.read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2]
    assert all(isinstance(r["wall_s"], float) and r["status"] == "ok" for r in rows)


def test_max_ops_cap(sdk):
    g = SpendGuard(max_steps=5, max_ops=3)
    w, _ = world(sdk, guard=g)  # mkdir is op 1
    w.run(["cat", "a"])
    w.run(["cat", "a"])
    with pytest.raises(SpendAbort, match="max_ops"):
        w.run(["cat", "a"])


def test_step_log_counts_ops_and_wall_time(sdk, tmp_path):
    log = tmp_path / "steps.jsonl"
    g = SpendGuard(max_steps=5, log_path=log)
    w, _ = world(sdk, guard=g)
    with g.step(pivot="p:1", action="a") as row:
        w.run(["cat", "a"])
        w.run(["cat", "a"])
        row["match"] = True
    r = json.loads(log.read_text().splitlines()[-1])
    assert r["ops"] == 2 and r["match"] is True and r["wall_s"] >= 0 and r["cost"] == pytest.approx(0.002)


@pytest.mark.parametrize("exc,billing", [
    (fake.ApiStatusCodeError(402, "Payment Required"), True),
    (fake.ApiStatusCodeError(403, "Forbidden"), True),
    (fake.TooManyRequestsError(), True),
    (RuntimeError("quota exceeded for sandboxes"), True),
    (RuntimeError("insufficient balance on the billing account"), True),
    (fake.FailedOperationError("Operation 1 has failed: node lost"), False),
    (fake.ApiStatusCodeError(404, "image not found"), False),
])
def test_billing_classifier(exc, billing):
    assert looks_like_billing(exc) is billing


# -- no credential leaks -----------------------------------------------------

def test_redact(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", KEY)
    monkeypatch.setenv("NEBIUS_PROJECT_ID", PROJECT)
    s = redact(f"token {KEY} project {PROJECT} Authorization: Bearer abc.def")
    assert KEY not in s and PROJECT not in s and "abc.def" not in s


def test_no_credential_in_logs_or_errors(sdk, tmp_path, capsys, caplog):
    log = tmp_path / "steps.jsonl"
    g = SpendGuard(max_steps=5, log_path=log)
    w, client = world(sdk, guard=g)
    assert KEY not in repr(w) and PROJECT not in repr(w)
    with pytest.raises(SpendAbort) as e:
        with g.step(pivot="p:1", action="a"):
            w.run(["boom"])
            w.run(["billing"])
    text = log.read_text() + str(e.value) + repr(e.value) + capsys.readouterr().out + caplog.text
    assert "[redacted]" in log.read_text()  # the planted key was in the error, and was removed
    assert KEY not in text and PROJECT not in text
    assert nebius.ENV_KEY == "NEBIUS_API_KEY" and nebius.ENV_PROJECT == "NEBIUS_PROJECT_ID"


# -- H34: the sandbox starts runs with HOME unset, so git ignored /root/.gitconfig and a revert could not commit

def test_runs_default_home_to_root_like_the_local_runner(sdk):
    w, client = world(sdk)
    w.run(["git", "config", "-l"])
    assert client.box.calls[-1]["env"].get("HOME") == "/root"


def test_an_explicit_home_is_kept(sdk):
    w, client = world(sdk)
    w.run(["true"], env={"HOME": "/home/x", "A": "1"})
    assert client.box.calls[-1]["env"] == {"HOME": "/home/x", "A": "1"}
