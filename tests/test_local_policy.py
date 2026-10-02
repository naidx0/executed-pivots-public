"""pivots.students.local_policy: the adapter between Gym's policy_model and a local llama-server / Ollama.

CPU only, no model: a scripted fake OpenAI server (tests/_fake_openai.py) stands in for the local model.
Covers where a thinking model puts its answer (content, <think> + fences, reasoning-only), malformed JSON,
the timeout and retry, the settings coming from the environment, and the proxy Gym talks to.
"""
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from _fake_openai import FakeOpenAI
from pivots.effect.terminus import parse_action
from pivots.students import local_policy as lp

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION = {"analysis": "look", "plan": "list files", "commands": [{"keystrokes": "ls -la\n", "duration": 0.1}],
          "task_complete": False}
ACTION_TEXT = json.dumps(ACTION)


@pytest.fixture
def fake():
    servers = []

    def make(script=None, default=("content", ACTION_TEXT)):
        s = FakeOpenAI(script, default).start()
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.stop()


def _policy(url, **kw):
    kw.setdefault("sleep", lambda s: None)
    return lp.LocalPolicy(url, "fake-local", **kw)


# --- where the action is ------------------------------------------------------------------------------
def test_plain_json_content_is_passed_verbatim(fake):
    r = _policy(fake().url).chat([{"role": "user", "content": "go"}])
    assert (r["text"], r["source"], r["error"]) == (ACTION_TEXT, "content", None)
    assert r["usage"]["completion_tokens"] == 20 and r["attempts"] == 1


def test_think_block_and_fences_are_stripped(fake):
    r = _policy(fake([("think", ACTION_TEXT)]).url).chat([{"role": "user", "content": "go"}])
    assert r["source"] == "content" and json.loads(r["text"]) == ACTION
    assert parse_action(r["text"]).strict_ok  # what J (strict json.loads) sees is the bare action
    assert "look first" in r["reasoning"] and "<think>" not in r["text"]


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_reasoning_only_reply_yields_the_action(fake, field):
    thinking = f"The listing will show the project. Final answer:\n{ACTION_TEXT}"
    r = _policy(fake([(field, thinking)]).url).chat([{"role": "user", "content": "go"}])
    assert r["source"] == "reasoning" and json.loads(r["text"]) == ACTION
    assert r["reasoning"] == thinking


def test_reasoning_without_an_action_is_empty_not_invented(fake):
    r = _policy(fake([("reasoning", "I am still thinking about {what to do}")]).url).chat([{"role": "user", "content": "go"}])
    assert (r["text"], r["source"]) == ("", "empty")
    assert parse_action(r["text"]).action is None  # executed_pivot pays it 0 as model_output_invalid


@pytest.mark.parametrize("bad", [
    '{"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\\n"}], "task_complete": false',  # truncated
    '{"analysis": "a", "plan": "p", "commands": "ls"}',  # schema violation
    "I would run ls -la now.",  # prose only
])
def test_malformed_json_passes_through_unchanged(fake, bad):
    r = _policy(fake([("content", bad)]).url).chat([{"role": "user", "content": "go"}])
    assert r["text"] == bad and r["source"] == "invalid" and r["error"]
    assert parse_action(r["text"]).action is None


def test_content_wins_over_reasoning_when_both_hold_actions():
    other = dict(ACTION, commands=[{"keystrokes": "pwd\n", "duration": 0.1}])
    ex = lp.extract_action({"content": ACTION_TEXT, "reasoning": json.dumps(other)})
    assert ex.source == "content" and json.loads(ex.text) == ACTION


# --- timeout and retry --------------------------------------------------------------------------------
def test_timeout_is_retried_then_answers(fake):
    f = fake([("sleep", 1.5, "{}")])
    r = _policy(f.url, timeout_s=0.3, retries=2).chat([{"role": "user", "content": "go"}])
    assert r["attempts"] == 2 and r["text"] == ACTION_TEXT and len(f.calls) == 2


def test_timeout_every_attempt_fails_with_the_count(fake):
    f = fake(default=("sleep", 1.5, ACTION_TEXT))
    with pytest.raises(lp.LocalPolicyError, match=r"gave up after 3 attempts, timeout 0.3 s"):
        _policy(f.url, timeout_s=0.3, retries=2).chat([{"role": "user", "content": "go"}])


def test_5xx_is_retried_and_4xx_is_not(fake):
    ok = _policy(fake([("status", 503), ("status", 500)]).url, retries=2).chat([{"role": "user", "content": "go"}])
    assert ok["attempts"] == 3 and ok["source"] == "content"
    f = fake([("status", 400)])
    with pytest.raises(lp.LocalPolicyError, match="HTTP 400"):
        _policy(f.url, retries=2).chat([{"role": "user", "content": "go"}])
    assert len(f.calls) == 1


def test_refused_connection_is_retried_then_fails():
    p = _policy("http://127.0.0.1:9/v1", timeout_s=1, retries=1)  # discard port: nothing listens
    with pytest.raises(lp.LocalPolicyError, match="gave up after 2 attempts"):
        p.chat([{"role": "user", "content": "go"}])
    assert p.attempts == 2


# --- settings from the environment, key never written ---------------------------------------------------
def test_settings_come_from_env_and_the_key_only_goes_in_the_header(fake, tmp_path):
    f = fake()
    env = {lp.ENV_BASE_URL: f.url, lp.ENV_MODEL: "qwen3:4b", lp.ENV_KEY: "sk-local-SENTINEL", lp.ENV_TIMEOUT: "7",
           lp.ENV_RETRIES: "1", lp.ENV_MAX_TOKENS: "512"}
    p = lp.LocalPolicy.from_env(env, sleep=lambda s: None)
    assert (p.model, p.timeout_s, p.retries, p.params["max_tokens"]) == ("qwen3:4b", 7.0, 1, 512)
    log = tmp_path / "proxy.jsonl"
    srv = lp.make_proxy(p, port=0, log_path=str(log))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        _post(f"http://127.0.0.1:{srv.server_port}/v1/chat/completions", {"model": "x", "messages": [{"role": "user", "content": "go"}]})
    finally:
        srv.shutdown()
        srv.log_fh.close()
    assert f.calls[0]["auth"] == "Bearer sk-local-SENTINEL" and f.calls[0]["body"]["model"] == "qwen3:4b"
    assert "SENTINEL" not in log.read_text()
    no_key = lp.LocalPolicy.from_env({lp.ENV_BASE_URL: f.url}, sleep=lambda s: None)
    no_key.chat([{"role": "user", "content": "go"}])
    assert f.calls[-1]["auth"] is None and f.calls[-1]["body"]["model"] == "local"
    with pytest.raises(lp.LocalPolicyError, match=lp.ENV_BASE_URL):
        lp.settings_from_env({})


def test_no_key_or_endpoint_is_written_in_the_config():
    text = (REPO_ROOT / "resources_servers/executed_pivot/configs/local_policy.yaml").read_text()
    assert "${oc.env:LOCAL_POLICY_PROXY_URL," in text and "${oc.env:LOCAL_POLICY_MODEL," in text
    assert "sk-" not in text and "uses_reasoning_parser: true" in text


# --- the proxy Gym talks to ---------------------------------------------------------------------------
def _post(url, body, timeout=30):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_proxy_returns_the_action_as_content_and_thinking_separately(fake, tmp_path):
    f = fake([("reasoning", f"thinking... {ACTION_TEXT}"), ("content", "not json at all"), ("status", 503),
              ("status", 503)])
    log = tmp_path / "proxy.jsonl"
    srv = lp.make_proxy(_policy(f.url, retries=1), port=0, log_path=str(log))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v1/chat/completions"
    try:
        s1, r1 = _post(url, {"model": "gym-name", "messages": [{"role": "user", "content": "go"}], "temperature": 0.5})
        s2, r2 = _post(url, {"messages": [{"role": "user", "content": "go"}]})
        s3, r3 = _post(url, {"messages": [{"role": "user", "content": "go"}]})
    finally:
        srv.shutdown()
        srv.log_fh.close()
    m1 = r1["choices"][0]["message"]
    assert s1 == 200 and json.loads(m1["content"]) == ACTION and m1["reasoning_content"].startswith("thinking")
    assert r1["local_policy"]["source"] == "reasoning" and f.calls[0]["body"]["temperature"] == 0.5
    assert f.calls[0]["body"]["model"] == "fake-local"  # the configured model, not Gym's name for it
    assert s2 == 200 and r2["choices"][0]["message"]["content"] == "not json at all"
    assert s3 == 502 and r3["error"]["code"] == "upstream_error"  # infrastructure, not a policy answer
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert [x["status"] for x in lines] == ["ok", "ok", "upstream_error"]
    assert [x.get("source") for x in lines[:2]] == ["reasoning", "invalid"]
