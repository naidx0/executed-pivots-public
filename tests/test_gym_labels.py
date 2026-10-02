"""pivots.gym.labels: reading both rollout shapes, joining E and P, and the H31 verdict. CPU, no sandbox:
the E/P labeller is replaced by a table, so this runs on Windows too (the real labeller is exercised in
tests/test_gym_local_rollouts.py on Linux)."""
import json
from pathlib import Path

from pivots.gym import labels as L

ACT = json.dumps({"analysis": "a", "plan": "p", "commands": [{"keystrokes": "ls\n", "duration": 0.1}],
                  "task_complete": False})


def _direct(uuid, text, reward, j, masked=False, kind=None):
    return {"uuid": uuid, "text": text, "reward": reward, "j_reward": j, "mask_sample": masked, "failure_kind": kind,
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50}, "gen_wall_s": 2.0, "verify_s": 0.5,
            "policy": {"source": "reasoning"}}


def _gym(uuid, text, reward, j):
    return {"uuid": uuid, "reward": reward, "j_reward": j, "mask_sample": False, "verify_s": 0.4,
            "response": {"output": [{"type": "reasoning", "summary": []},
                                    {"type": "message", "role": "assistant",
                                     "content": [{"type": "output_text", "text": text}]}],
                         "usage": {"input_tokens": 900, "output_tokens": 40}}}


def test_both_rollout_shapes_load_the_same_fields(tmp_path: Path):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(_direct("jq-signup-rollup:2", ACT, 1.0, 0.0)) + "\n" +
                 json.dumps(_gym("tally-go-build:10", ACT, 0.0, 1.0)) + "\n")
    a, b = L.load_rollouts(p)
    assert (a["task"], a["turn"], a["X"], a["J"], a["text"], a["source"]) == ("jq-signup-rollup", 2, 1, 0, ACT, "reasoning")
    assert (b["task"], b["turn"], b["X"], b["J"], b["text"]) == ("tally-go-build", 10, 0, 1, ACT)
    assert (b["prompt_tokens"], b["completion_tokens"]) == (900, 40)


def _labelled(pairs):
    """[(X, J, E)] -> labelled rows."""
    return [{"X": x, "J": j, "E": e, "P": e, "failure_kind": None} for x, j, e in pairs]


def test_verdict_win_loss_and_inconclusive():
    rollouts = [{"uuid": "t:0", "masked": False}]
    win = L.summarize(_labelled([(1, 0, 1), (0, 1, 0), (0, 0, 0)] * 40), rollouts)
    assert win["h31"]["verdict"] == "win" and win["X_vs_E"]["agree"] == 120 and win["J_vs_E"]["agree"] == 40
    assert win["paired"] == {"only_X_right": 80, "only_J_right": 0, "sign_test_p": L.binom_two_sided(80, 0)}
    one_false = L.summarize(_labelled([(1, 0, 1)] * 119 + [(1, 0, 0)]), rollouts)
    assert one_false["h31"]["verdict"] == "loss: X false credits 1"  # agreement alone does not win
    worse = L.summarize(_labelled([(0, 1, 1)] * 100 + [(0, 0, 0)] * 20), rollouts)
    assert worse["h31"]["verdict"].startswith("loss: X agrees with E on 20 < J's 120")
    invalid = [{"X": 0, "J": 0, "E": 0, "P": 0, "failure_kind": "executed_pivot:model_output_invalid"}] * 50
    mixed = L.summarize(_labelled([(1, 0, 1)] * 60) + invalid, rollouts)
    assert mixed["X_vs_E"]["agree"] == 110 and mixed["parsed"]["n"] == 60
    assert (mixed["parsed"]["X_vs_E"]["agree"], mixed["parsed"]["J_vs_E"]["agree"]) == (60, 0)
    few = L.summarize(_labelled([(1, 0, 1)] * 10), rollouts)
    assert few["h31"]["verdict"] == "inconclusive: 10 labelled rollouts < 100"


def test_sign_test():
    assert L.binom_two_sided(0, 0) is None
    assert L.binom_two_sided(5, 5) == 1.0
    assert abs(L.binom_two_sided(8, 0) - 2 / 256) < 1e-12


def test_run_joins_labels_skips_masked_and_executes_each_distinct_text_once(tmp_path: Path):
    other = ACT.replace("ls", "pwd")
    rows = [_direct("billing-invoice-bugfix:3", ACT, 1.0, 1.0), _direct("billing-invoice-bugfix:3", ACT, 1.0, 1.0),
            _direct("billing-invoice-bugfix:3", other, 0.0, 1.0),
            _direct("billing-invoice-bugfix:4", None, 0.0, None, True, "executed_pivot:policy_error")]
    p = tmp_path / "rollouts.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    calls = []

    def labeler(sp_dir, items, store, workers):
        calls.append((Path(sp_dir).name, sorted(set(items))))
        return {k: {"E": int(k[1] == ACT), "P": 1, "harm": 0} for k in set(items)}

    log = tmp_path / "policy.jsonl"
    log.write_text(json.dumps({"status": "ok", "source": "reasoning", "attempts": 2, "gen_s": 3.0,
                               "usage": {"prompt_tokens": 1000, "completion_tokens": 50}}) + "\n" +
                   json.dumps({"status": "upstream_error", "error": "HTTP 503"}) + "\n")
    s = L.run(str(p), str(tmp_path / "out"), "/tmp/x", labeler=labeler, policy_log=str(log))
    assert s["proxy"] == {"requests": 2, "upstream_errors": 1, "action_source": {"reasoning": 1}, "retried": 1,
                          "prompt_tokens": 1000, "completion_tokens": 50, "gen_s_total": 3.0, "gen_s_median": 3.0}
    assert calls == [("billing-invoice-bugfix", sorted({(3, ACT), (3, other)}))]
    assert s["labelled"] == 3 and s["masked"] == {"executed_pivot:policy_error": 1}
    assert s["X_vs_E"]["agree"] == 3 and s["J_vs_E"]["false_credit"] == 1 and s["X_vs_E"]["false_credit"] == 0
    assert s["cost"]["prompt_tokens"] == 4000 and s["action_source"] == {"reasoning": 4}
    out = tmp_path / "out"
    assert len((out / "labels.jsonl").read_text().splitlines()) == 3
    assert (out / "summary.md").read_text().startswith("# H31 rollouts: inconclusive")
