"""H16: crucible tool_use / code_exec ports, J, X and the oracle."""
import os
import random

import pytest

from pivots.xfamily import candidates as C
from pivots.xfamily import core, handwritten
from pivots.xfamily.envs import lookup_program

KIT = "/mnt/project-files/auto-search/evals/crucible/envs/tinylang.py"


def _tool(kind, seed=0):
    rng = random.Random(seed)
    while True:
        p = core.TOOL.generate(rng)
        if p["kind"] == kind:
            return p


def test_gold_traces_verify_and_expert_policy_reproduces_them():
    rng = random.Random(1)
    for _ in range(200):
        p = core.TOOL.generate(rng)
        assert core.TOOL.verify(p, p["trace"]) == 1.0
        for t, turn in enumerate(p["turns"]):
            events, _, _ = core.run_turns(p, p["turns"][:t])
            assert core.expert_next(p, events) == turn
            assert core.remaining(p, events) == len(p["turns"]) - t
    for _ in range(200):
        q = core.CODE.generate(rng)
        assert core.CODE.verify(q, q["gold"]) == 1.0
        assert core.CODE.verify(q, lookup_program(q["visible"])) == 0.0


def test_tool_oracle_working_definition():
    p = _tool("add")
    a, b = core.TOOL.involved(p)
    va, vb = p["state"][a], p["state"][b]
    assert core.tool_oracle(p, [], "!" + a)["working"] == 1
    assert core.tool_oracle(p, [], "!" + b)["working"] == 1           # reordered lookup: progress
    assert core.tool_oracle(p, ["!" + a], "!" + a)["working"] == 0    # re-read: no progress
    pre = ["!" + a, "!" + b]
    assert core.tool_oracle(p, pre, f"!{vb}+{va}")["working"] == 1
    assert core.tool_oracle(p, pre, str(va + vb))["working"] == 1     # early right answer
    assert core.tool_oracle(p, pre, str(va + vb + 1))["working"] == 0
    assert core.tool_oracle(p, pre, f"!{va}-{vb}")["working"] == 0


def test_add3_has_no_spare_turn():
    p = _tool("add3")
    a = core.TOOL.involved(p)[0]
    o = core.tool_oracle(p, ["!" + a], "!" + a)
    assert o["success"] is False and o["working"] == 0


def test_tool_x_compares_effects_not_text():
    p = _tool("add")
    a, b = core.TOOL.involved(p)
    va, vb = p["state"][a], p["state"][b]
    pre = ["!" + a, "!" + b]
    assert core.tool_x(p, pre, f"!{va}+{vb}", f"!{vb}+{va}")[0] == 1
    assert core.tool_x(p, pre, f"!{va}+{vb}", f"!({va}+{vb})")[0] == 1
    assert core.tool_x(p, pre, f"!{va}+{vb}", f"!{va}+{vb+1}")[0] == 0
    assert core.tool_x(p, pre, f"!{va}+{vb}", str(va + vb))[0] == 0   # answer is not a calculator result
    assert core.tool_x(p, [], "!" + a, f"! {a}")[0] == 1
    assert core.tool_x(p, [], "!" + a, "!" + b)[0] == 0              # horizon 0: a different fact
    assert core.tool_x(p, [], "!" + a, "!" + p["prompt"])[0] == 0      # tool error
    g = str(va + vb)
    assert core.tool_x(p, pre + [f"!{va}+{vb}"], g, "0" + g)[0] == 1
    assert core.tool_x(p, pre + [f"!{va}+{vb}"], g, "+" + g)[0] == 0


def test_code_x_and_j():
    assert core.code_x("2*x+7", "x*2+7")[0] == 1
    assert core.code_x("2*x+7", "7+2*x")[0] == 1
    assert core.code_x("2*x+7", "x=0?7:x=1?9:11")[0] == 0
    assert core.code_x("2*x+7", "2*x+7+x/100")[0] == 0
    assert core.code_x("2*x+7", "2x+7")[0] == 0
    assert core.j_credit("2*x*x-8*x+19", "2*x*x-8*x+18") == 1          # J's blind spot
    assert core.j_credit("2*x+7", "x*2+7") == 0


def test_candidate_generation_is_seeded_and_sized():
    a, b = C.tool_candidates(), C.tool_candidates()
    assert [r["cand"] for r in a] == [r["cand"] for r in b]
    c = C.code_candidates()
    assert len(a) >= 150 and len(c) >= 150
    for r in c:
        if r["draft"] is not None:
            assert core.CODE.verify(r["problem"], r["draft"]) == 0.0


def test_handwritten_sets():
    t, c = handwritten.tool_records(), handwritten.code_records()
    assert len(t) >= 40 and len(c) >= 40
    assert all(r["source"] == "handwritten" for r in t + c)
    for r in t:
        assert r["expert"] == r["problem"]["turns"][r["t"]]


def test_blind_sets():
    from pivots.xfamily import blind
    t, c = blind.tool_records(), blind.code_records()
    assert len(t) >= 60 and len(c) >= 60
    assert all(r["source"] == "blind" and r["cand"] != r["expert"] for r in t + c)
    assert {blind.TOOL_SEED, blind.CODE_SEED}.isdisjoint({C.TOOL_SEED, C.CODE_SEED, handwritten.TOOL_SEED,
                                                        handwritten.CODE_SEED})
    keys = [(r["pid"], r.get("t"), r["cand"]) for r in t + c]
    assert len(keys) == len(set(keys))
    for r in c:
        assert r["pid"].endswith(r["pivot"]) and (r["draft"] is None) == (r["pivot"] == "final")

@pytest.mark.skipif(not os.path.exists(KIT), reason="read-only crucible kit not mounted")
def test_tinylang_is_the_kit_copy():
    here = os.path.join(os.path.dirname(core.__file__), "tinylang.py")
    with open(here) as f:
        ours = f.read().split("\n", 2)[2]
    with open(KIT) as f:
        assert ours == f.read()
