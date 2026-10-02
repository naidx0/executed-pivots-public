"""H30: at read-only pivots, a candidate that also changes nothing is scored on informative-token recall.

Before H30 the output score was whole-output token F1: a working student who read the same facts with a wider
command (cat instead of grep) printed many more tokens and fell under the threshold (21 of 37 H24 misses).
"""
from pivots.effect import EffectJudge
from pivots.effect.probe import Effects
from pivots.effect.xreward import informative_recall, informative_tokens

SRC = "\n".join(f"// filler line {i} of the module body, nothing to see here" for i in range(40))
REF_OUT = ("3:const fs = require('fs');\n17:function summarize(rows, column_name) {\n"
           "42:  throw new TypeError('bad column');\n")
WIDE_OUT = ("const fs = require('fs');\n" + SRC + "\nfunction summarize(rows, column_name) {\n" + SRC
            + "\n  throw new TypeError('bad column');\n  return 3 + 17 + 42;\n")


def eff(output, changed=None):
    return Effects(output, 0, "/app", dict(changed or {}), set(), roots=["/app"])


def test_informative_tokens_are_paths_identifiers_numbers_and_error_names():
    toks = informative_tokens("ls: /app/src/index.js has 17 lines, see README.md; column_name fooBar "
                              "raised TypeError ENOENT and the file was fine 7 times\n")
    assert {"/app/src/index.js", "17", "README.md", "column_name", "fooBar", "TypeError", "ENOENT"} <= toks
    assert not toks & {"ls", "has", "lines", "see", "raised", "and", "the", "file", "was", "fine", "7", "times"}


def test_wider_read_that_recalls_the_facts_gets_credit():
    j = EffectJudge(world=None)
    c = j._compare(eff(REF_OUT), eff(WIDE_OUT), set(), {}, {})
    assert c["reward"] >= 0.75 and c["detail"]["readonly_recall"]["recall"] >= 0.6


def test_read_that_misses_the_facts_gets_no_credit():
    j = EffectJudge(world=None)
    miss = "const fs = require('fs');\n" + SRC + "\n"
    c = j._compare(eff(REF_OUT), eff(miss), set(), {}, {})
    assert c["reward"] < 0.75 and c["detail"]["readonly_recall"]["recall"] < 0.6


def test_recall_needs_a_candidate_that_changes_nothing():
    j = EffectJudge(world=None)
    c = j._compare(eff(REF_OUT), eff(WIDE_OUT, {"/app/z": "e" * 64}), set(), {}, {})
    assert "readonly_recall" not in c["detail"] and c["reward"] < 0.75


def test_state_changing_pivots_are_unchanged():
    j = EffectJudge(world=None)
    ref = eff(REF_OUT, {"/app/a": "a" * 64})
    c = j._compare(ref, eff(WIDE_OUT, {"/app/a": "a" * 64}), set(), {}, {})
    assert "readonly_recall" not in c["detail"]


def test_recall_off_restores_whole_output_f1():
    j = EffectJudge(world=None, readonly_recall=False)
    assert j._compare(eff(REF_OUT), eff(WIDE_OUT), set(), {}, {})["reward"] < 0.75


def test_no_informative_token_falls_back_to_f1():
    assert informative_recall("ok done", "something else") is None
