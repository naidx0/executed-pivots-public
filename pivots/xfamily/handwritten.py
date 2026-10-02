"""Handwritten H16 candidates, written by thinking as a student model would.

Written by hand before any label existed, on problems the generated set does not
use (tool_use seed 16101, code_exec seed 16102; the drafts come from
candidates.make_draft with seed 16102+7). Every record is marked
source="handwritten". Categories are the author's intent, not a label.

tool_use rows: (problem index, turn index t, candidate turn, category, note)
code_exec rows: (problem index, "final" | "edit", candidate program, category, note)
"""
from __future__ import annotations

import random

from . import core
from .candidates import make_draft

TOOL_SEED, CODE_SEED = 16101, 16102

TOOL = [
    # 0 kz+ly? add   kz=11 ly=92   ['!kz', '!ly', '!11+92', '103']
    (0, 0, "!ly", "equiv", "reads the second key first"),
    (0, 0, "!kz+ly", "near", "tries to add keys inside the calculator"),
    (0, 0, "kz", "early", "answers with the key name"),
    (0, 0, "!KZ", "near", "upper-case key"),
    (0, 1, "!11+ly", "near", "mixes a value and a key in the calculator"),
    (0, 2, "!92+11", "equiv", "commuted"),
    (0, 2, "!11+92=103", "near", "writes the result into the call"),
    (0, 2, "103", "early", "does the sum in its head"),
    (0, 2, "113", "early", "sum in its head, wrong"),
    (0, 3, "103.", "near", "trailing period"),
    (0, 3, "The answer is 103", "near", "prose answer"),
    # 1 be>kg? max   be=55 kg=49   ['!be', '!kg', '55']
    (1, 0, "!kg", "equiv", "reads the second key first"),
    (1, 0, "!be>kg", "near", "asks the tool to compare"),
    (1, 1, "!be", "useless", "re-reads be"),
    (1, 1, "55", "early", "answers with the only value it has"),
    (1, 2, "49", "near", "takes the min"),
    (1, 2, " 55", "equiv", "leading space"),
    # 3 td-jo? sub   td=91 jo=49   ['!td', '!jo', '!91-49', '42']
    (3, 2, "!49-91", "near", "operands swapped under subtraction"),
    (3, 2, "!91 - 49", "equiv", "spaces"),
    (3, 2, "42", "early", "subtracts in its head"),
    (3, 2, "!91+-49", "near", "unary minus the calculator rejects"),
    (3, 2, "!91+(0-49)", "equiv", "adds the negative"),
    (3, 3, "-42", "near", "sign flipped"),
    # 4 gm*3? mul   gm=35   ['!gm', '!35*3', '105']
    (4, 1, "!3*35", "equiv", "commuted"),
    (4, 1, "!35+35+35", "equiv", "repeated addition"),
    (4, 1, "!gm*3", "near", "key inside the calculator"),
    (4, 1, "105", "early", "multiplies in its head"),
    (4, 1, "!35*3*1", "equiv", "redundant factor"),
    (4, 2, "0105", "equiv", "leading zero"),
    (4, 2, "35*3", "near", "answers with the expression"),
    # 6 bv+iv+fj? add3   28 33 53   ['!bv', '!iv', '!fj', '!28+33+53', '114']
    (6, 0, "!fj", "equiv", "reads the last key first"),
    (6, 0, "!bv+iv+fj", "near", "keys in the calculator"),
    (6, 1, "!fj", "equiv", "reads fj before iv"),
    (6, 3, "!28+33+5", "near", "truncated the last operand"),
    (6, 3, "!61+53", "equiv", "added the first two in its head"),
    (6, 3, "114", "early", "answers without the calculator"),
    (6, 4, "!114", "near", "a tool call on the last turn"),
    # 8 gg+dw+rj? add3   46 93 72   ['!gg', '!dw', '!rj', '!46+93+72', '211']
    (8, 3, "!46+93+27", "near", "transposed digits"),
    (8, 3, "!46+39+72", "near", "transposed digits"),
    (8, 3, "201", "early", "carry slip in its head"),
    (8, 4, "221", "near", "carry slip"),
    # 9 pe+ze? add   49 71   ['!pe', '!ze', '!49+71', '120']
    (9, 0, "!ze", "equiv", "reads the second key first"),
    (9, 0, "!pe ", "equiv", "trailing space"),
    (9, 0, "!ys", "near", "reads an unrelated key"),
    (9, 2, "!71+49", "equiv", "commuted"),
    (9, 2, "!49+17", "near", "digit swap"),
    (9, 3, "12O", "near", "letter O for zero"),
    # 11 eg>az? max   20 45   ['!eg', '!az', '45']
    (11, 1, "20", "early", "answers with the first value"),
    (11, 1, "!eg", "useless", "re-reads eg"),
    (11, 2, "20", "near", "takes the min"),
    # 12 gr-my? sub   33 48   ['!gr', '!my', '!33-48', '-15']
    (12, 2, "!48-33", "near", "swapped, gives +15"),
    (12, 2, "-15", "early", "subtracts in its head"),
    (12, 2, "!(33)-(48)", "equiv", "brackets"),
    (12, 3, "15", "near", "drops the sign"),
    (12, 3, "- 15", "near", "space after the minus"),
    (12, 3, "-015", "equiv", "leading zero"),
    # 13 wg? get   85   ['!wg', '85']
    (13, 0, "!gw", "near", "letters swapped"),
    (13, 0, "50", "early", "guesses"),
    (13, 0, "!wg?", "near", "question mark kept"),
    (13, 1, "!85", "useless", "echoes the value through the calculator"),
    (13, 1, "wg=85", "near", "answers with an equation"),
]

CODE = [
    # 0 -x*x+2*x-18   draft -x*x+2*x-16
    (0, "final", "2*x-x*x-18", "equiv", "terms reordered"),
    (0, "final", "-(x*x)+2*x-18", "equiv", "brackets"),
    (0, "final", "-(x-1)*(x-1)-17", "equiv", "completed the square"),
    (0, "final", "x*x+2*x-18", "near", "dropped the leading minus"),
    (0, "final", "-x*x+2x-18", "near", "implicit multiplication"),
    (0, "edit", "-x*x+2*x-17", "near", "fixed the constant halfway"),
    (0, "edit", "-x*x+2*x-16-2", "equiv", "patched by appending -2"),
    # 1 x*x+4*x-2   draft 4*x-2
    (1, "final", "x*(x+4)-2", "equiv", "factored"),
    (1, "final", "(x+2)*(x+2)-6", "equiv", "completed the square"),
    (1, "final", "x*x+4*x+2", "near", "sign of the constant"),
    (1, "final", "5*x-2", "early", "line through the first two examples"),
    (1, "edit", "4*x-2+x*x", "equiv", "appended the missing term"),
    (1, "edit", "4*x-2", "useless", "draft unchanged"),
    # 2 -x*x-3*x+7   draft -3*x+7
    (2, "final", "7-3*x-x*x", "equiv", "constant first"),
    (2, "final", "-(x*x+3*x-7)", "equiv", "negated bracket"),
    (2, "final", "x=0?7:x=1?3:-3", "near", "hard-codes the examples"),
    (2, "edit", "-3*x+7-x*x", "equiv", "appended the missing term"),
    (2, "edit", "-3*x+7-x", "near", "appended x instead of x*x"),
    # 3 2*x*x-8*x+19   draft 2*x*x-7*x+19
    (3, "final", "2*(x*x-4*x)+19", "equiv", "factored the 2"),
    (3, "final", "2*x*x-8*x+18", "near", "constant off by one"),
    (3, "final", "x*x*2-x*8+19", "equiv", "coefficients after"),
    (3, "edit", "2*x*x-6*x+19", "near", "moved the coefficient the wrong way"),
    (3, "edit", "2*x*x-7*x+19-x", "equiv", "patched by appending -x"),
    # 4 2*x*x+9*x+19   draft 2*x*x+9*x+21
    (4, "edit", "2*x*x+9*x+20", "near", "fixed the constant halfway"),
    (4, "edit", "2*x*x+9*x+21-2", "equiv", "patched by appending -2"),
    (4, "final", "x*(2*x+9)+19", "equiv", "Horner"),
    (4, "final", "19+9*x+2*x*x", "equiv", "ascending order"),
    # 5 -x*x-4*x-16   draft lookup
    (5, "edit", "x=0?-16:x=1?-21:x=2?-28:0", "near", "extends the table"),
    (5, "edit", "-(x+2)*(x+2)-12", "equiv", "completed the square"),
    (5, "final", "-x*x-4*x+16", "near", "sign of the constant"),
    # 6 2*x*x+5*x+12   draft 5*x+12
    (6, "edit", "x*x+5*x+12", "near", "added x*x without the 2"),
    (6, "edit", "5*x+12+2*x*x", "equiv", "appended the missing term"),
    # 7 9*x-1   draft 9*x-2
    (7, "edit", "x*9-1", "equiv", "coefficient after"),
    (7, "edit", "9*x-1 ", "equiv", "trailing space"),
    (7, "edit", "9*x-3", "near", "moved the constant the wrong way"),
    (7, "final", "10*x-x-1", "equiv", "odd but equal"),
    (7, "final", "9*x+1", "near", "sign of the constant"),
    # 8 -x*x+9*x-7   draft -x*x+9*x-6
    (8, "edit", "-x*x+9*x-8", "near", "overshot the constant"),
    (8, "edit", "9*x-x*x-7", "equiv", "terms reordered"),
    # 9 2*x+1   draft lookup
    (9, "edit", "x+x+1", "equiv", "repeated addition"),
    (9, "edit", "x*2+1", "equiv", "coefficient after"),
    (9, "edit", "x=0?1:x=1?3:x=2?5:7", "near", "extends the table"),
    (9, "final", "(x+1)*2-1", "equiv", "rearranged"),
    (9, "final", "2*x-1", "near", "sign of the constant"),
    # 10 2*x*x+4*x-10   draft 2*x*x+4*x-12
    (10, "edit", "2*(x*x+2*x-5)", "equiv", "factored the 2"),
    (10, "edit", "2*x*x+4*x-11", "near", "fixed the constant halfway"),
    # 11 -x*x-16   draft -x*x-15
    (11, "edit", "-x*x-17", "near", "moved the constant the wrong way"),
    (11, "edit", "-16-x*x", "equiv", "constant first"),
    (11, "edit", "0-x*x-16", "equiv", "explicit zero"),
    # 13 x*x+9*x-20
    (13, "final", "(x+10)*x-20", "equiv", "factored"),
    (13, "final", "x*x+9*x+20", "near", "sign of the constant"),
    # 14 -6*x-9   draft -6*x-7
    (14, "edit", "-6*x-8", "near", "fixed the constant halfway"),
    (14, "edit", "-(6*x+9)", "equiv", "negated bracket"),
    (14, "edit", "-3*(2*x+3)", "equiv", "factored"),
    # 15 -x*x-4*x+16
    (15, "final", "16-4*x-x*x", "equiv", "constant first"),
    (15, "final", "x*x-4*x+16", "near", "dropped the leading minus"),
    # 16 x*x+8*x+16   draft x*x+9*x+16
    (16, "edit", "(x+4)*(x+4)", "equiv", "perfect square"),
    (16, "edit", "x*x+8*x+15", "near", "changed the wrong coefficient"),
    # 17 -x*x-2*x-5   draft -x*x-x-5
    (17, "edit", "-x*x-x-x-5", "equiv", "added another -x"),
    (17, "edit", "-(x+1)*(x+1)-4", "equiv", "completed the square"),
    # 18 2*x*x-5*x-16   draft 2*x*x-5*x-14
    (18, "edit", "2*x*x-5*x-14-2", "equiv", "patched by appending -2"),
    (18, "edit", "2*x*x-5*x-15", "near", "fixed the constant halfway"),
    # 20 6*x+19   draft 5*x+19
    (20, "edit", "x*6+19", "equiv", "coefficient after"),
    (20, "edit", "5*x+19+x", "equiv", "patched by appending +x"),
    (20, "edit", "7*x+19", "near", "overshot the coefficient"),
    # 21 -7*x+2   draft lookup
    (21, "final", "2-7*x", "equiv", "constant first"),
    (21, "final", "x=0?2:x=1?-5:-12", "near", "hard-codes the examples"),
    (21, "final", "-7*x-2", "near", "sign of the constant"),
]


def tool_records():
    rng = random.Random(TOOL_SEED)
    probs = [core.TOOL.generate(rng) for _ in range(14)]
    out = []
    for i, t, cand, cat, note in TOOL:
        p = probs[i]
        out.append({"family": "tool_use", "problem": p, "pid": f"tu{TOOL_SEED}-{i}", "t": t,
                    "prefix": p["turns"][:t], "expert": p["turns"][t], "cand": cand,
                    "category": cat, "source": "handwritten", "note": note})
    return out


def code_records():
    rng, drng = random.Random(CODE_SEED), random.Random(CODE_SEED + 7)
    probs, drafts = [], []
    for _ in range(22):
        p = core.CODE.generate(rng)
        probs.append(p)
        drafts.append(make_draft(p, drng))
    out = []
    for i, kind, cand, cat, note in CODE:
        p = probs[i]
        out.append({"family": "code_exec", "problem": p, "pid": f"ce{CODE_SEED}-{i}-{kind}", "pivot": kind,
                    "draft": drafts[i] if kind == "edit" else None, "expert": p["gold"], "cand": cand,
                    "category": cat, "source": "handwritten", "note": note})
    return out
