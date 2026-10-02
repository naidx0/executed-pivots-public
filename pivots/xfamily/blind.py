"""H16 blind re-check: fresh student candidates from a second author.

Written before scoring, without reading how X or J score a candidate (core.py scoring and
the handwritten.py candidate list were not read). Only envs.py (task format, tool and verify
semantics) and tinylang's grammar were read. Fresh problems: tool_use seed 16301,
code_exec seed 16302 (edit drafts from candidates.make_draft with seed 16302+7). No gold
copies. Categories are the author's intent, not a label:
  equiv    equivalent rewrite       near     near-miss wrong
  useless  changes nothing useful   early    answer before the work is done
  reorder  a reordered step (another needed key first; terms in another order)
Pre-registration: the H16 blind pre-registration (RESEARCH.md).

tool_use rows: (problem index, turn index t, candidate turn, category, note)
code_exec rows: (problem index, "final" | "edit", candidate program, category, note)
"""
from __future__ import annotations

import random

from . import core
from .candidates import make_draft

TOOL_SEED, CODE_SEED = 16301, 16302
TOOL_PROBLEMS, CODE_PROBLEMS = 16, 22

TOOL = [
    # 0 ly+pp? ly=25 pp=24: !ly !pp !25+24 49
    (0, 0, "!pp", "reorder", "reads the second key first"),
    (0, 0, "!ly ", "equiv", "trailing space"),
    (0, 0, "!kk", "useless", "irrelevant key"),
    (0, 1, "!ly", "useless", "re-reads ly"),
    (0, 1, "!PP", "near", "upper-case key"),
    (0, 2, "!24+25", "equiv", "commuted"),
    (0, 2, "49", "early", "mental arithmetic, skips the calculator"),
    (0, 2, "!25+42", "near", "transposed digits"),
    (0, 3, "49 ", "equiv", "trailing space"),
    (0, 3, "The answer is 49", "near", "prose answer"),
    # 1 ge-jq? ge=33 jq=63: !ge !jq !33-63 -30
    (1, 0, "!jq", "reorder", "reads the second key first"),
    (1, 1, "!jg", "near", "typo key"),
    (1, 2, "!33 - 63", "equiv", "spaces around the operator"),
    (1, 2, "!63-33", "near", "operands swapped in a subtraction"),
    (1, 2, "!0-30", "equiv", "difference precomputed, then checked"),
    (1, 2, "-30", "early", "mental arithmetic"),
    (1, 3, "-030", "equiv", "leading zero"),
    (1, 3, "30", "near", "dropped the sign"),
    (1, 3, "!-30", "near", "calls the tool instead of answering"),
    # 2 fb>xq? fb=75 xq=89: !fb !xq 89
    (2, 0, "!xq", "reorder", "reads the second key first"),
    (2, 0, "!fb+xq", "near", "keys inside the calculator"),
    (2, 1, "75", "early", "answers before reading xq"),
    (2, 1, "!su", "useless", "irrelevant key"),
    (2, 2, "!89", "useless", "echoes the value through the calculator"),
    (2, 2, "089", "equiv", "leading zero"),
    (2, 2, "75", "near", "min instead of max"),
    # 3 fl? fl=83: !fl 83
    (3, 0, "! fl", "equiv", "space after the bang"),
    (3, 0, "fl", "near", "forgot the bang"),
    (3, 0, "!cv", "useless", "irrelevant key"),
    (3, 1, "+83", "equiv", "explicit plus sign"),
    (3, 1, "!fl", "useless", "re-reads fl"),
    (3, 1, "fl=83", "near", "answer with the key"),
    # 4 jl-cl? jl=35 cl=41: !jl !cl !35-41 -6
    (4, 0, "!cl", "reorder", "reads the second key first"),
    (4, 1, "!ck", "near", "typo key"),
    (4, 2, "!35-41 ", "equiv", "trailing space"),
    (4, 2, "!35+-41", "equiv", "adds the negative"),
    (4, 2, "!41-35", "near", "operands swapped"),
    (4, 2, "-6", "early", "mental arithmetic"),
    (4, 3, "-6.", "near", "trailing period"),
    (4, 3, "6", "near", "dropped the sign"),
    # 5 wg>ck? wg=44 ck=39: !wg !ck 44
    (5, 1, "!wg", "useless", "re-reads wg"),
    (5, 1, "44", "early", "guesses before reading ck (happens to be right)"),
    (5, 2, "44 ", "equiv", "trailing space"),
    (5, 2, "39", "near", "min instead of max"),
    (5, 2, "!44", "useless", "echoes the value through the calculator"),
    # 6 ya+nc? ya=20 nc=43: !ya !nc !20+43 63
    (6, 0, "!nc", "reorder", "reads the second key first"),
    (6, 1, "!ve", "useless", "irrelevant key"),
    (6, 2, "!(20+43)", "equiv", "brackets"),
    (6, 2, "!20+34", "near", "transposed digits"),
    (6, 2, "!20*43", "near", "wrong operator"),
    (6, 2, "!60+3", "equiv", "regrouped sum"),
    (6, 3, "0063", "equiv", "leading zeros"),
    (6, 3, "64", "near", "off by one"),
    # 7 km+dl? km=69 dl=74: !km !dl !69+74 143
    (7, 1, "!kl", "near", "wrong key, one letter off"),
    (7, 2, "!74+69", "equiv", "commuted"),
    (7, 2, "143", "early", "mental arithmetic"),
    (7, 2, "133", "early", "mental arithmetic with a carry slip"),
    (7, 3, "143 ", "equiv", "trailing space"),
    (7, 3, "!143", "near", "calls the tool instead of answering"),
    # 8 ha+ij+da? ha=27 ij=25 da=16: !ha !ij !da !27+25+16 68
    (8, 0, "!da", "reorder", "reads the third key first"),
    (8, 1, "!da", "reorder", "reads the third key second"),
    (8, 2, "!tz", "useless", "irrelevant key"),
    (8, 2, "!27+25", "early", "sums before da is read"),
    (8, 3, "!16+25+27", "equiv", "reversed operands"),
    (8, 3, "!52+16", "equiv", "partial sum in the head"),
    (8, 3, "!27+25+61", "near", "transposed digits"),
    (8, 3, "68", "early", "mental arithmetic"),
    (8, 4, "!68", "near", "tool call on the forced-final turn"),
    (8, 4, " 68", "equiv", "leading space"),
    # 9 bh-kw? bh=65 kw=55: !bh !kw !65-55 10
    (9, 0, "!kw", "reorder", "reads the second key first"),
    (9, 2, "!65-55+0", "equiv", "adds zero"),
    (9, 2, "!55-65", "near", "operands swapped"),
    (9, 2, "10", "early", "mental arithmetic"),
    (9, 3, "-10", "near", "sign flipped"),
    (9, 3, "010", "equiv", "leading zero"),
    # 10 nq+sr? nq=60 sr=81: !nq !sr !60+81 141
    (10, 1, "!sr ", "equiv", "trailing space"),
    (10, 1, "!mi", "useless", "irrelevant key"),
    (10, 2, "!81+60", "equiv", "commuted"),
    (10, 2, "!60+18", "near", "transposed digits"),
    (10, 3, "14 1", "near", "stray space inside the number"),
    (10, 3, "142", "near", "off by one"),
    # 11 tw+wv? tw=68 wv=47: !tw !wv !68+47 115
    (11, 0, "!wv", "reorder", "reads the second key first"),
    (11, 2, "105", "early", "carry slip"),
    (11, 2, "115", "early", "mental arithmetic"),
    (11, 2, "!47+68", "equiv", "commuted"),
    (11, 3, "115;", "near", "trailing semicolon"),
    # 12 cc+xb? cc=30 xb=74: !cc !xb !30+74 104
    (12, 1, "!cc", "useless", "re-reads cc"),
    (12, 2, "!3*10+74", "equiv", "30 written as 3*10"),
    (12, 2, "!30+47", "near", "transposed digits"),
    (12, 3, "1O4", "near", "letter O for zero"),
    # 13 pr-fu? pr=20 fu=39: !pr !fu !20-39 -19
    (13, 2, "!-39+20", "equiv", "reordered with a leading negative"),
    (13, 2, "!20-(39)", "equiv", "brackets"),
    (13, 2, "-19", "early", "mental arithmetic"),
    (13, 3, "19", "near", "dropped the sign"),
    (13, 3, "-19 ", "equiv", "trailing space"),
    # 14 rz+mc+pl? rz=64 mc=65 pl=89: !rz !mc !pl !64+65+89 218
    (14, 1, "!pl", "reorder", "reads the third key second"),
    (14, 2, "!gh", "useless", "irrelevant key"),
    (14, 3, "!64+65+89 ", "equiv", "trailing space"),
    (14, 3, "!129+89", "equiv", "partial sum in the head"),
    (14, 3, "!64+56+89", "near", "transposed digits"),
    (14, 3, "208", "early", "carry slip"),
    (14, 4, "218 ", "equiv", "trailing space"),
    (14, 4, "!64+65+89", "near", "calculator on the forced-final turn"),
    # 15 di>ms? di=91 ms=67: !di !ms 91
    (15, 0, "!ms", "reorder", "reads the second key first"),
    (15, 1, "91", "early", "answers before reading ms (happens to be right)"),
    (15, 2, "67", "near", "min instead of max"),
    (15, 2, " 91", "equiv", "leading space"),
]

CODE = [
    # 0 -x*x-4*x+18 | draft -x*x-3*x+18
    (0, "final", "18-4*x-x*x", "reorder", "constant first"),
    (0, "final", "-(x*x+4*x-18)", "equiv", "negated bracket"),
    (0, "final", "22-(x+2)*(x+2)", "equiv", "completed square"),
    (0, "final", "18", "early", "constant from the first example"),
    (0, "edit", "-x*x-3*x+18-x", "equiv", "patch: appends -x"),
    (0, "edit", "-x*x-5*x+18", "near", "corrected the wrong way"),
    (0, "edit", "-x*x-3*x+18", "useless", "draft unchanged"),
    # 1 2*x*x+3*x-3 | draft 2*x*x+3*x-5
    (1, "final", "x*(2*x+3)-3", "equiv", "factored"),
    (1, "final", "5*x-3", "early", "line through the first two examples"),
    (1, "final", "2*x*x+3*x+3", "near", "sign of the constant"),
    (1, "edit", "2*x*x+3*x-5+2", "equiv", "patch: +2"),
    (1, "edit", "2*x*x+3*x-4", "near", "half-fixed constant"),
    # 2 x*x-2*x | draft x*x-2*x+1
    (2, "final", "x*(x-2)", "equiv", "factored"),
    (2, "final", "(x-1)*(x-1)-1", "equiv", "completed square"),
    (2, "final", "x*x-2", "near", "dropped the x"),
    (2, "final", "x=1?-1:0", "near", "fits the examples only"),
    (2, "edit", "(x-1)*(x-1)", "useless", "draft rewritten, same function"),
    (2, "edit", "x*x-2*x+1-1", "equiv", "patch: -1"),
    # 3 -x*x+3*x-1 | draft -x*x+3*x
    (3, "final", "3*x-x*x-1", "reorder", "linear term first"),
    (3, "final", "x*(3-x)-1", "equiv", "factored"),
    (3, "final", "-x*x+3*x+1", "near", "sign of the constant"),
    (3, "final", "x*x", "useless", "a bare square"),
    (3, "edit", "-x*x+3*x -1", "equiv", "patch appended after a space"),
    (3, "edit", "-x*x+3*x-2", "near", "wrong constant"),
    # 4 7*x+3 | draft x=0?3:x=1?10:17
    (4, "final", "x*7+3", "reorder", "commuted product"),
    (4, "final", "3+7*x", "reorder", "constant first"),
    (4, "final", "7*x+3*1", "equiv", "times one"),
    (4, "final", "7*x+3+x*(x-1)*(x-2)", "near", "agrees on the examples only"),
    (4, "edit", "x=0?3:x=1?10:x=2?17:7*x+3", "equiv", "keeps the table, adds the rule"),
    (4, "edit", "x=0?3:x=1?10:x=2?17:24", "near", "extends the table by one"),
    (4, "edit", "(x=0?3:x=1?10:17)", "useless", "brackets the table"),
    # 5 2*x*x-3*x-13 | draft x=0?-13:x=1?-14:-11
    (5, "final", "x*(2*x-3)-13", "equiv", "factored"),
    (5, "final", "2*x*x-13-3*x", "reorder", "constant before the linear term"),
    (5, "final", "2*x*x-3*x+13", "near", "sign of the constant"),
    (5, "final", "-13", "early", "constant from the first example"),
    (5, "edit", "(2*x-3)*x-13", "equiv", "factored"),
    (5, "edit", "x*x-2*x-13", "near", "wrong fit"),
    # 6 2*x*x+2*x+20 | draft 2*x+20
    (6, "final", "2*x*(x+1)+20", "equiv", "factored"),
    (6, "final", "2*(x*x+x+10)", "equiv", "common factor"),
    (6, "final", "4*x+20", "early", "line through the first two examples"),
    (6, "edit", "2*x+20+2*x*x", "equiv", "appends the square term"),
    (6, "edit", "2*x+20+x*x", "near", "square term without its coefficient"),
    (6, "edit", "2*x + 20", "useless", "reformatted draft"),
    # 7 -x*x-4*x+6 | draft -4*x+6
    (7, "final", "6-4*x-x*x", "reorder", "constant first"),
    (7, "final", "-x*x-4*x+6 ", "equiv", "trailing space"),
    (7, "final", "(x*x+4*x-6)/-1", "equiv", "divides by -1"),
    (7, "final", "x*x-4*x+6", "near", "sign of the square"),
    (7, "edit", "-4*x+6-x*x", "equiv", "appends the square term"),
    (7, "edit", "-4*x+6-x", "near", "appends x instead of x*x"),
    # 8 -x*x-5*x+6 | draft -5*x+6
    (8, "final", "-(x+6)*(x-1)", "equiv", "factored roots"),
    (8, "final", "(1-x)*(x+6)", "equiv", "factored roots"),
    (8, "final", "-x*x-5*x-6", "near", "sign of the constant"),
    (8, "edit", "-5*x+6-x*x", "equiv", "appends the square term"),
    (8, "edit", "-5*x+6", "useless", "draft unchanged"),
    # 9 8*x-10 | draft x=0?-10:x=1?-2:6
    (9, "final", "2*(4*x-5)", "equiv", "common factor"),
    (9, "final", "-10+8*x", "reorder", "constant first"),
    (9, "final", "8*x+10", "near", "sign of the constant"),
    (9, "final", "x*8-1", "near", "truncated constant"),
    (9, "edit", "8*x - 10", "equiv", "replaces the table, spaced"),
    (9, "edit", "x=0?-10:x=1?-2:x=2?6:14", "near", "extends the table by one"),
    # 10 x*x-6*x-6 | draft x*x-5*x-6
    (10, "final", "(x-3)*(x-3)-15", "equiv", "completed square"),
    (10, "final", "x*(x-6)-6", "equiv", "factored"),
    (10, "final", "x*x-6*x+6", "near", "sign of the constant"),
    (10, "final", "x", "useless", "identity"),
    (10, "edit", "x*x-5*x-6-x", "equiv", "patch: -x"),
    (10, "edit", "x*x-7*x-6", "near", "corrected the wrong way"),
    # 11 -x*x+8*x+10 | draft x=0?10:x=1?17:22
    (11, "final", "26-(x-4)*(x-4)", "equiv", "completed square"),
    (11, "final", "8*x+10-x*x", "reorder", "square term last"),
    (11, "final", "7*x+10", "early", "line through the first two examples"),
    (11, "edit", "x*(8-x)+10", "equiv", "factored"),
    (11, "edit", "x=0?10:x=1?17:22", "useless", "draft unchanged"),
    # 12 2*x*x+5*x+20 | draft 2*x*x+5*x+18
    (12, "final", "x*(2*x+5)+20", "equiv", "factored"),
    (12, "final", "2*x*x+5*x+2", "near", "dropped a digit"),
    (12, "edit", "2*x*x+5*x+18+2", "equiv", "patch: +2"),
    (12, "edit", "2*x*x+5*x+19", "near", "half-fixed constant"),
    # 13 x*x+3*x+8 | draft 3*x+8
    (13, "final", "x*(x+3)+8", "equiv", "factored"),
    (13, "final", "4*x+8", "early", "line through the first two examples"),
    (13, "final", "x=0?8:x=1?12:18", "near", "hard-codes the examples"),
    (13, "edit", "3*x+8+x*x", "equiv", "appends the square term"),
    (13, "edit", "3*x+8+x", "near", "appends x instead of x*x"),
    # 14 x*x-2*x-9 | draft x*x-2*x-10
    (14, "final", "(x-1)*(x-1)-10", "equiv", "completed square"),
    (14, "final", "x*x-2*x+9", "near", "sign of the constant"),
    (14, "edit", "x*x-2*x-10+1", "equiv", "patch: +1"),
    (14, "edit", "x*x-2*x-11", "near", "corrected the wrong way"),
    (14, "edit", "x*x-2*x-10", "useless", "draft unchanged"),
    # 15 2*x*x-5*x-17 | draft x=0?-17:x=1?-20:-19
    (15, "final", "x*(2*x-5)-17", "equiv", "factored"),
    (15, "final", "-17-5*x+2*x*x", "reorder", "terms reversed"),
    (15, "final", "-3*x-17", "early", "line through the first two examples"),
    (15, "edit", "(2*x-5)*x-17", "equiv", "factored"),
    (15, "edit", "2*x*x-5*x-7", "near", "dropped a digit"),
    # 16 7*x+16 | draft 7*x+17
    (16, "final", "16+7*x", "reorder", "constant first"),
    (16, "final", "7*(x+2)+2", "equiv", "regrouped"),
    (16, "final", "(14*x+32)/2", "equiv", "doubled then halved"),
    (16, "final", "7*x+61", "near", "transposed digits"),
    (16, "edit", "7*x+17-1", "equiv", "patch: -1"),
    (16, "edit", "7*x+15", "near", "corrected the wrong way"),
    (16, "edit", "7*x+17", "useless", "draft unchanged"),
    # 17 x*x-4*x-11 | draft x*x-4*x-10
    (17, "final", "(x-2)*(x-2)-15", "equiv", "completed square"),
    (17, "final", "x*x-4*x-1", "near", "dropped a digit"),
    (17, "final", "0", "useless", "constant zero"),
    (17, "edit", "x*x-4*x-10-1", "equiv", "patch: -1"),
    (17, "edit", "x*x-4*x-9", "near", "corrected the wrong way"),
    # 18 2*x*x-9*x-3 | draft x=0?-3:x=1?-10:-13
    (18, "final", "x*(2*x-9)-3", "equiv", "factored"),
    (18, "final", "2*x*x-9*x+3", "near", "sign of the constant"),
    (18, "final", "-7*x-3", "early", "line through the first two examples"),
    (18, "edit", "-3-9*x+2*x*x", "reorder", "terms reversed"),
    (18, "edit", "x=0?-3:x=1?-10:x=2?-13:-12", "near", "extends the table by one"),
    # 19 -x*x-x-3 | draft -x-3
    (19, "final", "-(x*x+x+3)", "equiv", "negated bracket"),
    (19, "final", "-x*x-x+3", "near", "sign of the constant"),
    (19, "edit", "-x-3-x*x", "equiv", "appends the square term"),
    (19, "edit", "-x-3", "useless", "draft unchanged"),
    (19, "edit", "-x-3-x", "near", "appends x instead of x*x"),
    # 20 x*x+9*x+7 | draft x=0?7:x=1?17:29
    (20, "final", "x*(x+9)+7", "equiv", "factored"),
    (20, "final", "7+9*x+x*x", "reorder", "terms reversed"),
    (20, "final", "10*x+7", "early", "line through the first two examples"),
    (20, "edit", "x*x+9*x + 7", "equiv", "replaces the table, spaced"),
    (20, "edit", "x*x+8*x+7", "near", "one-off coefficient"),
    # 21 -x*x+3*x+9 | draft -x*x+3*x+8
    (21, "final", "9+3*x-x*x", "reorder", "terms reversed"),
    (21, "final", "x*(3-x)+9", "equiv", "factored"),
    (21, "final", "x*x+3*x+9", "near", "sign of the square"),
    (21, "edit", "-x*x+3*x+8+1", "equiv", "patch: +1"),
    (21, "edit", "-x*x+3*x+10", "near", "corrected the wrong way"),
]


def tool_records():
    rng = random.Random(TOOL_SEED)
    probs = [core.TOOL.generate(rng) for _ in range(TOOL_PROBLEMS)]
    out = []
    for i, t, cand, cat, note in TOOL:
        p = probs[i]
        out.append({"family": "tool_use", "problem": p, "pid": f"tu{TOOL_SEED}-{i}", "t": t,
                    "prefix": p["turns"][:t], "expert": p["turns"][t], "cand": cand,
                    "category": cat, "source": "blind", "note": note})
    return out


def code_records():
    rng, drng = random.Random(CODE_SEED), random.Random(CODE_SEED + 7)
    probs, drafts = [], []
    for _ in range(CODE_PROBLEMS):
        p = core.CODE.generate(rng)
        probs.append(p)
        drafts.append(make_draft(p, drng))
    out = []
    for i, kind, cand, cat, note in CODE:
        p = probs[i]
        out.append({"family": "code_exec", "problem": p, "pid": f"ce{CODE_SEED}-{i}-{kind}", "pivot": kind,
                    "draft": drafts[i] if kind == "edit" else None, "expert": p["gold"], "cand": cand,
                    "category": cat, "source": "blind", "note": note})
    return out
