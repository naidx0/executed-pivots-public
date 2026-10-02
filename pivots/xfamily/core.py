"""J, X and the oracle for tool_use and code_exec pivots (H16).

A pivot is (problem, pre-state, expert step). A candidate is one step taken from
that pre-state instead of the expert's.

  tool_use   pre-state = the gold turns[:t] already run; expert step = turns[t].
             A step is one turn: a tool call "!ab" / "!17+42", or the final answer.
  code_exec  single-turn episodes. pivot "final": pre-state is the prompt, the step
             is the whole program. pivot "edit": pre-state is prompt + a wrong draft
             program, the step revises it into a full program. The expert step is the
             gold program in both cases.

J (baseline, the shipped string reward): SequenceMatcher(expert_text, cand_text).ratio() >= 0.9.

X (executed effect, in the spirit of pivots/effect): fork the pre-state, run the
expert step and the candidate step from it, compare effects, never text.
  tool_use   effect of a call = what the environment appended, as a fact:
               ("read", key, value) for a table lookup, ("calc", value) for the
               calculator, ("error",) when the tool answered "?".
             effect of a final turn = ("answer", int) or ("answer-invalid",).
             X = 1 iff effect(candidate) == effect(expert) and it is not an error.
  code_exec  effect = the program's outputs on the probe inputs PROBES (-10..10),
             or an error. X = 1 iff the outputs are a list equal to gold's.
Horizon 0: X looks at the expert's next step only, not the rest of its trajectory.

Oracle (ground truth, defined before labelling):
  tool_use   run the candidate from the pre-state. If it ends the episode, the
             oracle is verify(transcript). Otherwise the gold policy expert_next()
             finishes the episode from the candidate's state within the kit's
             max_turns budget and verify() scores it. "Working" = the episode
             still succeeds AND the step made progress, where progress means the
             gold policy's remaining number of turns went down:
             remaining(after) < remaining(before). A final answer that verifies
             is working (remaining goes to 0).
  code_exec  the candidate's program is the final output; the gold policy has
             nothing left to do. Working = verify(program) == 1 (visible + hidden
             tests -3..9). Progress holds automatically: every pre-state (no
             program, or a draft) fails verify.
  A false credit is credit = 1 while the oracle says not working.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from .envs import _INT, CodeExec, ToolUse, run_program

TOOL = ToolUse()
CODE = CodeExec()
PROBES = list(range(-10, 11))
J_THRESHOLD = 0.9


# --------------------------------------------------------------------------- J

def j_similarity(expert: str, cand: str) -> float:
    return SequenceMatcher(None, expert, cand).ratio()


def j_credit(expert: str, cand: str) -> int:
    return int(j_similarity(expert, cand) >= J_THRESHOLD)


# --------------------------------------------------------------------------- tool_use state

def run_turns(p, turns, start=0):
    """Run turns from a fresh episode. Returns (events, transcript, done)."""
    ctx, transcript, events, done = p["prompt"], "", [], False
    for i, text in enumerate(turns):
        t = start + i
        ctx, piece, done = TOOL.step(p, ctx, text, t == TOOL.max_turns - 1)
        transcript += piece
        res = None if done else TOOL.call_tool(p, text[1:])
        events.append((text, res, done))
        if done:
            break
    return events, transcript, done


def _knowledge(p, events):
    inv = TOOL.involved(p)
    known, calc = {}, set()
    for text, res, done in events:
        if done or res is None or res == "?":
            continue
        arg = text[1:].strip()
        if arg in p["state"]:
            if arg in inv:
                known[arg] = int(res)
        else:
            calc.add(int(res))
    return known, calc


def _need(p, known):
    """(value the task asks for, canonical calculator call or None). Needs all keys known."""
    inv = TOOL.involved(p)
    v = [known[k] for k in inv]
    kind = p["kind"]
    if kind == "get":
        return v[0], None
    if kind == "max":
        return max(v[0], v[1]), None
    if kind == "add":
        return v[0] + v[1], f"!{v[0]}+{v[1]}"
    if kind == "sub":
        return v[0] - v[1], f"!{v[0]}-{v[1]}"
    if kind == "add3":
        return v[0] + v[1] + v[2], f"!{v[0]}+{v[1]}+{v[2]}"
    n = int(re.findall("[0-9]+", p["prompt"])[0])
    return v[0] * n, f"!{v[0]}*{n}"


def expert_next(p, events):
    """The gold policy from any state: read missing involved keys in prompt order, then the
    calculator unless a calculator result already equals the needed value, then answer."""
    known, calc = _knowledge(p, events)
    for k in dict.fromkeys(TOOL.involved(p)):
        if k not in known:
            return "!" + k
    need, call = _need(p, known)
    if call is not None and need not in calc:
        return call
    return str(need)


def remaining(p, events):
    """Number of turns the gold policy needs to finish from this state."""
    known, calc = _knowledge(p, events)
    missing = len([k for k in dict.fromkeys(TOOL.involved(p)) if k not in known])
    if missing:
        return missing + (0 if p["kind"] in ("get", "max") else 1) + 1
    need, call = _need(p, known)
    return (1 if call is not None and need not in calc else 0) + 1


def tool_oracle(p, prefix, cand):
    t = len(prefix)
    events, transcript, done = run_turns(p, list(prefix))
    assert not done, "pivot pre-state must be mid-episode"
    before = remaining(p, events)
    ev, tr, d = run_turns(p, [cand], start=t)
    events, transcript = events + ev, transcript + tr
    if d:
        ok = TOOL.verify(p, transcript) == 1.0
        return {"success": ok, "progress": ok, "working": int(ok), "remaining_before": before,
                "remaining_after": 0, "terminal": True}
    after = remaining(p, events)
    turn = t + 1
    while turn < TOOL.max_turns:
        nxt = expert_next(p, events)
        ev, tr, d = run_turns(p, [nxt], start=turn)
        events, transcript, turn = events + ev, transcript + tr, turn + 1
        if d:
            break
    ok = TOOL.verify(p, transcript) == 1.0
    prog = after < before
    return {"success": ok, "progress": prog, "working": int(ok and prog), "remaining_before": before,
            "remaining_after": after, "terminal": False}


# --------------------------------------------------------------------------- tool_use X

def tool_effect(p, prefix, step):
    """Fork the pre-state, run one step, return the fact it added."""
    events, _, _ = run_turns(p, list(prefix))
    ev, _, done = run_turns(p, [step], start=len(prefix))
    text, res, done = ev[0]
    if done:
        fa = TOOL.final_answer(text)
        return ("answer", int(fa)) if _INT.fullmatch(fa) else ("answer-invalid",)
    if res == "?":
        return ("error",)
    arg = text[1:].strip()
    if arg in p["state"]:
        return ("read", arg, int(res))
    return ("calc", int(res))


def tool_x(p, prefix, expert, cand):
    e, c = tool_effect(p, prefix, expert), tool_effect(p, prefix, cand)
    return int(c == e and c[0] not in ("error", "answer-invalid")), {"expert_effect": list(e), "cand_effect": list(c)}


# --------------------------------------------------------------------------- code_exec

def code_effect(prog: str):
    prog = prog.strip()
    if not prog:
        return "error: empty"
    return run_program(prog, PROBES)


def code_x(gold, cand):
    e, c = code_effect(gold), code_effect(cand)
    return int(isinstance(c, list) and c == e), {"cand_effect": c if isinstance(c, str) else None}


def code_oracle(p, draft, cand):
    if draft is not None:
        assert CODE.verify(p, draft) == 0.0, "edit pivot drafts must fail verify"
    ok = CODE.verify(p, cand) == 1.0
    return {"success": ok, "progress": ok, "working": int(ok)}


# --------------------------------------------------------------------------- dispatch

def label_j(rec):
    return j_credit(rec["expert"], rec["cand"])


def label_x(rec):
    if rec["family"] == "tool_use":
        return tool_x(rec["problem"], rec["prefix"], rec["expert"], rec["cand"])
    return code_x(rec["expert"], rec["cand"])


def label_oracle(rec):
    if rec["family"] == "tool_use":
        return tool_oracle(rec["problem"], rec["prefix"], rec["cand"])
    return code_oracle(rec["problem"], rec.get("draft"), rec["cand"])
