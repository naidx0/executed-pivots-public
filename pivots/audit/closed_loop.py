"""Closed-loop executed outcome (plan §7.5's E_h): after the candidate action, a
continuation policy keeps working *from the state the candidate left*, seeing
its own terminal output, until it claims task_complete or hits the horizon; then
the task's verifier runs. This replaces open-loop splicing of the expert's
remaining turns, which passes almost any harmless action.

The continuation policy is external (GLM-5.1 in the plan; any agent here). This
module keeps episodes on disk so a policy in another process can drive them:

    python -m pivots.audit.closed_loop init   --audit out/audit --episodes out/eps --select decisive
    python -m pivots.audit.closed_loop show   out/eps/<id>.json          # transcript to continue from
    python -m pivots.audit.closed_loop step   out/eps/<id>.json action.json
    python -m pivots.audit.closed_loop summary --audit out/audit --episodes out/eps

An episode ends when an action sets task_complete, or after `horizon` policy
steps (the expert's remaining turn count plus slack); the verifier then runs on
the final state and the episode records E_closed.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
from pathlib import Path

from cleave.world.overlay import OverlayWorld
from pivots import specimen as S
from pivots.effect.terminus import parse_action
from pivots.replay.step import render, step_script

SLACK = 3
MAX_OBS = 6000


def _world(store: str, workdir: str):
    return OverlayWorld(store=store, workdir=workdir)


def _run_action(world, checkpoint: str, sp, text: str) -> tuple[str, str, bool, bool]:
    """Run one Terminus action from `checkpoint`; return (new checkpoint, screen, task_complete, parsed)."""
    pa = parse_action(text)
    if pa.action is None:
        return checkpoint, "(the previous response could not be parsed as the required JSON; nothing was run)\n", False, False
    f = world.fork(checkpoint)
    prog, _ = step_script(pa.action.commands, host=sp.host, default_cwd=sp.workdir)
    op = f.run(["bash", "-c", prog], cwd="/", timeout_s=60 * max(1, len(pa.action.commands)) + 60)
    screen = render(op.stdout)
    if len(screen) > MAX_OBS:
        screen = "[... output truncated ...]\n" + screen[-MAX_OBS:]
    return f.checkpoint(), screen, bool(pa.action.task_complete), True


def _labels_dir(episodes: str) -> str:
    return str(Path(episodes).resolve()) + "-labels"


def _finish(ep: dict, world, sp) -> None:
    v = S.verify(world, ep["checkpoint"], sp)
    ep["done"] = True
    ep["E_closed"] = int(v["reward"] == 1)


def init(audit: str, episodes: str, store: str, select: str, pools: list[str] | None = None,
         controls: tuple[str, ...] = ("ctrl:expert", "ctrl:no_op")) -> int:
    rows = [json.loads(line) for line in open(Path(audit) / "rows.jsonl")]
    decisive = {(r["task"], r["turn"]) for r in rows if r["action"] == "ctrl:no_op" and r["E"] == 0}
    keep = []
    for r in rows:
        if select == "decisive" and (r["task"], r["turn"]) not in decisive:
            continue
        is_student = r["action"].startswith("student:")
        if is_student and pools and r["action"].split(":")[1] not in pools:
            continue
        if not is_student and r["action"] not in controls:
            continue
        keep.append(r)
    os.makedirs(episodes, exist_ok=True)
    os.makedirs(_labels_dir(episodes), exist_ok=True)
    specs: dict[str, tuple] = {}
    n = 0
    for r in keep:
        sp_path = os.path.join("specimens", r["task"])
        if r["task"] not in specs:
            sp = S.load(sp_path)
            w = S.build(functools.partial(_world, store), sp)
            rr = S.run_expert(w, sp)
            specs[r["task"]] = (sp, w, rr, [s.rendered for s in rr.steps])
        sp, w, rr, obs = specs[r["task"]]
        eid = hashlib.sha1(f'{r["task"]}|{r["turn"]}|{r["action"]}|{r["text"]}'.encode()).hexdigest()[:12]
        msgs = S.prompt_for(sp, r["turn"], obs, f"root@{sp.host}:{sp.workdir}# ")
        cp, screen, done, _ = _run_action(w, rr.anchor(r["turn"]), sp, r["text"])
        msgs += [{"role": "assistant", "content": r["text"]}, {"role": "user", "content": "New Terminal Output:\n" + screen}]
        ep = {"id": eid, "task": r["task"], "turn": r["turn"], "spec": os.path.abspath(sp_path),
              "store": store,
              "checkpoint": cp, "messages": msgs, "policy_steps": 0,
              "horizon": (r["turns"] - 1 - r["turn"]) + SLACK, "done": False, "E_closed": None}
        # labels live apart from the episode so the continuation policy never sees them
        Path(_labels_dir(episodes), f"{eid}.json").write_text(json.dumps({"action": r["action"], "J": r["J"], "X": r["X"], "E_open": r["E"]}))
        if done:
            _finish(ep, w, sp)
        Path(episodes, f"{eid}.json").write_text(json.dumps(ep))
        n += 1
    return n


def show(path: str) -> str:
    ep = json.loads(Path(path).read_text())
    if ep["done"]:
        return f"episode {ep['id']} is finished (E_closed={ep['E_closed']})"
    first = ep["messages"][0]["content"]
    task = first.split("Task Description:\n", 1)[-1]
    out = [f"episode {ep['id']}  step {ep['policy_steps'] + 1} of at most {ep['horizon']}",
           "=== TASK (Terminus-2 preamble omitted; answer in its JSON format) ===", task]
    for m in ep["messages"][1:]:
        out.append(f"=== {'YOU (earlier response)' if m['role'] == 'assistant' else 'TERMINAL'} ===")
        out.append(m["content"])
    return "\n".join(out)


def step(path: str, action_text: str) -> str:
    ep = json.loads(Path(path).read_text())
    if ep["done"]:
        return f"episode already finished (E_closed={ep['E_closed']})"
    sp = S.load(ep["spec"])
    w = _world(ep["store"], sp.workdir)
    cp, screen, complete, parsed = _run_action(w, ep["checkpoint"], sp, action_text)
    ep["checkpoint"] = cp
    ep["policy_steps"] += 1
    ep["messages"] += [{"role": "assistant", "content": action_text},
                       {"role": "user", "content": "New Terminal Output:\n" + screen}]
    if complete or ep["policy_steps"] >= ep["horizon"]:
        ep["ended_by"] = "task_complete" if complete else "horizon"
        _finish(ep, w, sp)
    Path(path).write_text(json.dumps(ep))
    tail = f"\n[episode finished: {ep['ended_by']}]" if ep["done"] else ""
    return screen + tail


def summary(audit: str, episodes: str) -> dict:
    from pivots.audit.run import compare
    eps = [json.loads(p.read_text()) for p in Path(episodes).glob("*.json")]
    for e in eps:
        e.update(json.loads(Path(_labels_dir(episodes), f"{e['id']}.json").read_text()))
    done = [e for e in eps if e["done"]]
    out = {"episodes": len(eps), "finished": len(done)}
    # progress: the task still gets done, and the continuation needs no more steps than after the
    # expert's own action at the same pivot (E_closed alone asks only "is the task still solvable?")
    expert_steps = {(e["task"], e["turn"]): e["policy_steps"] for e in done if e["action"] == "ctrl:expert"}
    for e in done:
        ref = expert_steps.get((e["task"], e["turn"]))
        e["E_prog"] = int(bool(e["E_closed"]) and ref is not None and e["policy_steps"] <= ref)
    groups = {"expert": [e for e in done if e["action"] == "ctrl:expert"],
              "no_op": [e for e in done if e["action"] == "ctrl:no_op"]}
    for pool in sorted({e["action"].split(":")[1] for e in done if e["action"].startswith("student:")}):
        groups[f"student:{pool}"] = [e for e in done if e["action"].startswith(f"student:{pool}:")]
    for g, sub in groups.items():
        if not sub:
            continue
        rows = [{"J": e["J"], "X": e["X"], "E": e["E_closed"]} for e in sub]
        prog = [{"J": e["J"], "X": e["X"], "E": e["E_prog"]} for e in sub]
        out[g] = {"n": len(sub), "E_closed_rate": round(sum(e["E_closed"] for e in sub) / len(sub), 4),
                  "E_open_rate": round(sum(e["E_open"] for e in sub) / len(sub), 4),
                  "open_vs_closed_agree": round(sum(e["E_open"] == e["E_closed"] for e in sub) / len(sub), 4),
                  "E_prog_rate": round(sum(e["E_prog"] for e in sub) / len(sub), 4),
                  "mean_policy_steps": round(sum(e["policy_steps"] for e in sub) / len(sub), 2),
                  "J": compare(rows, "J"), "X": compare(rows, "X"),
                  "J_vs_prog": compare(prog, "J"), "X_vs_prog": compare(prog, "X")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init")
    a.add_argument("--audit", required=True)
    a.add_argument("--episodes", required=True)
    a.add_argument("--store", default="/tmp/cleave-closed-loop")
    a.add_argument("--select", default="decisive", choices=["decisive", "all"])
    a.add_argument("--pools", nargs="*")
    b = sub.add_parser("show")
    b.add_argument("episode")
    c = sub.add_parser("step")
    c.add_argument("episode")
    c.add_argument("action_file", help="file holding the Terminus-2 JSON response, or - for stdin")
    d = sub.add_parser("summary")
    d.add_argument("--audit", required=True)
    d.add_argument("--episodes", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "init":
        print(init(args.audit, args.episodes, args.store, args.select, args.pools), "episodes")
    elif args.cmd == "show":
        print(show(args.episode))
    elif args.cmd == "step":
        text = sys.stdin.read() if args.action_file == "-" else Path(args.action_file).read_text()
        print(step(args.episode, text))
    else:
        print(json.dumps(summary(args.audit, args.episodes), indent=2))


if __name__ == "__main__":
    main()
