"""H32: run stored demo steps on Nebius Sandboxes (contree-sdk) and compare J, X, P with the local runner.

    python scripts/h32_contree_rescore.py prepare-image [--base ubuntu:24.04] [--tag xp-h32-base]
    python scripts/h32_contree_rescore.py run [--max-steps 5] [--max-ops 400] [--start 0] [--image xp-h32-base]
                                              [--ref <file.jsonl> | --ref stored]
                                              [--out <file.jsonl>]
                                              [--log <file.jsonl>] [--no-preflight] [--allow-network]

A step is one stored demo action: the pivot's candidate text scored by pivots.demo.core.DemoCore.run, with
every world a cleave.world.nebius.NebiusWorld (the specimen is built from setup.sh on the base image, the
expert is replayed as kept runs, the probe and the verifier run on disposable forks). The reference labels
are the local hardened runner's (H19, <file.jsonl>, 218 rows, `live_JXP`), or with
`--ref stored` the audit labels in pivots/demo/stored.json. Steps run in the reference file's order.

Spend guard: at most --max-steps steps (default 5) and --max-ops platform operations; any billing, quota,
402, 403 or 429 error, or 3 platform errors in a row, stops the run at once (exit 3). One JSONL row per step
in --log: wall time, operation count, reported cost, status. Before the first step a preflight (one
disposable run) checks that the image has the specimens' tools and no egress (exit 4 otherwise).

Credentials: NEBIUS_API_KEY and NEBIUS_PROJECT_ID from the environment only (exit 2 when either is
missing). Nothing written or printed holds either value (cleave.world.nebius.redact).

Exit codes: 0 done, 2 credentials missing, 3 spend guard stopped the run, 4 preflight refused the image,
5 prepare-image failed.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from cleave.world.nebius import (DEFAULT_IMAGE, MissingCredentials, NebiusWorld, SpendAbort, SpendGuard,
                                 StepCapReached, make_client, redact, require_env)
from pivots.demo.sandbox import Limits

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REF = REPO / "lane" / "h19" / "local" / "rescore-hardened.jsonl"
DEFAULT_OUT = REPO / "lane" / "h32" / "local" / "rescore-contree.jsonl"
DEFAULT_LOG = REPO / "lane" / "h32" / "local" / "steps.jsonl"
TOOLS = "bash python3 git gcc make awk sed grep timeout tar gzip"
# the Dockerfile's base: the tools the six decisive specimens use
APT = ("set -e; export DEBIAN_FRONTEND=noninteractive; apt-get update; apt-get install -y --no-install-recommends "
       "python3 python3-venv git gcc make libc6-dev mawk tar gzip util-linux procps ca-certificates; "
       "rm -rf /var/lib/apt/lists/*")
PREFLIGHT = (f'for t in {TOOLS}; do command -v "$t" >/dev/null 2>&1 || echo "MISSING $t"; done; '
             'if timeout 5 bash -c "exec 3<>/dev/tcp/1.1.1.1/53" 2>/dev/null; then echo "EGRESS open"; '
             'else echo "EGRESS closed"; fi')


def say(msg: str, *, err: bool = False) -> None:
    print(redact(msg), file=sys.stderr if err else sys.stdout, flush=True)


def build_core(a, world_factory):
    from pivots.demo.core import DemoCore
    # the overlay store is never touched with a world_factory; DemoCore only records the path
    return DemoCore(os.path.join(tempfile.gettempdir(), "h32-unused-store"), limits=Limits(batch_timeout_s=10),
                    max_concurrent=2, max_queue=6, world_factory=world_factory)


def load_ref(ref: str, core) -> list[tuple[str, str, list[int]]]:
    if ref == "stored":
        out = []
        for key in core.decisive:
            audit = core.data["audit"].get(key, {})
            for c in core.candidates(key):
                r = audit.get(c["id"])
                if r is not None:
                    out.append((key, c["id"], [r["J"], r["X"], r["P"]]))
        return out
    rows = [json.loads(x) for x in Path(ref).read_text(encoding="utf-8").splitlines() if x.strip()]
    return [(r["pivot"], r["action"], list(r["live_JXP"])) for r in rows]


def preflight(client, guard: SpendGuard, image: str, allow_network: bool) -> str | None:
    """One disposable run: the image's tools and egress. Returns a refusal reason, or None."""
    w = NebiusWorld(image, workdir="", client=client, guard=guard, run_cap_s=60, _fresh=False)
    op = w.run(["bash", "-c", PREFLIGHT], cwd="/", keep=False, timeout_s=30)
    missing = re.findall(r"^MISSING (\S+)", op.stdout, re.M)
    egress = re.search(r"^EGRESS (open|closed)", op.stdout, re.M)
    guard.write({"preflight": {"image": image, "exit": op.exit_code, "missing": missing,
                               "egress": egress.group(1) if egress else None}})
    if not egress:
        return f"preflight did not report (exit {op.exit_code}): {op.stderr[-300:]}"
    if missing:
        return f"image {image} lacks {', '.join(missing)}; run prepare-image or pass --image"
    if egress.group(1) == "open" and not allow_network:
        return ("the sandbox has egress, and the local runner has none; pass --allow-network to run anyway "
                "(an owner decision: it is logged in the summary)")
    return None


def prepare_image(a, client, guard: SpendGuard) -> int:
    guard.before_op()
    try:
        base = client.images.oci(a.base)
    except Exception as e:  # noqa: BLE001
        guard.on_error(e)
        say(f"h32: could not import {a.base}: {type(e).__name__}: {e}", err=True)
        return 5
    guard.after_op()
    ref = str(base.uuid) if getattr(base, "uuid", None) is not None else a.base
    w = NebiusWorld(ref, workdir="", client=client, guard=guard, run_cap_s=None, _fresh=False)
    t0 = time.perf_counter()
    op = w.run(["bash", "-c", APT], cwd="/", timeout_s=1800)
    guard.write({"prepare_image": {"base": a.base, "exit": op.exit_code, "wall_s": round(time.perf_counter() - t0, 3)}})
    if not op.ok:
        say(f"h32: apt-get in {a.base} failed (exit {op.exit_code}); if the sandbox has no egress, build the "
            f"image elsewhere and pass --image.\n{op.stdout[-800:]}\n{op.stderr[-800:]}", err=True)
        return 5
    guard.before_op()
    w._img.tag_as(a.tag)
    guard.after_op()
    say(f"h32: image {w.checkpoint()} tagged {a.tag}")
    return 0


def summarize(rows: list[dict], guard: SpendGuard, aborted: str | None, wall: float, allow_network: bool) -> dict:
    done = [r for r in rows if r.get("live_JXP") is not None]
    ident = sum(r["match"] for r in rows)
    s = {"n": len(rows), "identical": ident, "changed": len(rows) - ident,
         "errors": len(rows) - len(done),
         "x_false_credit": sum(r["live_JXP"][1] == 1 and r["ref_JXP"][1] == 0 for r in done),
         "x_false_miss": sum(r["live_JXP"][1] == 0 and r["ref_JXP"][1] == 1 for r in done),
         "j_changed": sum(r["live_JXP"][0] != r["ref_JXP"][0] for r in done),
         "p_changed": sum(r["live_JXP"][2] != r["ref_JXP"][2] for r in done),
         "wall_s": round(wall, 3), "wall_s_per_step": round(wall / len(rows), 3) if rows else None,
         "ops": guard.ops, "cost": round(guard.cost, 6), "aborted": aborted, "allow_network": allow_network}
    s["bars"] = ({"identical_ge_216": ident >= 216, "x_false_credit_0": s["x_false_credit"] == 0}
                 if len(rows) == 218 else None)
    return s


def run(a, client, guard: SpendGuard) -> int:
    if a.preflight:
        why = preflight(client, guard, a.image, a.allow_network)
        if why:
            say(f"h32: preflight refused: {why}", err=True)
            return 4
    factory = functools.partial(NebiusWorld, a.image, client=client, guard=guard, run_cap_s=Limits().run_cap_s)
    core = build_core(a, factory)
    steps = load_ref(a.ref, core)[a.start:a.start + a.max_steps]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    aborted = None
    texts: dict[str, dict[str, str]] = {}
    t0 = time.perf_counter()
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        for pivot, aid, ref in steps:
            row = {"pivot": pivot, "action": aid, "ref_JXP": ref}
            try:
                with guard.step(pivot=pivot, action=aid) as log:
                    if pivot not in texts:
                        texts[pivot] = {c["id"]: c["text"] for c in core.candidates(pivot)}
                    if aid not in texts[pivot]:
                        raise KeyError(f"no candidate {aid} at {pivot}")
                    res = core.run(pivot, texts[pivot][aid], with_p=True)
                    if guard.tripped:  # an abort raised in the core's X thread
                        raise SpendAbort(guard.tripped)
                    p = res["P"]
                    live = [int(res["J"]["reward"]), int(res["X"]["reward"]),
                            p["P"] if isinstance(p, dict) else p]
                    row.update(live_JXP=live, match=live == ref)
                    if isinstance(p, dict):  # H33: keep P's check counts so a changed P can be traced
                        row.update(P_detail={k: v for k, v in p.items() if k != "P"})
                    row.update(X_detail={k: res["X"].get(k) for k in ("score", "state", "output", "missing",
                                                                        "extra", "self_agreement") if k in res["X"]})
                    log.update(live_JXP=live, ref_JXP=ref, match=live == ref)
            except SpendAbort as e:
                aborted = str(e)
                break
            except StepCapReached:
                break
            except Exception as e:  # noqa: BLE001 - a failed step is a changed label, the run goes on
                if guard.tripped:
                    aborted = guard.tripped
                    break
                row.update(live_JXP=None, match=False, error=f"{type(e).__name__}: {e}"[:1000])
            rows.append(row)
            fh.write(redact(json.dumps(row)) + "\n")
            fh.flush()
            say(f"step {len(rows)}/{len(steps)} {pivot} {aid}: live {row.get('live_JXP')} ref {ref} "
                f"{'ok' if row['match'] else 'CHANGED'}")
    s = summarize(rows, guard, aborted, time.perf_counter() - t0, a.allow_network)
    out.with_suffix(".summary.json").write_text(redact(json.dumps(s, indent=1)) + "\n", encoding="utf-8",
                                                newline="\n")
    say(f"h32 contree: {s['n']} steps, {s['identical']} identical, {s['changed']} changed, "
        f"x_false_credit {s['x_false_credit']}, {s['wall_s']}s, {s['ops']} ops, cost {s['cost']} -> {out}")
    if aborted:
        say(f"h32: STOPPED by the spend guard: {aborted}", err=True)
        return 3
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("prepare-image", help="import a base image, install the specimens' tools, tag it")
    pi.add_argument("--base", default="ubuntu:24.04")
    pi.add_argument("--tag", default=DEFAULT_IMAGE)
    r = sub.add_parser("run", help="run stored demo steps and compare labels")
    r.add_argument("--max-steps", type=int, default=5)
    r.add_argument("--max-ops", type=int, default=400, help="platform operations cap (0: none)")
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--image", default=DEFAULT_IMAGE)
    r.add_argument("--ref", default=str(DEFAULT_REF), help="an H19 rescore JSONL (live_JXP), or 'stored'")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True)
    r.add_argument("--allow-network", action="store_true")
    for p in (pi, r):
        p.add_argument("--log", default=str(DEFAULT_LOG))
    a = ap.parse_args(argv)
    try:
        require_env()
    except MissingCredentials as e:
        say(f"h32: {e}", err=True)
        return 2
    guard = SpendGuard(max_steps=getattr(a, "max_steps", 0), max_ops=getattr(a, "max_ops", None) or None,
                       log_path=a.log)
    client = make_client()
    try:
        return prepare_image(a, client, guard) if a.cmd == "prepare-image" else run(a, client, guard)
    except SpendAbort as e:
        say(f"h32: STOPPED by the spend guard: {e}", err=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
