"""Build executed_pivot rows and anchors from the local specimens (overlay backend).

For every specimen: build its environment in an overlay store, replay the expert,
and emit one row per pivot (expert turn) plus the checkpoint to fork for it.

    python resources_servers/executed_pivot/scripts/prepare.py specimens/billing-invoice-bugfix \
        --store /tmp/cleave-overlay-gym
    python resources_servers/executed_pivot/scripts/prepare.py --set all --store /tmp/gym-rollouts-store

With no specimen dirs, --set picks a named set: `original` (the six the audit was built on), `dockerfile` (the
five H17/H24 specimens that ship a Dockerfile), `h41` (the five H41 Dockerfile specimens) or `all` (the sixteen,
the default). The overlay backend runs setup.sh on the machine's own root, so that root needs each specimen's
toolchain (RUNNER below). The one image with all sixteen is h24-runner:local (docker/runner.Dockerfile); a
build stops early, naming the image, when a tool is missing.

writes, next to this script's server:
    data/anchors.jsonl   {"uuid", "anchor", "cwd", "next_answer", "rest_answers"} per pivot; names checkpoints in --store (machine-local)
    data/train.jsonl     every pivot, in the terminus_judge row shape
    data/example.jsonl   the first --n-example pivots (skipped with --n-example 0, and kept
                         when it already holds the same pivots)

The store must stay where it was built (under /tmp): anchors are only valid against it.
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SERVER_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cleave.world.overlay import OverlayWorld  # noqa: E402
from pivots import specimen as S  # noqa: E402

SPECIMENS_DIR = REPO_ROOT / "specimens"
ORIGINAL_SIX = ("billing-invoice-bugfix", "sensor-site-report", "backup-cron-repair", "ledger-git-revert",
                "access-log-summary", "statcli-make-fix")
# The five that ship a Dockerfile (H17: textstats, sqlite, jq; H24: csvsum, tally). Their Dockerfiles cannot run
# inside a world, so the overlay backend runs setup.sh on a root that already has the same toolchain.
DOCKERFILE_FIVE = ("textstats-pyproject-install", "sqlite-email-dedupe", "jq-signup-rollup", "csvsum-npm-package",
                   "tally-go-build")
# H41: five more Dockerfile specimens in new families (pip constraints, shell log rotation, SQL migrations, an INI
# config plus a systemd unit, a Perl MakeMaker dist). Their toolchains are already on the H17/H20 runners.
H41_FIVE = ("reportgen-pip-constraints", "shoplog-rotate-shell", "shopdb-sql-migrations", "acme-worker-ini-unit",
            "pricelist-perl-makemaker")
# H50: Terminal-Bench "easy" tasks (Apache-2.0), ported to the overlay runner (H50, see RESEARCH.md). A second task
# family, not written by this lane; they run on h50-runner:local (h24-runner plus pytest, pandas, pyarrow and friends).
TB_TWELVE = ("tb-analyze-access-logs", "tb-bank-trans-filter", "tb-cpp-compatibility", "tb-csv-to-parquet",
             "tb-fix-git", "tb-grid-pattern-transform", "tb-heterogeneous-dates", "tb-jq-data-processing",
             "tb-jsonl-aggregator", "tb-log-summary", "tb-pandas-etl", "tb-processing-pipeline")
SETS = {"original": ORIGINAL_SIX, "dockerfile": DOCKERFILE_FIVE, "h41": H41_FIVE,
        "all": ORIGINAL_SIX + DOCKERFILE_FIVE + H41_FIVE, "tb": TB_TWELVE}
# Specimen -> (the smallest runner image whose root has its toolchain, the commands it needs on that root).
# h24-runner:local (FROM h20-runner FROM h17-runner) has all of them. backup-cron-repair also needs a writable
# /etc: in Docker, unmount /etc/resolv.conf, /etc/hosts and /etc/hostname first (H20, see RESEARCH.md); so does
# acme-worker-ini-unit (/etc/acme, /etc/systemd/system).
RUNNER = {
    "billing-invoice-bugfix": ("h17-runner:local", ("python3",)),
    "sensor-site-report": ("h17-runner:local", ("python3",)),
    "backup-cron-repair": ("h17-runner:local", ("tar", "sha256sum", "python3")),
    "access-log-summary": ("h17-runner:local", ("awk", "sort", "python3")),
    "ledger-git-revert": ("h20-runner:local", ("git", "python3")),
    "statcli-make-fix": ("h20-runner:local", ("make", "gcc", "python3")),
    "textstats-pyproject-install": ("h17-runner:local", ("python3",)),
    "sqlite-email-dedupe": ("h17-runner:local", ("python3",)),
    "jq-signup-rollup": ("h17-runner:local", ("jq", "awk", "gzip", "python3")),
    "csvsum-npm-package": ("h24-runner:local", ("node", "npm", "python3")),
    "tally-go-build": ("h24-runner:local", ("go", "make", "python3")),
    "reportgen-pip-constraints": ("h17-runner:local", ("python3",)),
    "shoplog-rotate-shell": ("h17-runner:local", ("bash", "gzip", "zcat", "python3")),
    "shopdb-sql-migrations": ("h17-runner:local", ("python3",)),
    "acme-worker-ini-unit": ("h17-runner:local", ("python3",)),
    "pricelist-perl-makemaker": ("h20-runner:local", ("perl", "prove", "make", "python3")),
    **{n: ("h50-runner:local", ("python3",)) for n in TB_TWELVE},
}


def specimen_set(name: str) -> list[Path]:
    return [SPECIMENS_DIR / n for n in SETS[name]]


def missing_tools(name: str, which=None) -> list[str]:
    """The commands specimen `name` needs that this root lacks (an unknown specimen needs nothing listed)."""
    which = which or shutil.which
    return [t for t in RUNNER.get(name, ("", ()))[1] if which(t) is None]


def _cwd_at(world: OverlayWorld, anchor: str, default: str) -> str:
    f = world.fork(anchor)
    op = f.run(["bash", "-c", "cat /var/lib/xp/cwd 2>/dev/null"], cwd="/", keep=False)
    f.close()
    return op.stdout.strip() or default


def pivot_rows(specimen_dir: str | Path, store: str, turns: list[int] | None = None) -> tuple[list[dict], list[dict]]:
    """(rows, anchors) for the pivots of one specimen; `turns` limits which (default: every expert turn)."""
    sp = S.load(specimen_dir)
    lacking = missing_tools(sp.name)
    if lacking:
        img = RUNNER[sp.name][0]
        also = "" if img == "h24-runner:local" else " or h24-runner:local (all sixteen specimens)"
        raise SystemExit(f"{sp.name} needs {', '.join(lacking)} on this root; build it inside {img}{also}, privileged")
    turns = list(range(len(sp.actions))) if turns is None else sorted(turns)
    world = S.build(functools.partial(OverlayWorld, store=store), sp)
    rr = S.run_expert(world, sp, upto=max(turns) if turns else 0)
    observations = [s.rendered for s in rr.steps]
    rows, anchors = [], []
    for t in turns:
        uuid = f"{sp.name}:{t}"
        anchor = rr.anchor(t)
        anchors.append({"uuid": uuid, "anchor": anchor, "cwd": sp.workdir if t == 0 else _cwd_at(world, anchor, sp.workdir),
                        **({"next_answer": sp.actions[t + 1], "rest_answers": sp.actions[t + 1:]}
                           if t + 1 < len(sp.actions) else {})})
        rows.append(
            {
                "responses_create_params": {
                    "input": S.prompt_for(sp, t, observations, f"root@{sp.host}:{sp.workdir}# ")
                },
                "uuid": uuid,
                "expected_answer": sp.actions[t],
                "metadata": {"harness": "terminus_2", "task": sp.name, "turn": t,
                             "runner": RUNNER.get(sp.name, ("h24-runner:local",))[0]},
            }
        )
    return rows, anchors


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def write_example(path: Path, rows: list[dict]) -> bool:
    """Write the example rows unless the file already holds the same pivots.

    example.jsonl is committed, and rebuilt prompts differ run to run in timings
    ("Ran 7 tests in 0.002s"), so rewriting it would leave a modified file after
    every prepare run.
    """
    if path.exists():
        old = [json.loads(line).get("uuid") for line in path.read_text().splitlines() if line.strip()]
        if old == [r["uuid"] for r in rows]:
            return False
    _write_jsonl(path, rows)
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specimens", nargs="*", help="specimen dirs (default: the --set)")
    ap.add_argument("--set", choices=sorted(SETS), default="all", help="named set when no dirs are given")
    ap.add_argument("--store", default="/tmp/cleave-overlay-gym")
    ap.add_argument("--out-dir", default=str(SERVER_DIR / "data"))
    ap.add_argument("--n-example", type=int, default=5, help="pivots to write to example.jsonl (0: leave it alone)")
    a = ap.parse_args(argv)
    rows, anchors = [], []
    for p in a.specimens or specimen_set(a.set):
        r, an = pivot_rows(p, a.store)
        rows += r
        anchors += an
        print(f"{Path(p).name}: {len(r)} pivots")
    out = Path(a.out_dir)
    _write_jsonl(out / "anchors.jsonl", anchors)
    _write_jsonl(out / "train.jsonl", rows)
    # --n-example 0 leaves the committed example.jsonl alone; otherwise it is rewritten only if its pivots change
    if a.n_example > 0 and not write_example(out / "example.jsonl", rows[: a.n_example]):
        print(f"kept {out / 'example.jsonl'} (same pivots)")
    print(f"wrote {len(anchors)} anchors (store {a.store}) and {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
