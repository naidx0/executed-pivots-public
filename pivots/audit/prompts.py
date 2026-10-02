"""Write the exact Terminus-2 prompt a student sees at every pivot of every specimen.

    python -m pivots.audit.prompts specimens/* --out out/prompts

out/prompts/<task>/<turn>.json = {"task", "turn", "messages": [...]} in the released rows' message layout.
"""
from __future__ import annotations

import argparse
import functools
import json
import os

from cleave.world.overlay import OverlayWorld
from pivots import specimen as S


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("specimens", nargs="+")
    ap.add_argument("--out", default="out/prompts")
    ap.add_argument("--store", default=f"/tmp/cleave-prompts-{os.getpid()}")
    a = ap.parse_args(argv)
    wf = functools.partial(OverlayWorld, store=a.store)
    for p in a.specimens:
        if not os.path.exists(os.path.join(p, "task.json")):
            continue
        sp = S.load(p)
        w = S.build(wf, sp)
        rr = S.run_expert(w, sp)
        obs = [s.rendered for s in rr.steps]
        os.makedirs(os.path.join(a.out, sp.name), exist_ok=True)
        for t in range(len(sp.actions)):
            msgs = S.prompt_for(sp, t, obs, f"root@{sp.host}:{sp.workdir}# ")
            with open(os.path.join(a.out, sp.name, f"{t:02d}.json"), "w") as fh:
                json.dump({"task": sp.name, "turn": t, "messages": msgs}, fh, indent=1)
        print(sp.name, len(sp.actions), "pivots")


if __name__ == "__main__":
    main()
