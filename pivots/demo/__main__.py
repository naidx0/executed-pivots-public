"""Run the demo server: python -m pivots.demo [--host 127.0.0.1] [--port 7860].

Refuses to run as root: inside a fork, uid 0 without capabilities still owns whatever the server's uid owns
on the host, so the server must be an ordinary user (see pivots/demo/sandbox.py).
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pivots.demo")
    ap.add_argument("--host", default=os.environ.get("XP_DEMO_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("XP_DEMO_PORT", "7860")))
    ap.add_argument("--store", help="overlay store (default $XP_DEMO_STORE or /tmp/xp-demo-store)")
    ap.add_argument("--timeout", type=int, help="per-batch timeout in seconds (default 10)")
    ap.add_argument("--concurrency", type=int, help="sandbox runs at once (default 2)")
    ap.add_argument("--no-warm", action="store_true", help="build worlds on first use instead of at startup")
    ap.add_argument("--pivots", choices=["decisive", "all"],
                    help="pivots a visitor may open: the 17 decisive ones (default) or every turn")
    a = ap.parse_args(argv)
    if os.geteuid() == 0:
        sys.exit("refusing to run as root: start the demo as an ordinary user (see pivots/demo/sandbox.py)")
    from cleave.world.overlay import available
    if not available():
        sys.exit("this machine cannot build overlay worlds: it needs unprivileged user namespaces and overlayfs "
                 "(Linux or WSL2; in a container see the Dockerfile's run flags)")
    for flag, env in ((a.store, "XP_DEMO_STORE"), (a.timeout, "XP_DEMO_TIMEOUT"), (a.concurrency, "XP_DEMO_CONCURRENCY"),
                      (a.pivots, "XP_DEMO_PIVOTS")):
        if flag is not None:
            os.environ[env] = str(flag)
    if a.no_warm:
        os.environ["XP_DEMO_WARM"] = "0"
    import uvicorn
    uvicorn.run("pivots.demo.app:app_from_env", factory=True, host=a.host, port=a.port, workers=1,
                limit_concurrency=64, timeout_keep_alive=5, log_level="info")


if __name__ == "__main__":
    main()
