#!/usr/bin/env bash
# Run the Executed Pivots live demo ("try a step") locally on Linux or WSL2, as an ordinary user.
#
#   scripts/demo.sh                 # http://127.0.0.1:7860
#   PORT=8080 scripts/demo.sh
#   HOST=0.0.0.0 scripts/demo.sh    # listen on every interface (put it behind a proxy with TLS first)
#
# Needs Python 3.11+, unprivileged user namespaces and overlayfs (WSL2 and stock Ubuntu/Debian have both),
# plus the tools the six specimen tasks use: git, gcc, make, awk, tar, gzip. No Docker, no sudo.
# Creates .venv-demo in the repo on first run. Worlds live in $XP_DEMO_STORE (default /tmp/xp-demo-$USER),
# which is deleted when the server exits.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$(id -u)" = 0 ]; then
  echo "demo.sh: refusing to run as root. Visitors' commands run inside the fork as uid 0 of a user namespace;" >&2
  echo "under real root that uid owns the host's files. Run it as an ordinary user." >&2
  exit 1
fi
case "$(uname -s)" in Linux) ;; *) echo "demo.sh: Linux or WSL2 only (the overlay backend needs Linux namespaces)" >&2; exit 1;; esac

PY=${PYTHON:-python3}
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "demo.sh: needs Python 3.11 or newer" >&2; exit 1; }
for t in unshare setpriv prlimit git gcc make awk tar gzip; do
  command -v "$t" >/dev/null || echo "demo.sh: warning: '$t' is missing; some pivots will not behave as in the audit" >&2
done

if [ ! -x .venv-demo/bin/python ]; then
  "$PY" -m venv .venv-demo
  .venv-demo/bin/pip install -q --upgrade pip
  .venv-demo/bin/pip install -q -e '.[demo]'
fi

export XP_DEMO_STORE=${XP_DEMO_STORE:-/tmp/xp-demo-$(id -un)}
trap 'rm -rf "$XP_DEMO_STORE"' EXIT
echo "Executed Pivots demo: http://${HOST:-127.0.0.1}:${PORT:-7860}  (store $XP_DEMO_STORE; Ctrl-C to stop)"
.venv-demo/bin/python -m pivots.demo --host "${HOST:-127.0.0.1}" --port "${PORT:-7860}" "$@"
