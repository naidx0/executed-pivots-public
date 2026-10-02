#!/usr/bin/env bash
# Real executed_pivot rollouts from a local model, under the shared GPU lock. Git Bash on Windows (Docker Desktop)
# or Linux with Docker. See gym/README.md.
#
#   export LOCAL_POLICY_BASE_URL=http://127.0.0.1:8080/v1   # llama-server; Ollama: http://127.0.0.1:11434/v1
#   export LOCAL_POLICY_MODEL=<served model name>            # Ollama needs it; llama-server ignores it
#   # LOCAL_POLICY_API_KEY only if the server wants one; it is passed to Docker by name, never written down
#   gym/run_local_policy.sh -n 2
#
# Steps (the GPU lock is held for step 3 only):
#   1. dataset (CPU): prepare.py --set all in h24-runner:local -> $OUT/data (89 pivots), then the --pivots/--limit
#      selection -> $OUT/data/selected.jsonl
#   2. take the GPU lock ($GPU_LOCK, protocol line via gym/gpu_lock.py); exit 3 if another lane holds it
#   3. optional --serve-cmd starts the model server; wait for $LOCAL_POLICY_BASE_URL/models; then N rollouts per
#      pivot through the local-policy proxy: --driver direct (pivots.gym.rollouts, no Gym install) or
#      --driver gym (gym eval run in gym-runner:local with configs/local_policy.yaml)
#   4. stop the served model (if this script started it) and release the lock; also on any exit (trap)
#   5. labels (CPU): X and J from the rollouts, E and P from the specimens' verifiers -> labels.jsonl,
#      summary.json, summary.md (H31 verdict)
#
# Options:
#   -n N                 rollouts per pivot (default 2)
#   --pivots LIST        only these pivots, task:turn,task:turn (default: all 89)
#   --limit K            only the first K selected pivots
#   --driver D           direct (default) | gym
#   --serve-cmd CMD      start the model server with this command after taking the lock, stop it before release
#   --serve-wait S       seconds to wait for the endpoint (default 600)
#   --out DIR            output dir, relative to the repo root (default out/gym/run-<UTC stamp>)
#   --lane NAME          lane= in the lock line (default gym-rollouts)
#   --lock-wait S        keep trying for a held lock this long (default 0: exit 3 at once)
#   --skip-prepare       reuse $OUT/data from an earlier run of the same --out
#   --skip-labels        stop after the rollouts
#   --lock-only          take the lock, print it, release it (a check of the lock path; nothing else runs)
# Environment: GPU_LOCK (default ~/.gpu/gpu.lock), IMAGE (h24-runner:local), GYM_IMAGE
# (gym-runner:local), VOLUME (gym-store), PY (host python for the lock and the endpoint check), PORT (8913).
set -euo pipefail
export MSYS_NO_PATHCONV=1

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
W=$(pwd -W 2>/dev/null || pwd)

N=2; PIVOTS=""; LIMIT=""; DRIVER=direct; SERVE_CMD=""; SERVE_WAIT=600; LANE=gym-rollouts; LOCK_WAIT=0
SKIP_PREPARE=0; SKIP_LABELS=0; LOCK_ONLY=0
RUN="run-$(date -u +%Y%m%dT%H%M%SZ)"; OUT="out/gym/$RUN"
while [ $# -gt 0 ]; do
  case "$1" in
    -n) N=$2; shift 2 ;;
    --pivots) PIVOTS=$2; shift 2 ;;
    --limit) LIMIT=$2; shift 2 ;;
    --driver) DRIVER=$2; shift 2 ;;
    --serve-cmd) SERVE_CMD=$2; shift 2 ;;
    --serve-wait) SERVE_WAIT=$2; shift 2 ;;
    --out) OUT=${2%/}; shift 2 ;;
    --lane) LANE=$2; shift 2 ;;
    --lock-wait) LOCK_WAIT=$2; shift 2 ;;
    --skip-prepare) SKIP_PREPARE=1; shift ;;
    --skip-labels) SKIP_LABELS=1; shift ;;
    --lock-only) LOCK_ONLY=1; shift ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
done
case "$OUT" in /*|?:*) echo "--out must be relative to the repo root (the containers see it at /mnt/src)" >&2; exit 2 ;; esac
case "$DRIVER" in direct|gym) ;; *) echo "--driver must be direct or gym" >&2; exit 2 ;; esac

GPU_LOCK=${GPU_LOCK:-~/.gpu/gpu.lock}
IMAGE=${IMAGE:-h24-runner:local}; GYM_IMAGE=${GYM_IMAGE:-gym-runner:local}; VOLUME=${VOLUME:-gym-store}
PORT=${PORT:-8913}
if [ -z "${PY:-}" ]; then
  for c in "$ROOT/.venv/Scripts/python.exe" "$ROOT/.venv/bin/python" python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi
  done
fi
export PYTHONPATH="$W"  # a Windows python needs the Windows form of the path
# The Windows pid of this shell holds the card (born= is that process's start time); $$ elsewhere.
HOLDER=$(cat /proc/$$/winpid 2>/dev/null || echo $$)
UNETC='umount /etc/resolv.conf /etc/hosts /etc/hostname 2>/dev/null || true'
# Unmounting /etc/resolv.conf and /etc/hosts (backup-cron-repair needs /etc overlayable) also drops Docker's
# resolver, so the model host's name is pinned to its IPv4 address first.
PINHOST='IP=$(getent ahostsv4 host.docker.internal | awk "NR==1{print \$1}"); if [ -n "$IP" ]; then export LOCAL_POLICY_BASE_URL=${LOCAL_POLICY_BASE_URL//host.docker.internal/$IP}; fi;
  RH=$(echo "$LOCAL_POLICY_BASE_URL" | sed -E "s#^[a-z]+://([^/:]+).*#\1#"); RIP=$(getent ahostsv4 "$RH" | awk "NR==1{print \$1}")'
# A remote endpoint (https) keeps its host name for TLS, so after the unmount its address goes back into /etc/hosts.
REPIN='if [ -n "$RIP" ] && [ "$RIP" != "$RH" ]; then echo "$RIP $RH" >> /etc/hosts; fi'
STORE=/tmp/gym-rollouts-$(basename "$OUT")

# Docker Desktop resolves host.docker.internal itself (host-gateway maps it to an unreachable IPv6 there);
# plain Linux Docker needs the mapping.
if [ "$W" = "$ROOT" ]; then ADDHOST=--add-host=host.docker.internal:host-gateway; else ADDHOST=; fi

log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }
dock() {  # image command...
  local img=$1; shift
  docker run --rm --privileged $ADDHOST \
    -v "$VOLUME":/tmp -v "$W":/mnt/src -w /mnt/src -e PYTHONPATH=/mnt/src -e PYTHONDONTWRITEBYTECODE=1 \
    -e LOCAL_POLICY_BASE_URL="${CBASE:-}" -e LOCAL_POLICY_MODEL -e LOCAL_POLICY_API_KEY -e LOCAL_POLICY_TIMEOUT \
    -e LOCAL_POLICY_RETRIES -e LOCAL_POLICY_MAX_TOKENS "$img" bash -c "$*"
}

HAVE_LOCK=0; SERVE_PID=""
stop_served() {
  [ -n "$SERVE_PID" ] || return 0
  local wp; wp=$(cat /proc/$SERVE_PID/winpid 2>/dev/null || true)
  log "stopping the served model (pid $SERVE_PID${wp:+, windows pid $wp})"
  if [ -n "$wp" ] && command -v taskkill >/dev/null 2>&1; then taskkill /PID "$wp" /T /F >/dev/null 2>&1 || true
  else kill "$SERVE_PID" 2>/dev/null || true; fi
  SERVE_PID=""
}
release_lock() {
  [ "$HAVE_LOCK" = 1 ] || return 0
  "$PY" gym/gpu_lock.py release --lock "$GPU_LOCK" --lane "$LANE" --pid "$HOLDER" && HAVE_LOCK=0
}
cleanup() { local rc=$?; stop_served; release_lock || true; exit $rc; }
trap cleanup EXIT
trap 'exit 130' INT TERM

take_lock() {
  local what=$1 rc=0
  "$PY" gym/gpu_lock.py take --lock "$GPU_LOCK" --lane "$LANE" --pid "$HOLDER" --wait "$LOCK_WAIT" \
    --what "$what" || rc=$?
  if [ $rc -ne 0 ]; then log "GPU lock not taken (exit $rc); nothing ran on the GPU"; exit 3; fi
  HAVE_LOCK=1
}

if [ "$LOCK_ONLY" = 1 ]; then
  take_lock "lock check only (gym/run_local_policy.sh --lock-only)"
  release_lock
  exit 0
fi

[ -n "${LOCAL_POLICY_BASE_URL:-}" ] || { echo "set LOCAL_POLICY_BASE_URL (see --help)" >&2; exit 2; }
# The containers reach a model server on this machine's loopback through host.docker.internal.
CBASE=$(printf '%s' "$LOCAL_POLICY_BASE_URL" | sed -E 's#//(127\.0\.0\.1|localhost)([:/])#//host.docker.internal\2#')
mkdir -p "$OUT"
log "run $OUT: driver $DRIVER, $N rollouts per pivot${PIVOTS:+, pivots $PIVOTS}${LIMIT:+, first $LIMIT}"

# 1. dataset (CPU, before the lock)
if [ "$SKIP_PREPARE" = 0 ]; then
  log "dataset: prepare.py --set all in $IMAGE (store $STORE)"
  dock "$IMAGE" "$UNETC; rm -rf $STORE; python3 resources_servers/executed_pivot/scripts/prepare.py --set all \
    --store $STORE --out-dir $OUT/data --n-example 0" > "$OUT/prepare.log" 2>&1 \
    || { log "prepare failed; see $OUT/prepare.log"; exit 1; }
fi
"$PY" - "$OUT/data/train.jsonl" "$OUT/data/selected.jsonl" "$PIVOTS" "$LIMIT" <<'EOF'
import json, sys
from pivots.gym.rollouts import select_rows
src, dst, pivots, limit = sys.argv[1:]
rows = select_rows([json.loads(x) for x in open(src, encoding="utf-8") if x.strip()], pivots or None,
                   int(limit) if limit else None)
if not rows:
    sys.exit("no pivots selected")
with open(dst, "w", encoding="utf-8", newline="\n") as fh:
    fh.writelines(json.dumps(r) + "\n" for r in rows)
print(f"selected {len(rows)} pivots")
EOF
NPIV=$(wc -l < "$OUT/data/selected.jsonl" | tr -d ' ')

# 2. the GPU lock
take_lock "executed_pivot rollouts: $NPIV pivots x $N, ${LOCAL_POLICY_MODEL:-local} ($OUT)"
T0=$(date +%s)

# 3. model server, endpoint check, rollouts
if [ -n "$SERVE_CMD" ]; then
  log "starting the model server: $SERVE_CMD"
  bash -c "$SERVE_CMD" > "$OUT/serve.log" 2>&1 &
  SERVE_PID=$!
fi
log "waiting for $LOCAL_POLICY_BASE_URL/models (up to ${SERVE_WAIT}s)"
deadline=$(( $(date +%s) + SERVE_WAIT ))
until "$PY" -m pivots.students.local_policy --check > "$OUT/endpoint.json" 2>> "$OUT/endpoint.err"; do
  [ "$(date +%s)" -lt "$deadline" ] || { log "endpoint not up; see $OUT/endpoint.err"; exit 1; }
  sleep 5
done
log "endpoint: $(cat "$OUT/endpoint.json")"

set +e
PROXY="python3 -m pivots.students.local_policy --serve $PORT --log $OUT/policy.jsonl > $OUT/proxy.log 2>&1 & P=\$!"
if [ "$DRIVER" = direct ]; then
  ROLLOUTS=$OUT/rollouts.jsonl
  dock "$IMAGE" "$PINHOST; $UNETC; $REPIN; $PROXY; python3 -m pivots.gym.rollouts --rows $OUT/data/selected.jsonl \
    --anchors $OUT/data/anchors.jsonl --store $STORE --policy-url http://127.0.0.1:$PORT/v1 --n $N \
    --out $ROLLOUTS; rc=\$?; kill \$P; exit \$rc" 2>&1 | tee "$OUT/rollouts.log"
else
  ROLLOUTS=$OUT/gym/rollouts.jsonl
  # Gym reads the dataset and anchors from the server's data dir (both gitignored, machine-local).
  cp "$OUT/data/selected.jsonl" resources_servers/executed_pivot/data/train.jsonl
  cp "$OUT/data/anchors.jsonl" resources_servers/executed_pivot/data/anchors.jsonl
  S=executed_pivot_overlay_resources_server.resources_servers.executed_pivot
  VENVS=/tmp/gym-venvs
  dock "$GYM_IMAGE" "$PINHOST; $UNETC; $REPIN; $PROXY; \
    for d in resources_servers/executed_pivot responses_api_agents/simple_agent responses_api_models/inference_provider; do \
      mkdir -p $VENVS/\$d; ln -sfn /opt/gym $VENVS/\$d/.venv; done; \
    export PYTHONPATH=/mnt/src:/opt/Gym LOCAL_POLICY_PROXY_URL=http://127.0.0.1:$PORT/v1; \
    /opt/gym/bin/gym eval run \
      --config resources_servers/executed_pivot/configs/executed_pivot_overlay.yaml \
      --config resources_servers/executed_pivot/configs/local_policy.yaml \
      --agent executed_pivot_overlay_simple_agent --split train --num-repeats $N --concurrency 2 \
      --output $ROLLOUTS ++$S.overlay_store=$STORE +uv_venv_dir=$VENVS +skip_venv_if_present=true; \
    rc=\$?; kill \$P; exit \$rc" 2>&1 | tee "$OUT/rollouts.log"
fi
RC=${PIPESTATUS[0]}
set -e
HELD=$(( $(date +%s) - T0 ))

# 4. give the card back before the CPU-only labelling
stop_served
release_lock
[ "$RC" = 0 ] || { log "rollouts failed (exit $RC); see $OUT/rollouts.log"; exit 1; }
log "rollouts done in ${HELD}s under the lock: $(wc -l < "$ROLLOUTS" | tr -d ' ') in $ROLLOUTS"

# 5. labels (CPU)
if [ "$SKIP_LABELS" = 0 ]; then
  log "labels: E and P in $IMAGE"
  dock "$IMAGE" "$UNETC; python3 -m pivots.gym.labels --rollouts $ROLLOUTS --out $OUT \
    --store /tmp/gym-labels-$(basename "$OUT") --workers 4 --policy-log $OUT/policy.jsonl; rc=\$?; \
    rm -rf /tmp/gym-labels-$(basename "$OUT")-* $STORE; exit \$rc" > "$OUT/labels.log" 2>&1 \
    || { log "labels failed; see $OUT/labels.log"; exit 1; }
  cat "$OUT/summary.md"
fi
log "done: $OUT"
