#!/usr/bin/env bash
# Nebius Token Factory students for the Executed Pivots audit, in order, once the host is reachable:
#   0. preflight   unauthenticated GET /v1/models: is api.tokenfactory.nebius.com reachable? (no key sent, not billed)
#   1. models      list the served models (1 request); stop if $MODEL is not among them
#   2. smoke       one short chat completion (1 request)
#   3. audit       sample $N students at each of the $PIVOTS pivots and label them J/X/E and P ($N x pivots requests)
#
#   scripts/nebius_students.sh             # the real run, writes $OUT (default /mnt/project-files/nebius/tokenfactory)
#   scripts/nebius_students.sh --dry-run   # the same path against a local mock server, 2 pivots, no key, no network
#
# Knobs (environment): MODEL, N, PIVOTS (task:turn,...), OUT, MAX_REQUESTS (hard cap for the audit step), WORKERS.
# The key is read from $NEBIUS_API_KEY or /root/.config/nebius/tf_key and is never printed or written.
# Any 401/402/403, billing or quota error stops the step at once (exit 3); nothing is retried after that.
# Responses are cached in $OUT/cache, so rerunning after a stop only pays for what is missing.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD

MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B}
N=${N:-5}
# One decisive pivot per specimen (skipping the expert's batch there fails the task), from the stand-in audit.
PIVOTS=${PIVOTS:-access-log-summary:4,backup-cron-repair:3,billing-invoice-bugfix:3,ledger-git-revert:3,sensor-site-report:4,statcli-make-fix:5}
OUT=${OUT:-/mnt/project-files/nebius/tokenfactory}
WORKERS=${WORKERS:-4}
KEY_FILE=${KEY_FILE:-/root/.config/nebius/tf_key}
PY=${PY:-python3}
DRY=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY=1
  PIVOTS=${DRY_PIVOTS:-billing-invoice-bugfix:3,ledger-git-revert:3}
  OUT=${DRY_OUT:-$OUT/dryrun}
fi

n_pivots=$(tr ',' '\n' <<<"$PIVOTS" | grep -c .)
audit_requests=$((n_pivots * N))
expected=$((audit_requests + 2))
MAX_REQUESTS=${MAX_REQUESTS:-$((audit_requests + audit_requests / 3 + 4))}  # headroom for 429/5xx retries only
CACHE=${CACHE:-$OUT/cache}
STORE=/tmp/cleave-tf-audit-$$
mkdir -p "$OUT"

MOCK_PID=
cleanup() { [[ -n "$MOCK_PID" ]] && kill "$MOCK_PID" 2>/dev/null || true; rm -rf "$STORE"; (( DRY )) && rm -rf "$CACHE" || true; }
trap cleanup EXIT

if (( DRY )); then
  CACHE=${DRY_CACHE:-/tmp/cleave-tf-dryrun-cache-$$}
  PORT=${MOCK_PORT:-8913}
  "$PY" -m pivots.students.mock_server --port "$PORT" >/dev/null &
  MOCK_PID=$!
  export TOKENFACTORY_BASE_URL="http://127.0.0.1:$PORT/v1"
  export NEBIUS_API_KEY="dry-run-not-a-key"  # the real key never goes to the mock
  for _ in $(seq 50); do curl -s -o /dev/null "$TOKENFACTORY_BASE_URL/models" && break; sleep 0.1; done
else
  export TOKENFACTORY_BASE_URL=${TOKENFACTORY_BASE_URL:-https://api.tokenfactory.nebius.com/v1}
  if [[ -z "${NEBIUS_API_KEY:-}" ]]; then
    [[ -s "$KEY_FILE" ]] || { echo "no key: set NEBIUS_API_KEY or put it in $KEY_FILE" >&2; exit 2; }
    NEBIUS_API_KEY=$(<"$KEY_FILE")
    export NEBIUS_API_KEY
  fi
fi

echo "== Token Factory students ($([[ $DRY == 1 ]] && echo 'DRY RUN against a mock' || echo 'live'))"
echo "   model $MODEL; $N students x $n_pivots pivots -> $OUT"
echo "   expected model requests: $expected (1 list + 1 smoke + $audit_requests samples; fewer on cache hits)"
echo "   audit cap --max-requests $MAX_REQUESTS; base URL $TOKENFACTORY_BASE_URL"
echo "   run: $PY -m pivots.audit.run specimens/*/ --student tokenfactory:$MODEL --n-students $N --pivots $PIVOTS --progress --max-requests $MAX_REQUESTS --tf-cache $CACHE --out $OUT/audit"

if (( ! DRY )); then
  echo "== 0. preflight (no key sent)"
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$TOKENFACTORY_BASE_URL/models" 2>"$OUT/preflight.err" || true)
  if [[ "$code" == "000" || -z "$code" ]]; then
    echo "   STOP: $TOKENFACTORY_BASE_URL is not reachable: $(head -c 300 "$OUT/preflight.err")" >&2
    echo "   (a CONNECT 403 means this container's egress policy still blocks the host)" >&2
    exit 4
  fi
  echo "   reachable (HTTP $code without a key)"
fi

tf() { "$PY" -m pivots.students.tokenfactory --model "$MODEL" --base-url "$TOKENFACTORY_BASE_URL" --cache-dir "$CACHE" "$@"; }

echo "== 1. models"
tf --list-models --max-requests 3 --out "$OUT/models.json" > "$OUT/models.txt"
grep -i nemotron "$OUT/models.txt" | sed 's/^/   /' || true
if ! grep -qxF "$MODEL" "$OUT/models.txt"; then
  echo "   STOP: $MODEL is not served. Pick one of the ids above: MODEL=<id> $0 $*" >&2
  exit 5
fi

echo "== 2. smoke (1 request)"
tf --smoke --max-requests 3 --out "$OUT/smoke.json" | sed 's/^/   /'

echo "== 3. audit: sample + label J/X/E and P"
"$PY" -m pivots.audit.run "$REPO"/specimens/*/ --student "tokenfactory:$MODEL" --n-students "$N" \
  --pivots "$PIVOTS" --progress --max-requests "$MAX_REQUESTS" --tf-base-url "$TOKENFACTORY_BASE_URL" \
  --tf-cache "$CACHE" --out "$OUT/audit" --store "$STORE" --workers "$WORKERS"

total=$(wc -l < "$CACHE/requests.jsonl" 2>/dev/null || echo 0)
echo "== done: $total model requests recorded in $CACHE/requests.jsonl (all runs sharing this cache)"
echo "   results: $OUT/audit/{rows.jsonl,summary.json,progress.jsonl,progress.summary.json,students.jsonl,student_stats.json}"
