mkdir -p /app/ops /srv/shop/logs /srv/shop/state
cat > /app/ops/rotate.sh <<'EOF'
#!/bin/bash
# Rotate the shop's logs: NAME.log -> NAME.log.1 -> NAME.log.2.gz -> ... -> NAME.log.KEEP.gz, then dropped.
# Settings come from rotate.conf (ROTATE_CONF overrides the path). Cron runs this nightly from /.
set -euo pipefail

CONF=${ROTATE_CONF:-rotate.conf}
. "$CONF"

report=()
for log in $LOG_DIR/*.log; do
    size=$(stat -c %s "$log")
    # drop the oldest generation, then shift the compressed ones up by one
    rm -f $log.$RETAIN.gz
    for i in $(seq 2 $((RETAIN - 1))); do
        if [ -f $log.$i.gz ]; then
            mv $log.$i.gz $log.$((i + 1)).gz
        fi
    done
    if [ -f $log.1 ]; then
        gzip -c $log.1 > $log.2.gz && rm $log.1
    fi
    mv $log $log.1
    : > $log
    report+=("$(basename "$log") $size")
done
printf '%s\n' "${report[@]}" | sort > "$STATE_DIR/last_rotation.txt"
echo "rotated ${#report[@]} logs"
EOF
chmod +x /app/ops/rotate.sh
cat > /app/ops/rotate.conf <<'EOF'
# rotate.sh settings (sourced)
LOG_DIR=/srv/shop/logs
STATE_DIR=/srv/shop/state
# generations kept: NAME.log.1 plus NAME.log.2.gz .. NAME.log.KEEP.gz
KEEP=4
EOF
cat > /app/ops/README.md <<'EOF'
# ops

rotate.sh rotates every *.log in LOG_DIR (see rotate.conf). For each NAME.log, in this order:

1. delete NAME.log.KEEP.gz;
2. rename NAME.log.N.gz to NAME.log.(N+1).gz, for N from KEEP-1 down to 2;
3. gzip NAME.log.1 into NAME.log.2.gz and delete NAME.log.1;
4. rename NAME.log to NAME.log.1 and create a new empty NAME.log.

Then it writes STATE_DIR/last_rotation.txt: one "NAME SIZE" line per log (the size of NAME.log before the
rotation, in bytes), sorted. Log names can contain spaces. Cron runs it nightly from / as
`/app/ops/rotate.sh`; it reads rotate.conf next to itself unless ROTATE_CONF names another file.
EOF
cd /srv/shop/logs
gen() {  # file tag n: n lines
    for i in $(seq 1 "$3"); do printf '2024-06-%02d 0%d:00:00 %s line %d\n' "$((20 - ${4:-0}))" "$((i % 10))" "$2" "$i"; done > "$1"
}
gen api.log "api current" 6
gen api.log.1 "api gen1" 5 1
gen api.log.2 "api gen2" 4 2 && gzip -n api.log.2
gen api.log.3 "api gen3" 3 3 && gzip -n api.log.3
gen api.log.4 "api gen4" 2 4 && gzip -n api.log.4
gen worker.log "worker current" 7
gen worker.log.1 "worker gen1" 3 1
gen worker.log.2 "worker gen2" 2 2 && gzip -n worker.log.2
gen "payment gateway.log" "payment current" 5
gen "payment gateway.log.1" "payment gen1" 4 1
touch -d '2024-06-20 00:00:00' /srv/shop/logs/*
