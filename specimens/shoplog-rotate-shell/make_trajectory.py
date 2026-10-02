"""Writes trajectory.json for shoplog-rotate-shell (the expert's turns are authored here for JSON escaping)."""
import json
from pathlib import Path


def turn(analysis, plan, cmds, done=False):
    return {"analysis": analysis, "plan": plan,
            "commands": [{"keystrokes": k, "duration": d} for k, d in cmds], "task_complete": done}


SCRATCH = ("rm -rf /tmp/rt && mkdir -p /tmp/rt/state && cp -a /srv/shop/logs /tmp/rt/logs && "
           "printf 'LOG_DIR=/tmp/rt/logs\\nSTATE_DIR=/tmp/rt/state\\nKEEP=4\\n' > /tmp/rt/rotate.conf\n")
SHOW = ("for f in /tmp/rt/logs/api.log*; do printf '%s: ' \"${f##*/}\"; zcat -f \"$f\" | head -n 1; echo; done; "
        "cat /tmp/rt/state/last_rotation.txt\n")
LOOP = """cat > rotate.sh <<'EOF'
#!/bin/bash
# Rotate the shop's logs: NAME.log -> NAME.log.1 -> NAME.log.2.gz -> ... -> NAME.log.KEEP.gz, then dropped.
# Settings come from rotate.conf (ROTATE_CONF overrides the path). Cron runs this nightly from /.
set -euo pipefail

CONF=${ROTATE_CONF:-rotate.conf}
. "$CONF"

report=()
for log in "$LOG_DIR"/*.log; do
    size=$(stat -c %s "$log")
    # drop the oldest generation, then shift the compressed ones up by one
    rm -f "$log.$KEEP.gz"
    for i in $(seq 2 $((KEEP - 1))); do
        if [ -f "$log.$i.gz" ]; then
            mv "$log.$i.gz" "$log.$((i + 1)).gz"
        fi
    done
    if [ -f "$log.1" ]; then
        gzip -c "$log.1" > "$log.2.gz" && rm "$log.1"
    fi
    mv "$log" "$log.1"
    : > "$log"
    report+=("$(basename "$log") $size")
done
printf '%s\\n' "${report[@]}" | sort > "$STATE_DIR/last_rotation.txt"
echo "rotated ${#report[@]} logs"
EOF
"""

traj = [
    turn("I'm at /app. The rotation script /app/ops/rotate.sh is broken. Before touching the real logs I need to read "
         "the script, its config and the README, and see what is in /srv/shop/logs.",
         "Read rotate.sh, rotate.conf and the README, and list the log directory.",
         [("cat /app/ops/rotate.sh /app/ops/rotate.conf /app/ops/README.md\n", 0.1), ("ls -la /srv/shop/logs\n", 0.1)]),
    turn("rotate.sh sources $ROTATE_CONF or rotate.conf, then for each *.log removes .$RETAIN.gz, shifts the .N.gz files "
         "with `seq 2 $((RETAIN - 1))`, gzips .1 into .2.gz, moves the log to .1 and truncates it. rotate.conf sets "
         "LOG_DIR=/srv/shop/logs, STATE_DIR=/srv/shop/state and KEEP=4 (not RETAIN). The logs are api.log (with .1 and "
         ".2.gz to .4.gz), worker.log (.1, .2.gz) and 'payment gateway.log' (.1), a name with a space. Running the "
         "script on the real logs must happen exactly once, so I test on a copy.",
         "Copy the logs to /tmp/rt with a scratch config, and run the script against it with ROTATE_CONF.",
         [(SCRATCH, 0.5), ("cd /app/ops\n", 0.1), ("ROTATE_CONF=/tmp/rt/rotate.conf ./rotate.sh; echo rc=$?\n", 1.0)]),
    turn("It fails at once: `RETAIN: unbound variable` (set -u), exit 1. The config names the setting KEEP, and "
         "rotate.conf must not change, so the script has to read KEEP.",
         "Rename RETAIN to KEEP in rotate.sh and rerun on a fresh copy.",
         [("sed -i 's/RETAIN/KEEP/g' rotate.sh\n", 0.1), (SCRATCH, 0.5),
          ("ROTATE_CONF=/tmp/rt/rotate.conf ./rotate.sh; echo rc=$?\n", 1.0), ("ls /tmp/rt/logs\n", 0.1)]),
    turn("Now it gets past api.log but dies on 'payment gateway.log': `[: /tmp/rt/logs/payment: binary operator "
         "expected` and `mv: target 'gateway.log.1': No such file or directory`, rc=1. Every $log expansion is "
         "unquoted, so the name splits at the space. The copy is half-rotated (api.log done, the rest not).",
         "Rewrite rotate.sh with every expansion quoted (\"$LOG_DIR\"/*.log, \"$log.$i.gz\", ...), keeping the same "
         "steps, and check it parses.",
         [(LOOP, 0.5), ("bash -n rotate.sh && echo syntax ok\n", 0.1)]),
    turn("The rewrite is in place and `bash -n` reports syntax ok.",
         "Run it on a fresh copy and look at the first line of each api.log generation and at the report.",
         [(SCRATCH, 0.5), ("ROTATE_CONF=/tmp/rt/rotate.conf ./rotate.sh; echo rc=$?\n", 1.0), (SHOW, 0.5)]),
    turn("All three logs rotate now (rotated 3 logs, rc=0) and the report lists api.log 234, payment gateway.log 215 "
         "and worker.log 294. But the api generations are wrong: api.log.2.gz holds gen1 as it should, api.log.3.gz is "
         "missing, and api.log.4.gz holds gen2. The shift runs upwards (seq 2 3): .2.gz moves onto .3.gz, overwriting "
         "gen3, and then that file moves on to .4.gz. It has to go from KEEP-1 down to 2.",
         "Make the shift loop count down, rerun on a fresh copy and check the generations again.",
         [("sed -i 's/seq 2 $((KEEP - 1))/seq $((KEEP - 1)) -1 2/' rotate.sh && grep -n 'seq' rotate.sh\n", 0.1),
          (SCRATCH, 0.5), ("ROTATE_CONF=/tmp/rt/rotate.conf ./rotate.sh; echo rc=$?\n", 1.0), (SHOW, 0.5)]),
    turn("Now the api generations are right: api.log is empty, .1 is the old current log, .2.gz gen1, .3.gz gen2, "
         ".4.gz gen3, and gen4 was dropped. Cron runs the script as /app/ops/rotate.sh from /, where the default "
         "`rotate.conf` is a relative path.",
         "Test that on a copy: put the script next to the scratch config in /tmp/rt and run it from / without "
         "ROTATE_CONF.",
         [(SCRATCH, 0.5), ("cp rotate.sh /tmp/rt/rotate.sh\n", 0.1),
          ("cd / && env -i PATH=/usr/bin:/bin /tmp/rt/rotate.sh; echo rc=$?\n", 1.0)]),
    turn("From / it fails: `rotate.conf: No such file or directory`, rc=1, so the nightly cron run would never rotate "
         "anything. The config has to be found next to the script.",
         "Default CONF to the directory of $0, then repeat the from-/ test on a fresh copy.",
         [("cd /app/ops\n", 0.1),
          ("sed -i 's|^CONF=${ROTATE_CONF:-rotate.conf}$|CONF=${ROTATE_CONF:-$(dirname \"$0\")/rotate.conf}|' rotate.sh "
           "&& grep -n '^CONF' rotate.sh\n", 0.1),
          (SCRATCH, 0.5), ("cp rotate.sh /tmp/rt/rotate.sh\n", 0.1),
          ("cd / && env -i PATH=/usr/bin:/bin /tmp/rt/rotate.sh; echo rc=$?; " + SHOW, 1.0)]),
    turn("From / with a minimal environment the copy now reads /tmp/rt/rotate.conf, rotates 3 logs with rc=0, and the "
         "generations are right. The script is fixed, and the real logs are still untouched.",
         "Run the real rotation once, the way cron does, and list the result.",
         [("cd / && /app/ops/rotate.sh; echo rc=$?\n", 1.0), ("ls -la /srv/shop/logs\n", 0.1),
          ("cat /srv/shop/state/last_rotation.txt\n", 0.1)]),
    turn("The real run rotated 3 logs, rc=0. /srv/shop/logs now has empty api.log, worker.log and 'payment gateway.log', "
         "each with a .1, and the .gz generations shifted; last_rotation.txt lists api.log 234, payment gateway.log "
         "215, worker.log 294.",
         "Spot-check the first line of each generation of the real logs without changing anything.",
         [("for f in /srv/shop/logs/*; do printf '%s: ' \"${f##*/}\"; zcat -f \"$f\" | head -n 1; echo; done\n", 0.5)]),
    turn("Each real generation holds what it should: .1 the previous current log, .2.gz gen1, .3.gz gen2 and .4.gz gen3 "
         "for api; worker and payment gateway shifted the same way. rotate.sh reads KEEP, quotes names with spaces, "
         "shifts downwards and finds its config from any directory, and the real logs were rotated exactly once.",
         "The task is complete.", [], True),
]
Path(__file__).with_name("trajectory.json").write_text(json.dumps(traj, indent=1) + "\n", newline="\n")
