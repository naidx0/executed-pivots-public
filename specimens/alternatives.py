"""Alternative-approach checks: different-but-correct solutions must pass, plausible wrong ones must fail.

    python3 specimens/alternatives.py [-v]

Each case builds the specimen, runs one batch from the base state, and verifies the result.
"""
import functools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cleave.world.overlay import OverlayWorld
from pivots import specimen as S
from pivots.effect.terminus import Command
from pivots.replay import replay
wf = functools.partial(OverlayWorld, store="/tmp/cleave-specimens-store")


def run(name, script, want):
    sp = S.load(HERE / name)
    w = S.build(wf, sp)
    rr = replay(w, [[Command(script if script.endswith("\n") else script + "\n")]], default_cwd=sp.workdir)
    v = S.verify(w, rr.steps[-1].checkpoint, sp)
    flag = "ok " if v["reward"] == want else "UNEXPECTED"
    print(f"{flag} {name}: want {want} got {v['reward']} :: {script.splitlines()[0][:90]}")
    if v["reward"] != want or "-v" in sys.argv:
        print(rr.steps[-1].rendered[-800:])
        print(v["tail"])
    return v["reward"] == want


SENSOR_OK = '''cd /app/pipeline && python3 - <<'EOF'
s = open('report.py').read()
s = s.replace('    raise NotImplementedError("build_site_summary is not implemented yet")', """    import statistics
    by = {}
    for r in records:
        if r["status"] == "ok":
            by.setdefault(r["site"], []).append([x["value"] for x in r["readings"] if x["value"] is not None])
    return [dict(site=s, sensors=len(v), readings=sum(map(len, v)), mean_value=statistics.mean([x for l in v for x in l]), max_value=max(x for l in v for x in l)) for s, v in sorted(by.items())]""")
open('report.py', 'w').write(s)
EOF
REPORT_DIR=/app/reports python3 run.py'''
SENSOR_NOSTATUS = SENSOR_OK.replace('        if r["status"] == "ok":\n    ', '    ')
LOGS_PY = r'''cd /app/reports && python3 - <<'EOF'
import collections
reqs = []
for f in ['/var/log/webapp/access.log.1', '/var/log/webapp/access.log']:
    for line in open(f):
        p = line.split()
        path = p[6].split('?')[0]
        if path != '/healthz':
            reqs.append((p[0], path, p[8]))
ips = collections.Counter(r[0] for r in reqs)
cls = collections.Counter(r[2][0] for r in reqs)
out = [f'total_requests={len(reqs)}', f'unique_ips={len(ips)}'] + [f'status_{k}xx={cls[str(k)]}' for k in range(2, 6)] + ['top_ips:']
out += [f'{ip} {n}' for ip, n in sorted(ips.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
open('traffic_summary.txt', 'w').write('\n'.join(out) + '\n')
err = collections.Counter(r[1] for r in reqs if r[2].startswith('5'))
open('errors_by_path.csv', 'w').write('path,count\n' + ''.join(f'{p},{n}\n' for p, n in sorted(err.items(), key=lambda kv: (-kv[1], kv[0]))))
EOF'''
LOGS_NUMERIC_SORT = LOGS_PY.replace("key=lambda kv: (-kv[1], kv[0]))[:3]", "key=lambda kv: (-kv[1], tuple(map(int, kv[0].split('.')))))[:3]")
CMP_FIX = r"sed -i 's/return (int)(\*(const double \*)a - \*(const double \*)b);/return (*(const double *)a > *(const double *)b) - (*(const double *)a < *(const double *)b);/' src/stats.c"

A = [
    ("billing-invoice-bugfix", r"cd /app && sed -i 's/qty > BULK/qty >= BULK/; s/start = page \* per_page/start = per_page * (page - 1)/' billing/invoice.py", 1),
    ("billing-invoice-bugfix", r"cd /app && sed -i 's/qty > BULK/qty >= BULK/' billing/invoice.py && sed -i 's/range(1, 21)/range(21, 41)/; s/\[41, 42, 43, 44, 45\]/[]/' tests/test_invoice.py && python3 -m unittest discover -s tests 2>&1 | tail -1", 0),
    ("sensor-site-report", SENSOR_OK, 1),
    ("sensor-site-report", SENSOR_NOSTATUS, 0),
    ("sensor-site-report", SENSOR_OK.replace("REPORT_DIR=/app/reports python3 run.py", "python3 run.py && ls out"), 0),
    ("backup-cron-repair", "printf 'SRC_DIR=/app/data\\nDEST_DIR=/var/backups/app\\nARCHIVE_NAME=app-data.tar.gz\\nMANIFEST_GLOB=\"*.csv\"\\n' > /app/ops/backup.conf && sed -i 's|/etc/app/backup.conf|/app/ops/backup.conf|; s|/MANIFEST\"|/MANIFEST.txt\"|' /app/ops/backup.sh && echo '30 02 * * * root bash /app/ops/backup.sh' > /app/ops/backup.cron && cp /app/ops/backup.cron /etc/cron.d/app-backup && /app/ops/backup.sh", 1),
    ("backup-cron-repair", "sed -i 's/^DEST_DIR = /DEST_DIR=/' /app/ops/backup.conf && sed -i 's|/MANIFEST\"|/MANIFEST.txt\"|' /app/ops/backup.sh && echo '30 2 * * * root /app/ops/backup.sh' > /app/ops/backup.cron && cp /app/ops/backup.cron /etc/cron.d/app-backup && BACKUP_CONF=/app/ops/backup.conf /app/ops/backup.sh", 0),
    ("ledger-git-revert", r"cd /app/ledger && sed -i 's|    return int(amount \* rate \* 100) / 100|    return round(amount * rate, 2)|' ledger/tax.py && git commit -qam 'Fix rounding' && git branch release/1.1 && python3 -m unittest 2>&1 | tail -1", 1),
    ("ledger-git-revert", "cd /app/ledger && git reset -q --hard 84bda7f && git cherry-pick a72c823 c8c1125 >/dev/null && git revert --no-edit HEAD~2 2>&1 | tail -1; git branch release/1.1 && python3 -m unittest 2>&1 | tail -1", 0),
    ("ledger-git-revert", "cd /app/ledger && git revert --no-edit cdf209b >/dev/null && git checkout -q -b release/1.1", 0),
    ("access-log-summary", LOGS_PY, 1),
    ("access-log-summary", LOGS_NUMERIC_SORT, 0),
    ("statcli-make-fix", "cd /app/statcli && sed -i 's/^CFLAGS = .*/& -lm/' Makefile && " + CMP_FIX + " && make 2>&1 | tail -2", 0),
    ("statcli-make-fix", "cd /app/statcli && sed -i 's/-o $@ $(OBJS)$/-o $@ $(OBJS) -lm/' Makefile && " + CMP_FIX + " && make test", 1),
    ("statcli-make-fix", "cd /app/statcli && sed -i 's/-o $@ $(OBJS)$/-o $@ $(OBJS) -lm/' Makefile && make test", 0),
]
if __name__ == "__main__":
    results = [run(*a) for a in A]
    print(f"{sum(results)}/{len(results)} cases as expected")
    sys.exit(0 if all(results) else 1)
