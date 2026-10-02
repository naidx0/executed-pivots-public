"""Outcome checks for jq-signup-rollup: the report from /data/events is right, and the fixed script gives the
right report on a hidden events directory (4 days, a different mix, retries) written to a new output path."""
import os
import subprocess
import sys

sys.path.insert(0, "/tests")
import gen  # noqa: E402

ok = True


def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)


def read(p):
    return [ln.strip() for ln in open(p).read().splitlines() if ln.strip()] if os.path.isfile(p) else None


want = gen.expected(gen.make(424242, ["2026-03-09", "2026-03-10", "2026-03-11"], 42, 9))
got = read("/app/out/signups_by_plan.csv")
check(f"/app/out/signups_by_plan.csv: want {want} got {got}", got == want)

hidden = "/tmp/verify-events"
evs = gen.write(hidden, 777, ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-04"], 30, 7)
want_h = gen.expected(evs)
out = "/tmp/verify-out/nested/report.csv"
env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8"}
r = subprocess.run(["bash", "/app/etl/rollup.sh", hidden, out], cwd="/", env=env, capture_output=True, text=True,
                   timeout=120)
check(f"rollup.sh on a hidden directory exits 0 (rc={r.returncode}, stderr={r.stderr.strip()[-200:]!r})", r.returncode == 0)
got_h = read(out)
check(f"rollup.sh on a hidden directory: want {want_h} got {got_h}", got_h == want_h)
check("rollup.sh did not touch /app/out when given another output path", read("/app/out/signups_by_plan.csv") == got)

print("RESULT", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
