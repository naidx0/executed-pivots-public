mkdir -p /var/log/webapp /app/reports /app/bin
cat > /app/README.md <<'EOF'
webapp ops notes

- nginx-style access logs: /var/log/webapp/access.log (today), access.log.1 (yesterday)
- error log: /var/log/webapp/error.log (not needed for traffic reports)
- reports go in /app/reports
EOF
python3 - <<'EOF'
# Deterministic log generator (fixed LCG, no clock, no random module).
state = 20260312
def rnd(n):
    global state
    state = (state * 1103515245 + 12345) % 2**31
    return (state >> 8) % n

ips = ["10.0.0.5", "10.0.0.12", "10.0.0.7", "10.0.1.20", "192.168.4.2", "172.16.0.9", "10.0.0.50", "203.0.113.8"]
weights = [9, 9, 6, 5, 4, 3, 2, 1]
pool = [ip for ip, w in zip(ips, weights) for _ in range(w)]
paths = ["/", "/login", "/api/orders", "/api/orders?page=2", "/api/orders?page=3", "/api/users", "/api/users?id=17",
         "/static/app.js", "/static/app.css", "/api/export", "/api/export?fmt=csv", "/search?q=boots"]
statuses = [200] * 14 + [201, 204, 301, 302, 304, 304, 400, 401, 403, 404, 404, 500, 502, 503]
agents = ["Mozilla/5.0 (X11; Linux x86_64)", "curl/8.5.0", "python-requests/2.31"]

def lines(day, n):
    out = []
    for i in range(n):
        sec = i * 37
        ts = f"{day:02d}/Mar/2026:{8 + sec // 3600:02d}:{sec // 60 % 60:02d}:{sec % 60:02d} +0000"
        if i % 9 == 4:
            out.append(f'10.0.0.1 - - [{ts}] "GET /healthz HTTP/1.1" 200 2 "-" "ELB-HealthChecker/2.0"')
            continue
        ip = pool[rnd(len(pool))]
        path = paths[rnd(len(paths))]
        st = statuses[rnd(len(statuses))]
        if path.startswith("/api/export") and rnd(3) == 0:
            st = 504
        method = "POST" if path == "/login" else "GET"
        size = 0 if st in (204, 304) else 120 + rnd(4000)
        out.append(f'{ip} - - [{ts}] "{method} {path} HTTP/1.1" {st} {size} "-" "{agents[rnd(3)]}"')
    return out

with open("/var/log/webapp/access.log.1", "w") as fh:
    fh.write("\n".join(lines(11, 140)) + "\n")
today = lines(12, 170)
# a late burst from one client: it ties 10.0.0.12 for second place (byte-wise order puts 10.0.0.12 first)
for k in range(13):
    today.append(f'10.0.0.7 - - [12/Mar/2026:09:{45 + k // 4:02d}:{(k * 13) % 60:02d} +0000] "GET /static/app.js HTTP/1.1" 200 5120 "-" "Mozilla/5.0 (X11; Linux x86_64)"')
with open("/var/log/webapp/access.log", "w") as fh:
    fh.write("\n".join(today) + "\n")
with open("/var/log/webapp/error.log", "w") as fh:
    fh.write("2026/03/12 09:14:02 [error] upstream timed out while reading response header\n")
EOF
