mkdir -p /app/ops /app/data/archive
cat > /app/ops/backup.sh <<'EOF'
#!/bin/bash
# Nightly backup of the application data directory.
set -euo pipefail

CONF="${BACKUP_CONF:-/etc/app/backup.conf}"
source "$CONF"

mkdir -p "$DEST_DIR"
cd "$SRC_DIR"
tar -czf "$DEST_DIR/$ARCHIVE_NAME" .

: > "$DEST_DIR/MANIFEST.txt"
for f in $(find . -type f -name "$MANIFEST_GLOB" | sort); do
    sha256sum "$f" >> "$DEST_DIR/MANIFEST"
done

echo "backup ok: $DEST_DIR/$ARCHIVE_NAME"
EOF
chmod +x /app/ops/backup.sh
cat > /app/ops/backup.conf <<'EOF'
# Backup settings, sourced by backup.sh
SRC_DIR=/app/data
DEST_DIR = /var/backups/app
ARCHIVE_NAME=app-data.tar.gz
MANIFEST_GLOB="*.csv"
EOF
cat > /app/ops/backup.cron <<'EOF'
# m h dom mon dow user command
30 2 * * root /app/ops/backup.sh >> /var/log/app-backup.log 2>&1
EOF
cat > /app/ops/README.md <<'EOF'
Ops scripts. backup.sh is run nightly by cron (see backup.cron, installed
into /etc/cron.d/app-backup). Config lives next to the script in backup.conf.
EOF
cat > /app/data/orders.csv <<'EOF'
order_id,customer_id,total
1001,C01,19.99
1002,C07,5.00
1003,C01,42.10
EOF
cat > /app/data/customers.csv <<'EOF'
customer_id,name
C01,Ada Lovelace
C07,Grace Hopper
EOF
cat > /app/data/archive/refunds.csv <<'EOF'
order_id,amount
0998,3.50
EOF
cat > /app/data/notes.txt <<'EOF'
Q3 data freeze: do not edit orders.csv by hand.
EOF
