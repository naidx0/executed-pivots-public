# sqlite-email-dedupe: environment image.
# The overlay backend used by validate.py and the audit cannot run this file (no Docker daemon); it runs
# setup.sh on the host root instead. The task uses only python3's sqlite3 module (SQLite 3.45 on the host).
FROM python:3.11-slim
# Deliberately no sqlite3 CLI: the README tells the agent to use Python's sqlite3 module.
COPY setup.sh /tmp/setup.sh
RUN bash -e /tmp/setup.sh && rm /tmp/setup.sh
WORKDIR /app
