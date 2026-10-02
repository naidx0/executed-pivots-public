# jq-signup-rollup: environment image.
# The overlay backend used by validate.py and the audit cannot run this file (no Docker daemon); it runs
# setup.sh on the host root instead, which already has bash, jq 1.7, mawk, gzip and python3.
FROM debian:bookworm-slim
# Needs the network, so it cannot run on this box; the host provides the same tools.
RUN apt-get update && apt-get install -y --no-install-recommends jq gawk gzip python3 && rm -rf /var/lib/apt/lists/*
COPY setup.sh /tmp/setup.sh
RUN bash -e /tmp/setup.sh && rm /tmp/setup.sh
WORKDIR /app
