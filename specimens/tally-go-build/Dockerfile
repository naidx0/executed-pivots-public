# tally-go-build: environment image.
# The overlay backend used by validate.py and the audit cannot run this file (no Docker daemon inside a world); it
# runs setup.sh on the runner's root instead (docker/runner.Dockerfile), which has the same toolchain:
# Go 1.19, make and python3 from Debian bookworm.
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends golang-go make python3 && rm -rf /var/lib/apt/lists/*
COPY setup.sh /tmp/setup.sh
RUN bash -e /tmp/setup.sh && rm /tmp/setup.sh
WORKDIR /app
