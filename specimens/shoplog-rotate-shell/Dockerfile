# shoplog-rotate-shell: environment image.
# The overlay backend used by validate.py and the audit cannot run this file (no Docker daemon inside a world); it
# runs setup.sh on the runner's root instead (h17-runner:local, python:3.11-slim-bookworm), which is this base:
# bash, coreutils and gzip, plus python3 for the verifier.
FROM python:3.11-slim-bookworm
COPY setup.sh /tmp/setup.sh
RUN bash -e /tmp/setup.sh && rm /tmp/setup.sh
WORKDIR /app
