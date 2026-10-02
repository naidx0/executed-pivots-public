# acme-worker-ini-unit: environment image.
# The overlay backend used by validate.py and the audit cannot run this file (no Docker daemon inside a world); it
# runs setup.sh on the runner's root instead (h17-runner:local, python:3.11-slim-bookworm), which is this base.
# There is no systemd in the container: the unit is checked by /app/worker/unitcheck.py and enabled by hand.
# In Docker the task needs a writable /etc (H20: unmounts /etc/resolv.conf, /etc/hosts, /etc/hostname).
FROM python:3.11-slim-bookworm
COPY setup.sh /tmp/setup.sh
RUN bash -e /tmp/setup.sh && rm /tmp/setup.sh
WORKDIR /app
