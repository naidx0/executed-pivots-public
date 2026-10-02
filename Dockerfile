# Executed Pivots live demo ("try a step"). Not published anywhere yet; see pivots/demo/README.md.
#
#   docker build -t executed-pivots-demo .
#   docker run --rm -p 7860:7860 \
#     --security-opt seccomp=unconfined --security-opt apparmor=unconfined --security-opt systempaths=unconfined \
#     -v xp-demo-store:/mnt/xp-store \
#     executed-pivots-demo
#
# Why those flags (and nothing more: no --privileged, no --cap-add, the process is uid 10001):
#   seccomp=unconfined      Docker's default seccomp profile refuses unshare(CLONE_NEWUSER) and mount() to
#                           processes without CAP_SYS_ADMIN; the overlay backend makes a user namespace per run.
#   apparmor=unconfined     the docker-default AppArmor profile denies mount, including overlay mounts made
#                           inside that user namespace (hosts without AppArmor ignore this flag).
#   systempaths=unconfined  Docker masks parts of /proc (kcore, sysrq-trigger, ...) with bind mounts, and the
#                           kernel refuses a fresh procfs mount in a new pid namespace while /proc is masked.
#   -v ...:/mnt/xp-store    the overlay store must sit on a real filesystem (ext4, xfs): overlayfs does not
#                           accept an upper directory on overlayfs, which is what the container's own root is.
#                           It must also sit outside the directories a world sees (/mnt and /tmp are left out).
# The host kernel must allow unprivileged user namespaces (Ubuntu 23.10+ with AppArmor may also need
# sysctl kernel.apparmor_restrict_unprivileged_userns=0).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# The container's root filesystem is the base image of every forked world, so it carries the tools the
# six specimen tasks use (python3, git, gcc, make, mawk, tar, gzip, coreutils) and util-linux for
# unshare, setpriv and prlimit.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv git gcc make libc6-dev mawk tar gzip util-linux procps ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Code lives in /opt: /app is the specimens' working directory inside the worlds.
WORKDIR /opt/executed-pivots
COPY pyproject.toml README.md ./
COPY cleave ./cleave
COPY pivots ./pivots
COPY specimens ./specimens
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -e '.[demo]'

RUN useradd --uid 10001 --create-home demo && mkdir -p /mnt/xp-store && chown demo:demo /mnt/xp-store
USER demo
ENV XP_DEMO_STORE=/mnt/xp-store/store XP_DEMO_TIMEOUT=10 XP_DEMO_CONCURRENCY=2
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health', timeout=4)"
CMD ["/opt/venv/bin/python", "-m", "pivots.demo", "--host", "0.0.0.0", "--port", "7860"]
