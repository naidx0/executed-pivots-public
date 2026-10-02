# pricelist-perl-makemaker: environment image.
# The overlay backend used by validate.py and the audit cannot run this file (no Docker daemon inside a world); it
# runs setup.sh on the runner's root instead (h20-runner:local, bookworm), which has the same toolchain:
# perl 5.36 with ExtUtils::MakeMaker and prove, make, and python3 for the verifier.
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends perl make python3 && rm -rf /var/lib/apt/lists/*
COPY setup.sh /tmp/setup.sh
RUN bash -e /tmp/setup.sh && rm /tmp/setup.sh
WORKDIR /app
