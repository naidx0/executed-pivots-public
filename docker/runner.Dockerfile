# H24 runner: h20-runner:local (bookworm, python3.11, jq, gawk, gzip, git, make, gcc) plus the toolchains the two
# new specimens need. The overlay backend's base root is the container's own root, so the union of the specimens'
# Dockerfile toolchains has to live here: nodejs + npm (csvsum-npm-package) and golang-go (tally-go-build).
FROM h20-runner:local
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm golang-go ca-certificates \
    && rm -rf /var/lib/apt/lists/*
