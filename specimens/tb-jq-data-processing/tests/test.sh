#!/bin/bash
# Verifier for tb-jq-data-processing: Terminal-Bench's tests/test_outputs.py (Apache-2.0), unchanged, run with pytest 8.4.1
# from the runner image (docker/tb-runner.Dockerfile). Offline. Reward 1 only if every test passes.
mkdir -p /logs/verifier
cd /app 2>/dev/null || cd /
if python3 -m pytest -p no:cacheprovider -rA /tests/test_outputs.py; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
