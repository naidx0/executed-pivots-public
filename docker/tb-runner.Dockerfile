# H50 runner for the Terminal-Bench ports (specimens/tb-*): h24-runner:local plus what the ported tasks' own
# Dockerfiles and verifiers install. The overlay backend's base root is this container's root, so every task's
# toolchain has to live here; network is used only at image build time, the verifiers run offline.
#   g++          cpp-compatibility
#   dos2unix     processing-pipeline (its Dockerfile runs unix2dos; the expert may use dos2unix)
#   yq           jq-data-processing (apt yq, the jq wrapper its verifier calls)
#   pytest       every tb-* verifier (Terminal-Bench runs pytest 8.4.1)
#   pandas, numpy, pyarrow   heterogeneous-dates, pandas-etl, csv-to-parquet, grid-pattern-transform
#   faker, pyyaml            jq-data-processing (data generator, verifier)
FROM h24-runner:local
RUN apt-get update && apt-get install -y --no-install-recommends g++ dos2unix yq \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pytest==8.4.1 pandas==2.3.0 numpy==2.3.1 pyarrow==20.0.0 faker==24.0.0 pyyaml==6.0.2
