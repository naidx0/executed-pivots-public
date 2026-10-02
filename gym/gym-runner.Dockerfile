# Gym runner for executed_pivot rollouts on this machine (Docker Desktop, Linux containers).
#
# h24-runner:local is the overlay backend's base root for all eleven specimens (python3.11, jq, gawk, gzip, git,
# make, gcc from h20-runner, plus nodejs/npm and golang-go for csvsum-npm-package and tally-go-build). This adds
# NeMo Gym main at the commit resources_servers/executed_pivot/requirements.txt pins, in its own Python 3.13 venv
# at /opt/gym, from a clone at /opt/Gym so Gym's model and agent servers (responses_api_models/*,
# responses_api_agents/*) are importable too. The repo itself is mounted at run time (PYTHONPATH=/mnt/src).
#
#   docker build -t gym-runner:local -f gym/gym-runner.Dockerfile gym
FROM h24-runner:local
ARG GYM_COMMIT=2921bf03c9dbe652545c47a246f350b25b8c9843
RUN python3 -m pip install --no-cache-dir uv
RUN git clone --filter=blob:none https://github.com/NVIDIA-NeMo/Gym /opt/Gym \
    && git -C /opt/Gym checkout "$GYM_COMMIT"
RUN uv python install 3.13 && uv venv --seed --python 3.13 /opt/gym \
    && uv pip install --python /opt/gym/bin/python -e "/opt/Gym[dev]" pytest
ENV GYM_PYTHON=/opt/gym/bin/python GYM_ROOT=/opt/Gym
