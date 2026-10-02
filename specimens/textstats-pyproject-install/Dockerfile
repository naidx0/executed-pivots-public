# textstats-pyproject-install: environment image.
# The overlay backend used by validate.py and the audit cannot run this file (no Docker daemon); it runs
# setup.sh on the host root instead, which has the same toolchain (Python 3.11, setuptools >= 61, pip).
FROM python:3.11-slim
# python:3.11-slim ships pip and setuptools; the task is offline, so nothing else is installed.
RUN python3 -c "import setuptools, sys; assert int(setuptools.__version__.split('.')[0]) >= 61, setuptools.__version__"
COPY setup.sh /tmp/setup.sh
RUN bash -e /tmp/setup.sh && rm /tmp/setup.sh
WORKDIR /app
