"""executed_pivot inside NeMo Gym: the server's own suite, collected from the repo's tests/.

The tests live with the server (resources_servers/executed_pivot/tests/test_app.py, where
`gym env test` runs them). They need NeMo Gym main (the commit pinned in the server's
requirements.txt, which needs Python >= 3.13.14); without it this module skips. To run it,
use the Gym venv from resources_servers/executed_pivot/README.md:

    .venv-gym/bin/python -m pytest tests/test_gym_executed_pivot.py
"""
import pytest

gym = pytest.importorskip("nemo_gym.base_resources_server", reason="NeMo Gym is not installed (see resources_servers/executed_pivot/README.md)")
if not hasattr(gym, "ReverifyMode"):
    pytest.skip("this NeMo Gym is older than the pinned main commit (PyPI 0.4.0 lacks ReverifyMode and mask_sample)",
                allow_module_level=True)

from resources_servers.executed_pivot.tests.test_app import *  # noqa: E402,F401,F403
