"""ContreeWorld against contree-client's own in-memory double (no key, no network).

Checks the wire contract we depend on: network off by default, the result image
becomes the checkpoint, keep=False spawns are disposable and do not advance,
exit codes and stdout are read from metadata.result.
"""
import pytest

contree_client = pytest.importorskip("contree_client")
from contree_client import models, testing  # noqa: E402

from cleave.world.contree import ContreeWorld  # noqa: E402


def op(uuid, image, result_image, exit_code=0, stdout="", status="SUCCESS", timed_out=False):
    return models.OperationResponse.from_dict({
        "uuid": uuid, "kind": "instance", "status": status, "result_image_uuid": result_image,
        "metadata": {"command": "x", "image": image,
                     "result": {"state": {"exit_code": exit_code, "timed_out": timed_out},
                                "stdout": {"value": stdout, "encoding": "ascii"},
                                "stderr": {"value": "", "encoding": "ascii"}}},
    })


def spawn(uuid):
    return models.InstanceSpawnResponse.from_dict({"uuid": uuid})


def test_run_checkpoint_fork_and_disposable():
    c = testing.ContreeClient()  # queued results are consumed in order, the last one is sticky
    c.mock("spawn_instance", spawn("op-1"))
    c.mock("spawn_instance", spawn("op-2"))
    c.mock("wait_operation", op("op-1", "img-0", "img-1", stdout="hi\n"))
    c.mock("wait_operation", op("op-2", "img-1", None, exit_code=3))
    w = ContreeWorld("img-0", client=c, _fresh=False)
    r = w.run(["echo", "hi"])
    assert r.ok and r.stdout == "hi\n" and w.checkpoint() == "img-1"
    call = c.calls_for("spawn_instance")[0]
    assert call.args[1] == "img-0"
    assert call.kwargs["networking"].enabled is False and call.kwargs["disposable"] is False

    fork = w.fork(w.checkpoint())
    r2 = fork.run(["sh", "-c", "exit 3"], keep=False)
    assert r2.exit_code == 3 and fork.checkpoint() == "img-1"
    assert c.calls_for("spawn_instance")[-1].kwargs["disposable"] is True
    assert c.calls_for("spawn_instance")[-1].args[1] == "img-1"


def test_timeout_and_failed_operation_are_reported():
    c = testing.ContreeClient()
    c.mock("spawn_instance", spawn("op-3"))
    c.mock("wait_operation", op("op-3", "img-0", None, exit_code=None, timed_out=True, status="FAILED"))
    w = ContreeWorld("img-0", client=c, _fresh=False)
    r = w.run(["sleep", "999"], timeout_s=1)
    assert r.exit_code == 124 and "FAILED" in r.stderr and w.checkpoint() == "img-0"


def test_network_opt_in():
    c = testing.ContreeClient()
    c.mock("spawn_instance", spawn("op-4"))
    c.mock("wait_operation", op("op-4", "img-0", "img-4"))
    ContreeWorld("img-0", client=c, _fresh=False, network=True).run(["true"])
    assert c.calls_for("spawn_instance")[0].kwargs["networking"].enabled is True
