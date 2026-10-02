"""H50: an action too large for the sandbox's argv (E2BIG) is a backend error, not a crash of the judge."""
import errno

from pivots.effect.probe import execute


class _Fork:
    def run(self, argv, **kw):
        raise OSError(errno.E2BIG, "Argument list too long", "unshare")

    def close(self):
        pass


class _World:
    def fork(self, anchor):
        return _Fork()


def test_oversized_action_is_a_backend_error():
    eff = execute(_World(), "ck", "echo " + "A" * 200_000 + "\n", cwd="/app")
    assert eff.backend_error and "Argument list too long" in eff.backend_error
    assert eff.changed == {}
