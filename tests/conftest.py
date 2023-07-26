"""Configuration for pytest."""
import pytest

from mrinufft.operators import check_backend
from mrinufft.operators.interfaces import _REGISTERED_BACKENDS


def pytest_configure(config):
    """Configuration hook for pytest."""
    print("Available backends:")
    for backend in _REGISTERED_BACKENDS:
        print(f"{backend:<14}: {_REGISTERED_BACKENDS[backend][0]}")


# # for test directly parametrized by a backend
def pytest_runtest_setup(item):
    """Skip tests based on the backend."""
    if hasattr(item, "callspec"):
        if "backend" in item.callspec.params:
            backend = item.callspec.params["backend"]
            if not check_backend(backend):
                pytest.skip(f"Backend {backend} not available.")


# for test parametrized by an operator fixture, depending on a backend.
# This is more tricky.
def pytest_generate_tests(metafunc):
    """Generate the tests."""
    if "operator" in metafunc.fixturenames:
        for callspec in metafunc._calls:
            op_call = callspec.params["operator"]
            # Complicated datastructure
            # Acces the value for the backend parameter.
            backend = op_call.argvalues[
                [
                    i
                    for i, v in enumerate(op_call.param_defs)
                    if v.argnames[0] == "backend"
                ][0]
            ]
            # Only keep the callspec if the backend is available.
            if not check_backend(backend):
                callspec.marks.append(
                    pytest.mark.skip(f"Backend {backend} not available.")
                )
