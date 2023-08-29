"""Configuration for pytest."""
import pytest

from mrinufft.operators import check_backend


def pytest_addoption(parser):
    """Add option to pytest."""
    parser.addoption(
        "--backend",
        action="append",
        default=[],
        help="list of backend to test. Unavailable backend will be skipped.",
    )


def isrunnable(backend, context):
    """Check if test is runnable with the backend and its context."""
    backend_args = context.config.getoption("backend")
    return check_backend(backend) and (backend_args and backend in backend_args)


# # for test directly parametrized by a backend
def pytest_runtest_setup(item):
    """Skip tests based on the backend."""
    if hasattr(item, "callspec"):
        if "backend" in item.callspec.params:
            backend = item.callspec.params["backend"]
            if not isrunnable(backend, item):
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
            if not isrunnable(backend, metafunc):
                callspec.marks.append(
                    pytest.mark.skip(f"Backend {backend} not available.")
                )
