"""Configuration for pytest."""

import pytest

from mrinufft.operators import FourierOperatorBase, check_backend, list_backends


def pytest_addoption(parser):
    """Add options to pytest."""
    parser.addoption(
        "--backend",
        action="append",
        default=[],
        help="NUFFT backend on which the tests are performed.",
    )
    parser.addoption(
        "--ref",
        default="finufft",
        help="Reference backend on which the tests are performed.",
    )


def pytest_configure(config):
    """Configure hook for pytest."""
    available = {b: FourierOperatorBase.interfaces[b][0] for b in list_backends()}

    if selected := config.getoption("backend"):
        # hijacks the availability of interfaces:
        for backend in list_backends():
            FourierOperatorBase.interfaces[backend] = (
                backend in selected,
                FourierOperatorBase.interfaces[backend][1],
            )
    # ensure the ref backend is available
    ref_backend = config.getoption("ref")
    FourierOperatorBase.interfaces[ref_backend] = (
        True,
        FourierOperatorBase.interfaces[ref_backend][1],
    )
    selected = {b: FourierOperatorBase.interfaces[b][0] for b in list_backends()}

    available[ref_backend] = "REF"
    selected[ref_backend] = "REF"
    print(f"{'backends':>20}: {'avail':>5} {'select':<5}")
    for b in list_backends():
        print(f"{b:>20}: {str(available[b]):>5} {str(selected[b]):>5}")


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
            try:
                op_call = callspec.params["operator"]
            except KeyError:
                continue
            # Complicated datastructure
            # Acces the value for the backend parameter.
            backend = op_call.argvalues[
                [
                    i
                    for i, v in enumerate(op_call.param_defs)
                    if v.argnames[0] == "backend"
                ][0]
            ]
            print("backend detected", backend)
            # Only keep the callspec if the backend is available.
            if not check_backend(backend):
                callspec.marks.append(
                    pytest.mark.skip(f"Backend {backend} not available.")
                )
            if (
                metafunc.config.getoption("backend") != []
                and backend != metafunc.config.getoption("backend")[0]
            ):
                # Skip tests if the backend does not match what we want to test.
                callspec.marks.append(
                    pytest.mark.skip(f"Not testing {backend} as not requested")
                )
