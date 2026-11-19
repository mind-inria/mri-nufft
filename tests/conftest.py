"""Configuration for pytest.

Tests exercising a NUFFT backend carry a ``backend(name)`` marker, usually set on
the backend parameter with :func:`helpers.param_backends`.

* ``--backend NAME`` keeps only the tests marked for ``NAME``.
* ``-m "not backend"`` keeps only the tests not bound to a backend.
* ``-m gpu`` / ``-m "not gpu"`` keeps or drops the tests needing a GPU.
* Without options, everything runs and unavailable backends are skipped.

Tests parametrized with :func:`helpers.parametrize_array_interface` are deselected
for the array interfaces their backend does not accept.
"""

import pytest
from pytest import fixture, mark

from mrinufft.operators import check_backend, list_backends

from helpers import accepts_interface


def pytest_addoption(parser):
    """Add options to pytest."""
    parser.addoption(
        "--backend",
        action="append",
        default=[],
        help="Only run the tests marked for this NUFFT backend (repeatable).",
    )
    parser.addoption(
        "--ref",
        default="finufft",
        help="Reference backend on which the tests are performed.",
    )


def pytest_configure(config):
    """Register the markers and check the requested backends."""
    for marker in [
        "backend(name): test exercising the NUFFT backend ``name``.",
        "array_interface(name): test fed with arrays from ``name``.",
        "gpu: test needing a GPU.",
    ]:
        config.addinivalue_line("markers", marker)
    for backend in config.getoption("backend"):
        try:
            available = check_backend(backend)
        except ValueError as e:
            raise pytest.UsageError(str(e)) from e
        if not available:
            raise pytest.UsageError(
                f"--backend {backend} was requested, but its dependencies failed "
                "to import."
            )


def pytest_report_header(config):
    """Report the available and requested backends."""
    requested = ", ".join(config.getoption("backend")) or "all"
    return [
        f"backends available: {', '.join(list_backends(available_only=True))}",
        f"backends requested: {requested}",
    ]


@fixture(scope="session")
def ref_backend(request):
    """Reference backend, from ``--ref``."""
    return request.config.getoption("ref")


def _marker_arg(item, name):
    marker = item.get_closest_marker(name)
    return marker.args[0] if marker else None


def pytest_collection_modifyitems(config, items):
    """Deselect the tests of other backends, skip those of unavailable ones."""
    requested = set(config.getoption("backend"))
    selected, deselected = [], []
    for item in items:
        backend = _marker_arg(item, "backend")
        interface = _marker_arg(item, "array_interface")
        if (requested and backend not in requested) or (
            backend and interface and not accepts_interface(backend, interface)
        ):
            deselected.append(item)
            continue
        if backend is not None and not check_backend(backend):
            item.add_marker(mark.skip(reason=f"Backend {backend} not available."))
        selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
