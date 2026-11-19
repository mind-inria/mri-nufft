"""Configuration for pytest.

Tests exercising a NUFFT backend carry a ``backend(name)`` marker, usually set on
the backend parameter with :func:`helpers.param_backends`.

* ``--backend NAME`` keeps only the tests marked for ``NAME``.
* ``-m "not backend"`` keeps only the tests not bound to a backend.
* Without options, everything runs and unavailable backends are skipped.
"""

import pytest
from pytest import fixture, mark

from mrinufft.operators import check_backend, list_backends


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
    """Register the backend marker and check the requested backends."""
    config.addinivalue_line(
        "markers", "backend(name): test exercising the NUFFT backend ``name``."
    )
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


def pytest_collection_modifyitems(config, items):
    """Deselect the tests of other backends, skip those of unavailable ones."""
    requested = set(config.getoption("backend"))
    selected, deselected = [], []
    for item in items:
        marker = item.get_closest_marker("backend")
        backend = marker.args[0] if marker else None
        if requested and backend not in requested:
            deselected.append(item)
            continue
        if backend is not None and not check_backend(backend):
            item.add_marker(mark.skip(reason=f"Backend {backend} not available."))
        selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
