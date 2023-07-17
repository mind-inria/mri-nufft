import pytest

from mrinufft.operators import check_backend


def pytest_runtest_setup(item):
    """Skip tests based on the backend."""
    if hasattr(item, "callspec"):
        if "backend" in item.callspec.params:
            backend = item.callspec.params["backend"]
            if not check_backend(backend):
                pytest.skip(f"Backend {backend} not available.")
