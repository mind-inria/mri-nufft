"""Backend parametrization for the tests."""

import pytest


def param_backends(*names):
    """Return ``backend`` parameters, each marked with its backend.

    The marker lets ``conftest.py`` select or skip the tests by backend.
    """
    return [
        pytest.param(name, marks=pytest.mark.backend(name), id=name) for name in names
    ]
