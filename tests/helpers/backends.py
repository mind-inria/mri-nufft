"""Backend and array interface parametrization for the tests."""

import pytest

from .factories import CUPY_AVAILABLE, TORCH_AVAILABLE, TORCH_CUDA_AVAILABLE

ALL_INTERFACES = ("numpy", "cupy", "torch-cpu", "torch-gpu")
CPU_INTERFACES = ("numpy", "torch-cpu")
# Array interfaces accepted by each backend, CPU_INTERFACES if not listed.
BACKEND_INTERFACES = {
    "cufinufft": ALL_INTERFACES,
    "gpunufft": ALL_INTERFACES,
    "torchkbnufft-gpu": ("cupy", "torch-gpu"),
    "tensorflow": ("numpy", "cupy"),
}

_INTERFACE_MARKS = {
    "numpy": [],
    "cupy": [
        pytest.mark.gpu,
        pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy not available"),
    ],
    "torch-cpu": [
        pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available"),
    ],
    "torch-gpu": [
        pytest.mark.gpu,
        pytest.mark.skipif(not TORCH_CUDA_AVAILABLE, reason="torch-gpu not available"),
    ],
}


def param_backends(*names):
    """Return ``backend`` parameters, each marked with its backend.

    The marker lets ``conftest.py`` select or skip the tests by backend.
    """
    return [
        pytest.param(name, marks=pytest.mark.backend(name), id=name) for name in names
    ]


def parametrize_array_interface(*names, argname="array_interface"):
    """Parametrize a test with array interfaces, all of them by default.

    Combinations with a backend that does not accept the interface are deselected by
    ``conftest.py``, and GPU interfaces are marked with ``gpu``.
    """
    return pytest.mark.parametrize(
        argname,
        [
            pytest.param(
                name,
                marks=[pytest.mark.array_interface(name), *_INTERFACE_MARKS[name]],
                id=name,
            )
            for name in (names or ALL_INTERFACES)
        ],
    )


def backend_interfaces(backend):
    """Return the array interfaces accepted by ``backend``."""
    return BACKEND_INTERFACES.get(backend, CPU_INTERFACES)


def accepts_interface(backend, interface):
    """Return True if ``backend`` accepts arrays from ``interface``."""
    return interface in backend_interfaces(backend)
