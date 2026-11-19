"""Helper functions for testing the operators."""

from .asserts import assert_almost_allclose, assert_correlate, assert_allclose
from .backends import (
    accepts_interface,
    backend_interfaces,
    param_backends,
    parametrize_array_interface,
)
from .cases import case_id, cases_of
from .factories import (
    kspace_from_op,
    image_from_op,
    batched_smaps_from_op,
    to_interface,
    from_interface,
    CUPY_AVAILABLE,
    TORCH_AVAILABLE,
    TORCH_CUDA_AVAILABLE,
)

__all__ = [
    "assert_almost_allclose",
    "assert_correlate",
    "assert_allclose",
    "accepts_interface",
    "backend_interfaces",
    "param_backends",
    "parametrize_array_interface",
    "case_id",
    "cases_of",
    "kspace_from_op",
    "image_from_op",
    "batched_smaps_from_op",
    "to_interface",
    "from_interface",
    "CUPY_AVAILABLE",
    "TORCH_AVAILABLE",
    "TORCH_CUDA_AVAILABLE",
]
