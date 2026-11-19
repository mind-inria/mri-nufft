"""Helper functions for testing the operators."""

from .asserts import (
    assert_adjoint,
    assert_almost_allclose,
    assert_correlate,
    assert_allclose,
)
from .backends import (
    accepts_interface,
    backend_interfaces,
    param_backends,
    parametrize_array_interface,
)
from .cases import CasesBatch, CasesTrajectories, case_id, cases_of
from .factories import (
    kspace_from_op,
    image_from_op,
    batched_smaps_from_op,
    random_smaps,
    to_interface,
    from_interface,
    CUPY_AVAILABLE,
    TORCH_AVAILABLE,
    TORCH_CUDA_AVAILABLE,
)

__all__ = [
    "assert_adjoint",
    "assert_almost_allclose",
    "assert_correlate",
    "assert_allclose",
    "accepts_interface",
    "backend_interfaces",
    "param_backends",
    "parametrize_array_interface",
    "CasesBatch",
    "CasesTrajectories",
    "case_id",
    "cases_of",
    "kspace_from_op",
    "image_from_op",
    "batched_smaps_from_op",
    "random_smaps",
    "to_interface",
    "from_interface",
    "CUPY_AVAILABLE",
    "TORCH_AVAILABLE",
    "TORCH_CUDA_AVAILABLE",
]
