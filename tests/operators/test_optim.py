"""Test for the cg function."""

import numpy as np
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture
from mrinufft import get_operator
from case_trajectories import CasesTrajectories

from helpers import (
    image_from_op,
    param_array_interface,
)
from helpers import assert_almost_allclose

from mrinufft.extras.optim import loss_l2_reg


@fixture(scope="module")
@parametrize(
    "backend",
    [
        "bart",
        "finufft",
        "cufinufft",
        "gpunufft",
        "sigpy",
        "torchkbnufft-cpu",
        "torchkbnufft-gpu",
        "tensorflow",
    ],
)
@parametrize_with_cases(
    "kspace_locs, shape",
    cases=[
        CasesTrajectories.case_random2D,
        # CasesTrajectories.case_grid2D,
        # CasesTrajectories.case_grid3D,
    ],
)
def operator(
    request,
    backend="pynfft",
    kspace_locs=None,
    shape=None,
    n_coils=1,
):
    """Generate an operator."""
    if backend in ["pynfft", "sigpy"] and kspace_locs.shape[-1] == 3:
        pytest.skip("3D for slow cpu is not tested")
    return get_operator(backend)(kspace_locs, shape, n_coils=n_coils, smaps=None)


@fixture(scope="module")
def image_data(operator):
    """Generate a random image. Remains constant for the module."""
    return image_from_op(operator)


@parametrize("optim", ["lsqr", "lsmr", "cg"])
def test_pinv(operator, image_data, optim):
    """Compare the interface to the raw NUDFT implementation."""
    kspace_nufft = operator.op(image_data).squeeze()

    _, residuals = operator.pinv_solver(
        kspace_nufft, optim=optim, n_iter=10, callback=loss_l2_reg
    )
    assert residuals[-1] <= residuals[0]
