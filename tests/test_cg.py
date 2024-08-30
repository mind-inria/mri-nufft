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
from tests.helpers.asserts import assert_almost_allclose


@fixture(scope="module")
@parametrize(
    "backend",
    ["finufft", "torchkbnufft-cpu"],
)
@parametrize_with_cases("kspace_locs, shape", cases=CasesTrajectories)
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


@param_array_interface
def test_cg(operator, array_interface, image_data):
    """Compare the interface to the raw NUDFT implementation."""
    kspace_nufft = operator.op(image_data).squeeze()

    image_cg = operator.cg(kspace_nufft)
    kspace_cg = operator.op(image_cg).squeeze()

    assert_almost_allclose(
        kspace_nufft,
        kspace_cg,
        atol=1e-1,
        rtol=1e-1,
        mismatch=20,
    )
