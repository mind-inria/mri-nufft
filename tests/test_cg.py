"""Test for the cg function."""

import numpy as np
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture
from mrinufft import get_operator
from case_trajectories import CasesTrajectories

from helpers import (
    kspace_from_op,
    image_from_op,
    to_interface,
    from_interface,
    param_array_interface,
)


@fixture(scope="module")
@parametrize(
    "backend",
    ["torchkbnufft-gpu"],
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


@fixture(scope="module")
def kspace_data(operator):
    """Generate a random kspace. Remains constant for the module."""
    return kspace_from_op(operator)


@param_array_interface
def test_cg(operator, array_interface, image_data):
    """Compare the interface to the raw NUDFT implementation."""
    image_data_ = to_interface(image_data, array_interface)
    kspace_nufft = operator.op(image_data_).squeeze()

    image_cg = operator.cg(kspace_nufft)
    kspace_cg = operator.op(image_cg).squeeze()
    assert np.allclose(kspace_nufft, kspace_cg, atol=1e-5, rtol=1e-5)

