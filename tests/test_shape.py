"""Test for the check_shape function."""

import numpy as np
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture
from mrinufft import get_operator
from case_trajectories import CasesTrajectories

from mrinufft.operators.interfaces.utils.utils import check_shape_op, check_shape_adj_op
from helpers import (
    wrong_image_from_op,
    wrong_kspace_from_op,
    kspace_from_op,
    image_from_op,
    to_interface,
    from_interface,
    param_array_interface,
)


@fixture(scope="module")
@parametrize(
    "backend",
    [
        "torchkbnufft-gpu",
    ],
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
def test_check_shape_op_pass(operator, array_interface, image_data):
    """Compare the interface to the raw NUDFT implementation."""
    image_data_ = to_interface(image_data, array_interface)
    try:
        check_shape_op(operator, image_data_)
    except ValueError:
        pytest.fail("Unexpected ValueError raised")


@param_array_interface
def test_check_shape_adj_op_pass(operator, array_interface, kspace_data):
    """Compare the interface to the raw NUDFT implementation."""
    kspace_data_ = to_interface(kspace_data, array_interface)
    try:
        check_shape_adj_op(operator, kspace_data_)
    except ValueError:
        pytest.fail("Unexpected ValueError raised")


@fixture(scope="module")
def wrong_image_data(operator):
    """Generate a random image. Remains constant for the module."""
    return wrong_image_from_op(operator)


@fixture(scope="module")
def wrong_kspace_data(operator):
    """Generate a random image. Remains constant for the module."""
    return wrong_kspace_from_op(operator)


@param_array_interface
def test_check_shape_op_fail(operator, array_interface, wrong_image_data):
    """Compare the interface to the raw NUDFT implementation."""
    image_data_ = to_interface(wrong_image_data, array_interface)
    with pytest.raises(ValueError):
        check_shape_op(operator, image_data_)


@param_array_interface
def test_check_shape_adj_op_fail(operator, array_interface, wrong_kspace_data):
    """Compare the interface to the raw NUDFT implementation."""
    kspace_data_ = to_interface(wrong_kspace_data, array_interface)
    with pytest.raises(ValueError):
        check_shape_adj_op(operator, kspace_data_)
