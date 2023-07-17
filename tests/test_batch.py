"""Test for batch computations.

Only finufft and cufinufft support batch computations.
"""
import numpy as np
import numpy.testing as npt
from pytest_cases import parametrize_with_cases, parametrize, fixture
import pytest

from mrinufft import get_operator
from case_trajectories import CasesTrajectories


@fixture(scope="module")
@parametrize(
    "n_coils, n_batch, n_trans",
    [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)],
)
@parametrize_with_cases("kspace_locs, shape", cases=CasesTrajectories)
@parametrize(backend=["finufft", "cufinufft"])
def operator(
    request, kspace_locs, shape, n_coils=1, n_batch=1, n_trans=1, backend="finufft"
):
    """Generate a batch operator."""
    return get_operator(backend)(kspace_locs, shape, n_coils=n_coils, smaps=None)


@fixture(scope="module")
def flat_operator(operator):
    """Generate a batch operator with n_batch=1."""
    return get_operator(operator.backend)(
        operator.samples, operator.shape, n_coils=operator.n_coils
    )


@fixture(scope="module")
def image_data(operator):
    """Generate a random image."""
    shape = (operator.n_batchs, operator.n_coils, *operator.shape)
    img = np.random.rand(*shape).astype(operator.cpx_dtype)
    img = 1j * np.random.rand(*shape).astype(operator.cpx_dtype)
    return img


@fixture(scope="module")
def kspace_data(operator):
    """Generate a random image."""
    shape = (operator.n_batchs, operator.n_coils, operator.n_samples)
    img = np.random.rand(*shape).astype(operator.cpx_dtype)
    img = 1j * np.random.rand(*shape).astype(operator.cpx_dtype)
    return img


def test_batch_type2(operator, flat_operator, image_data):
    """Test the batch type 1."""
    kspace_data = operator.op(image_data)

    image_flat = image_data.reshape(-1, operator.n_coils, *operator.shape)
    kspace_flat = [None] * operator.n_batchs
    for i in range(len(kspace_flat)):
        kspace_flat[i] = flat_operator.op(image_flat[i])

    kspace_flat = np.reshape(
        np.concatenate(kspace_flat, axis=0),
        (operator.n_batchs, operator.n_coils, operator.n_samples),
    )

    npt.assert_array_almost_equal_nulp(kspace_data, kspace_flat)


def test_batch_type1(operator, flat_operator, kspace_data):
    """Test the batch type 1."""
    image_data = operator.adj_op(kspace_data)

    kspace_flat = kspace_data.reshape(-1, operator.n_coils, operator.n_samples)
    image_flat = [None] * operator.n_batchs
    for i in range(len(image_flat)):
        image_flat[i] = flat_operator.adj_op(kspace_flat[i])

    image_flat = np.reshape(
        np.concatenate(image_flat, axis=0),
        (operator.n_batchs, operator.n_coils, *operator.shape),
    )

    npt.assert_allclose(image_data, image_flat, atol=1e-5, rtol=1e-4)
