"""Test for batch computations.

Only finufft and cufinufft support batch computations.
"""
import numpy as np
from pytest_cases import parametrize_with_cases, parametrize, fixture
import pytest

from mrinufft import get_operator
from mrinufft.operators.interfaces import CUFINUFFT_AVAILABLE
from case_trajectories import CasesTrajectories


@fixture(scope="module")
@parametrize(
    backend=[
        "finufft",
        pytest.param(
            "cufinufft",
            marks=pytest.mark.skipif(
                not CUFINUFFT_AVAILABLE, reason="cufinufft not yet implemented"
            ),
        ),
    ]
)
@parametrize(
    "n_coils, n_batch, n_trans",
    [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)],
)
@parametrize_with_cases("kspace_locs, shape", cases=CasesTrajectories)
def operator(
    request, kspace_locs, shape, n_coils=1, n_batch=1, n_trans=1, backend="finufft"
):
    return get_operator(backend)(kspace_locs, shape, n_coils=n_coils, smaps=None)


@fixture(scope="module")
def flat_operator(operator):
    """Generate a batch operator."""
    return get_operator(operator.backend)(
        operators.kspace_locs, operator.shape, n_coils=n_coils
    )


@fixture(scope="module")
def image_data(operator):
    """Generate a random image."""
    img = np.random.rand(operator.n_batch, operator.n_coils, *operator.shape).astype(
        operator.cpx_dtype
    )
    img = 1j * np.random.rand(
        operator.n_batch, operator.n_coils, *operator.shape
    ).astype(operator.cpx_dtype)
    return img


@fixture(scope="module")
def image_data(operator):
    """Generate a random image."""
    img = np.random.rand(operator.n_batch, operator.n_coils, operator.n_samples).astype(
        operator.cpx_dtype
    )
    img = 1j * np.random.rand(
        operator.n_batch, operator.n_coils, *operator.n_samples
    ).astype(operator.cpx_dtype)
    return img


def test_batch_type2(operator, flat_operator, image_data):
    """Test the batch type 1."""
    kspace_data = operator.op(image_data)

    image_flat = image.reshape(-1, operator.shape)
    kspace_flat = [] * operator.n_batch * operator.n_coils
    for i in range():
        kspace_flat[i] = flat_operator.op(image_data[i])

    kspace_flat = np.reshape(
        np.concatenate(kspace_flat, axis=0),
        (operator.n_batch, operator.n_coils, operator.n_samples),
    )

    assert np.allclose(kspace_data, kspace_flat)


def test_batch_type1(operator, flat_operator, kspace_data):
    """Test the batch type 1."""
    image_data = operator.adj_op(kspace_data)

    kspace_flat = kspace_data.reshape(-1, operator.shape)
    image_flat = [] * operator.n_batch * operator.n_coils
    for i in range():
        image_flat[i] = flat_operator.adj_op(kspace_data[i])

    image_flat = np.reshape(
        np.concatenate(image_flat, axis=0),
        (operator.n_batch, operator.n_coils, *operator.shape),
    )

    assert np.allclose(kspace_data, kspace_flat)
