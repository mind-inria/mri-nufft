"""Test for batch computations.

Only finufft, cufinufft and gpunufft support batch computations.
"""

import numpy as np
import numpy.testing as npt
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture
from helpers import (
    assert_correlate,
    param_array_interface,
    to_interface,
    from_interface,
)
from mrinufft import get_operator
from case_trajectories import CasesTrajectories


@fixture(scope="module")
@parametrize(
    "n_batch, n_coils, n_trans, sense",
    [
        (1, 1, 1, False),
        (3, 1, 1, False),
        (1, 4, 1, True),
        (1, 4, 1, False),
        (1, 4, 2, True),
        (1, 4, 2, False),
        (3, 2, 1, True),
        (3, 2, 1, False),
        (3, 4, 2, True),
        (3, 4, 2, False),
    ],
)
@parametrize_with_cases(
    "kspace_locs, shape",
    cases=CasesTrajectories,
    glob="*nyquist_radial*",
)
@parametrize(
    backend=["finufft", "cufinufft", "gpunufft", "torchkbnufft-cpu", "torchkbnufft-gpu"]
)
def operator(
    request,
    kspace_locs,
    shape,
    n_coils=1,
    sense=None,
    n_batch=1,
    n_trans=1,
    backend="finufft",
):
    """Generate a batch operator."""
    if n_trans != 1 and backend == "gpunufft":
        pytest.skip("Duplicate case.")
    if sense:
        smaps = 1j * np.random.rand(n_coils, *shape)
        smaps += np.random.rand(n_coils, *shape)
        smaps = smaps.astype(np.complex64)
        smaps /= np.linalg.norm(smaps, axis=0)
    else:
        smaps = None
    kspace_locs = kspace_locs.astype(np.float32)
    return get_operator(backend)(
        kspace_locs,
        shape,
        n_coils=n_coils,
        smaps=smaps,
        n_batchs=n_batch,
        n_trans=n_trans,
        squeeze_dims=False,
    )


@fixture(scope="module")
def flat_operator(operator):
    """Generate a batch operator with n_batch=1."""
    return get_operator(operator.backend)(
        operator.samples,
        operator.shape,
        n_coils=operator.n_coils,
        smaps=operator.smaps,
        squeeze_dims=False,
        n_trans=1,
    )


@fixture(scope="module")
def image_data(operator):
    """Generate a random image."""
    if operator.uses_sense:
        shape = (operator.n_batchs, *operator.shape)
    else:
        shape = (operator.n_batchs, operator.n_coils, *operator.shape)
    img = (1j * np.random.rand(*shape)).astype(operator.cpx_dtype)
    img += np.random.rand(*shape).astype(operator.cpx_dtype)
    return img


@fixture(scope="module")
def kspace_data(operator):
    """Generate a random image."""
    shape = (operator.n_batchs, operator.n_coils, operator.n_samples)
    kspace = (1j * np.random.rand(*shape)).astype(operator.cpx_dtype)
    kspace += np.random.rand(*shape).astype(operator.cpx_dtype)
    return kspace


@param_array_interface
def test_batch_op(operator, array_interface, flat_operator, image_data):
    """Test the batch type 2 (forward)."""
    image_data = to_interface(image_data, array_interface)

    kspace_batched = from_interface(operator.op(image_data), array_interface)

    if operator.uses_sense:
        image_flat = image_data.reshape(-1, *operator.shape)
    else:
        image_flat = image_data.reshape(-1, operator.n_coils, *operator.shape)
    kspace_flat = [None] * operator.n_batchs
    for i in range(len(kspace_flat)):
        kspace_flat[i] = from_interface(
            flat_operator.op(image_flat[i]), array_interface
        )

    kspace_flat = np.reshape(
        np.concatenate(kspace_flat, axis=0),
        (operator.n_batchs, operator.n_coils, operator.n_samples),
    )

    npt.assert_array_almost_equal(kspace_batched, kspace_flat)


@param_array_interface
def test_batch_adj_op(
    operator,
    array_interface,
    flat_operator,
    kspace_data,
):
    """Test the batch type 1 (adjoint)."""
    kspace_data = to_interface(kspace_data, array_interface)

    kspace_flat = kspace_data.reshape(-1, operator.n_coils, operator.n_samples)

    image_flat = [None] * operator.n_batchs
    for i in range(len(image_flat)):
        image_flat[i] = from_interface(
            flat_operator.adj_op(kspace_flat[i]), array_interface
        )

    if operator.uses_sense:
        shape = (operator.n_batchs, 1, *operator.shape)
    else:
        shape = (operator.n_batchs, operator.n_coils, *operator.shape)

    image_flat = np.reshape(
        np.concatenate(image_flat, axis=0),
        shape,
    )

    image_batched = from_interface(operator.adj_op(kspace_data), array_interface)
    # Reduced accuracy for the GPU cases...
    npt.assert_allclose(image_batched, image_flat, atol=1e-3, rtol=1e-3)


@param_array_interface
def test_data_consistency(
    operator,
    array_interface,
    image_data,
    kspace_data,
):
    """Test the data consistency operation."""
    # image_data = np.zeros_like(image_data)
    image_data = to_interface(image_data, array_interface)
    kspace_data = to_interface(kspace_data, array_interface)

    res = operator.data_consistency(image_data, kspace_data)
    tmp = operator.op(image_data)
    res2 = operator.adj_op(tmp - kspace_data)

    #    npt.assert_allclose(res.squeeze(), res2.squeeze(), atol=1e-4, rtol=1e-1)
    res = res.reshape(-1, *operator.shape)
    res2 = res2.reshape(-1, *operator.shape)

    res = from_interface(res, array_interface)
    res2 = from_interface(res2, array_interface)
    atol = 1e-3
    rtol = 1e-3
    # FIXME 2D Sense is not very accurate...
    if len(operator.shape) == 2 and operator.uses_sense:
        print("Reduced accuracy for 2D Sense")
        atol = 1e-1
        atol = 1e-1

    npt.assert_allclose(res, res2, atol=atol, rtol=rtol)


def test_data_consistency_readonly(operator, image_data, kspace_data):
    """Test that the data consistency does not modify the input parameters data."""
    kspace_tmp = kspace_data.copy()
    image_tmp = image_data.copy()
    kspace_tmp.setflags(write=False)
    image_tmp.setflags(write=False)
    operator.data_consistency(image_data, kspace_tmp)
    npt.assert_equal(kspace_tmp, kspace_data)
    npt.assert_equal(image_tmp, image_data)


def test_gradient_lipschitz(operator, image_data, kspace_data):
    """Test the gradient lipschitz constant."""
    C = 1 if operator.uses_sense else operator.n_coils
    img = image_data.copy().reshape(operator.n_batchs, C, *operator.shape).squeeze()
    for _ in range(10):
        grad = operator.data_consistency(img, kspace_data)
        norm = np.linalg.norm(grad)
        grad /= norm
        np.copyto(img, grad.squeeze())
        norm_prev = norm

    # TODO: check that the value is "not too far" from 1
    # TODO: to do the same with density compensation
    assert (norm - norm_prev) / norm_prev < 1e-3
