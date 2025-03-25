"""Test NUFFT with simultaneous subspace projection."""

from itertools import product

import numpy as np
import numpy.testing as npt

from pytest_cases import parametrize_with_cases, parametrize, fixture
from helpers import (
    param_array_interface,
    to_interface,
    from_interface,
)
from case_trajectories import case_multicontrast2D

from mrinufft import get_operator
from mrinufft.operators import MRISubspace


@fixture(scope="module")
@parametrize(
    "n_coils, n_batchs, sense, use_gpu",
    list(product([1, 2, 4], [1, 3], [False, True], [False, True])),
)
@parametrize_with_cases("kspace_locs, shape", cases=[case_multicontrast2D])
@parametrize(
    backend=["finufft", "cufinufft", "gpunufft", "torchkbnufft-cpu", "torchkbnufft-gpu"]
)
def operator(
    request,
    kspace_locs,
    shape,
    n_coils=1,
    n_batchs=1,
    sense=None,
    use_gpu=None,
    backend="finufft",
):
    """Generate a subspace operator."""
    if sense:
        n_coils = 2
        smaps = 1j * np.random.rand(n_coils, *shape)
        smaps += np.random.rand(n_coils, *shape)
        smaps = smaps.astype(np.complex64)
        smaps /= np.linalg.norm(smaps, axis=0)
    else:
        smaps = None

    kspace_locs = kspace_locs.astype(np.float32)

    _op = get_operator(backend)(
        kspace_locs,
        shape,
        n_coils=n_coils,
        n_batchs=n_batchs,
        smaps=smaps,
        squeeze_dims=False,
    )

    # generate random orthonormal basis
    train_data = np.exp(-4 * np.arange(48)[:, None] / np.arange(1, 1000, 1))
    _, _, _basis = np.linalg.svd(train_data.T, full_matrices=False)
    _basis = _basis[:5]
    _basis = _basis.astype(_op.cpx_dtype)

    # get reference operator
    ref_operator = [
        get_operator(backend)(
            samples=kspace_locs[n],
            shape=shape,
            n_coils=n_coils,
            n_batchs=n_batchs,
            smaps=smaps,
            squeeze_dims=False,
        )
        for n in range(_basis.shape[-1])
    ]

    return MRISubspace(_op, _basis, use_gpu), ref_operator, _basis


@fixture(scope="module")
def image_data(operator):
    """Generate a random image."""
    _operator = operator[0]
    _shape = _operator.shape
    if _operator._fourier_op.uses_sense:
        shape = (_operator.n_batchs, *_shape)
    else:
        shape = (_operator.n_batchs, _operator.n_coils, *_shape)

    img = (1j * np.zeros(shape)).astype(_operator._fourier_op.cpx_dtype)
    img += np.zeros(shape).astype(_operator._fourier_op.cpx_dtype)
    img[..., 32, 32] = 1.0

    sig = np.exp(-4 * np.arange(48) / 200.0)
    weights = _operator.subspace_basis @ sig
    image = np.stack([weights[n] * img for n in range(_operator.n_coeffs)], axis=0)
    return np.ascontiguousarray(image.swapaxes(0, 1))


@fixture(scope="module")
def kspace_data(operator):
    """Generate a random image."""
    _operator = operator[0]
    n_samples = int(_operator.n_samples / _operator.n_frames)
    shape = (_operator.n_batchs, _operator.n_coils, n_samples)
    kspace = (1j * np.zeros(shape)).astype(_operator._fourier_op.cpx_dtype)
    kspace += np.ones(shape).astype(_operator._fourier_op.cpx_dtype)

    sig = np.exp(-4 * np.arange(48) / 200.0)
    kspace = sig[:, None, None, None] * kspace[None, ...]
    return np.ascontiguousarray(kspace.swapaxes(0, 1))


@param_array_interface
def test_subspace_op(operator, array_interface, image_data):
    subspace_op, ref_op, subspace_basis = operator

    # get reference
    # 1. back-project to time/contrast-domain image space
    tmp = image_data.swapaxes(0, 1)
    tmp = tmp[..., None].swapaxes(0, -1)[0, ...]
    tmp = tmp @ subspace_basis.conj()
    tmp = tmp[None, ...].swapaxes(0, -1)[..., 0]

    # 2. apply NUFFT
    tmp = [ref_op[n].op(tmp[n]) for n in range(tmp.shape[0])]
    tmp = np.stack(tmp, axis=0)
    kspace_ref = tmp.swapaxes(0, 1)

    # actual computation
    image_data = to_interface(image_data, array_interface)
    kspace = from_interface(subspace_op.op(image_data), array_interface)

    npt.assert_allclose(kspace, kspace_ref, rtol=1e-3, atol=1e-3)


@param_array_interface
def test_subspace_op_adj(operator, array_interface, kspace_data):
    subspace_op, ref_op, subspace_basis = operator

    # get reference
    # 1. apply adjoint NUFFT
    tmp = kspace_data.swapaxes(0, 1)
    tmp = [ref_op[n].adj_op(tmp[n]) for n in range(tmp.shape[0])]

    # 2. project to subspace image space
    tmp = np.stack(tmp, axis=-1)
    tmp = tmp @ subspace_basis.T
    tmp = tmp[None, ...].swapaxes(0, -1)[..., 0]
    image_ref = tmp.swapaxes(0, 1)

    # actual computation
    kspace_data = to_interface(kspace_data, array_interface)
    image = from_interface(subspace_op.adj_op(kspace_data), array_interface)

    npt.assert_allclose(image, image_ref, rtol=1e-3, atol=1e-3)


@param_array_interface
def test_data_consistency(
    operator,
    array_interface,
    image_data,
    kspace_data,
):
    """Test the data consistency operation."""
    subspace_op, _, _ = operator
    image_data = to_interface(image_data, array_interface)
    kspace_data = to_interface(kspace_data, array_interface)

    res = subspace_op.data_consistency(image_data, kspace_data)
    tmp = subspace_op.op(image_data)
    res2 = subspace_op.adj_op(tmp - kspace_data)

    res = res.reshape(-1, *subspace_op.shape)
    res2 = res2.reshape(-1, *subspace_op.shape)

    res = from_interface(res, array_interface)
    res2 = from_interface(res2, array_interface)
    atol = 1e-3
    rtol = 1e-3
    # FIXME 2D Sense is not very accurate...
    if len(subspace_op.shape) == 2 and subspace_op._fourier_op.uses_sense:
        print("Reduced accuracy for 2D Sense")
        atol = 1e-1
        atol = 1e-1

    npt.assert_allclose(res, res2, atol=atol, rtol=rtol)


def test_data_consistency_readonly(operator, image_data, kspace_data):
    """Test that the data consistency does not modify the input parameters data."""
    subspace_op, _, _ = operator
    kspace_tmp = kspace_data.copy()
    image_tmp = image_data.copy()
    kspace_tmp.setflags(write=False)
    image_tmp.setflags(write=False)
    subspace_op.data_consistency(image_data, kspace_tmp)
    npt.assert_equal(kspace_tmp, kspace_data)
    npt.assert_equal(image_tmp, image_data)


# def test_gradient_lipschitz(operator, image_data, kspace_data):
#     """Test the gradient lipschitz constant."""
#     subspace_op, _, _ = operator

#     img = image_data.copy()
#     kspace = kspace_data.copy()

#     for _ in range(10):
#         grad = subspace_op.data_consistency(img, kspace)
#         norm = np.linalg.norm(grad)
#         grad /= norm
#         np.copyto(img, grad.reshape(*img.shape))
#         norm_prev = norm

#     # TODO: check that the value is "not too far" from 1
#     # TODO: to do the same with density compensation
#     assert (norm - norm_prev) / norm_prev < 1e-3
