"""
Test module for the MRI operations.
"""

import pytest
from itertools import product

import numpy as np
import cupy as cp

from mriCufinufft import MRICufiNUFFT
SHAPES = [(512, 512), (32, 32, 32)]
SAMPLING_RATIOS = [0.1, 1, 10]

EPS = [1e-4]

SMAPS = [True, False]
SMAPS_CACHED = [False, True]
N_COILS = [4]


def rand11(size):
    """Return a uniformly random array in [-1,1]."""
    return np.random.uniform(low=-1.0, high=1, size=size)


def product_dict(**kwargs):
    """Transform a dict of list into a list of dict."""
    return [dict(zip(kwargs.keys(), values))
            for values in product(*kwargs.values())]


CONFIG_NO_SMAPS_H = product_dict(
    shape=SHAPES,
    sampling_ratio=SAMPLING_RATIOS,
    data_loc=["H"],
    eps=EPS,
    n_coils=[1, 4]
)

CONFIG_NO_SMAPS_D = product_dict(
    shape=SHAPES,
    sampling_ratio=SAMPLING_RATIOS,
    data_loc=["D"],
    eps=EPS,
    n_coils=[1, 4]
)
CONFIG_WITH_SMAPS = product_dict(
    shape=[SHAPES[1]],
    sampling_ratio=SAMPLING_RATIOS,
    data_loc=["D", "H"],
    eps=EPS,
    smaps=SMAPS,
    smaps_cached=SMAPS_CACHED,
    n_coils=[4]
)
CONFIG = CONFIG_NO_SMAPS_H + CONFIG_NO_SMAPS_D + CONFIG_WITH_SMAPS
CONFIG_H = [c for c in CONFIG if c.get('data_loc') == "H"]


def make_id(val):
    return (f"{val['n_coils']}{val['shape']}-{val['sampling_ratio']}"
            f"-{val['data_loc']}-{val.get('smaps', 0)}"
            f"{val.get('smaps_cached',0)}-{val['eps']:.0e}")


@pytest.fixture()
def mri_op(request):
    n_samples = int(
        np.prod(request.param['shape']) * request.param['sampling_ratio'])
    shape = request.param['shape']
    n_coils = request.param['n_coils']
    if request.param.get('smaps', False):
        smaps = np.random.randn(n_coils, *shape) + 1j * \
            np.random.randn(n_coils, *shape)
        smaps = smaps / np.linalg.norm(smaps, axis=0)
        smaps = smaps.astype(np.complex64)
    else:
        smaps = None
    samples = rand11((n_samples, len(shape))).astype(np.float32) * np.pi
    obj = MRICufiNUFFT(samples, shape, n_coils=n_coils,
                       smaps=smaps,
                       smaps_cached=request.param.get('smaps_cached', False),
                       plan_setup="persist",
                       density=False, eps=request.param['eps'])
    yield obj
    del obj
    del samples
    del smaps


@pytest.fixture()
def image_data(request):
    shape = request.param['shape']
    n_coils = request.param['n_coils']
    if request.param.get('smaps', False):
        n_coils = 1
    img = np.random.randn(n_coils, *shape) + \
        1j * np.random.randn(n_coils, *shape)
    img = np.squeeze(img)
    img = img.astype(np.complex64)
    img = np.ascontiguousarray(img)
    if request.param['data_loc'] == "D":
        img = cp.asarray(img)
        img = cp.ascontiguousarray(img)
    return img


@pytest.fixture()
def kspace_data(request):
    n_s = int(
        np.prod(request.param['shape']) * request.param['sampling_ratio'])
    n_c = request.param['n_coils']
    ksp = np.squeeze(
        np.random.randn(n_c, n_s) + 1j * np.random.randn(n_c, n_s))
    ksp = ksp.astype(np.complex64)
    ksp = np.ascontiguousarray(ksp)
    if request.param['data_loc'] == "D":
        ksp = cp.asarray(ksp)
        ksp = cp.ascontiguousarray(ksp)
    return ksp


@pytest.mark.parametrize("mri_op, image_data",
                         zip(CONFIG_H, CONFIG_H),
                         ids=[make_id(val) for val in CONFIG_H],
                         indirect=True)
def test_op_ok_value(mri_op, image_data, allclose):
    ret = mri_op.op(image_data)
    image_data_d = cp.asarray(image_data)
    ret_d = mri_op.op(image_data_d)
    assert type(ret) == type(image_data)
    assert type(ret_d) == type(image_data_d)
    assert not np.isnan(ret).any()
    assert not cp.isnan(ret_d).any()
    assert allclose(abs(ret), abs(cp.asnumpy(ret_d)),
                    rtol=10 * mri_op.raw_op.eps)


@pytest.mark.parametrize("mri_op, kspace_data",
                         zip(CONFIG_H,
                             CONFIG_H),
                         ids=[make_id(val) for val in CONFIG_H],
                         indirect=True)
def test_adj_op_ok_value(mri_op, kspace_data, allclose):
    ret = mri_op.adj_op(kspace_data)
    kspace_data_d = cp.asarray(kspace_data)
    ret_d = mri_op.adj_op(kspace_data_d)
    assert type(ret) == type(kspace_data)
    assert type(ret_d) == type(kspace_data_d)
    assert not np.isnan(ret).any()
    assert not cp.isnan(ret_d).any()


@pytest.mark.parametrize("mri_op, kspace_data, image_data",
                         zip(CONFIG,
                             CONFIG,
                             CONFIG),
                         ids=[make_id(val) for val in CONFIG],
                         indirect=True)
def test_adjoint_property(mri_op, kspace_data, image_data, allclose):
    """Test the adjoint property of MRICufi in various settings."""
    adjoint = mri_op.adj_op(kspace_data)
    forward = mri_op.op(image_data)
    adjoint = cp.asnumpy(adjoint)
    forward = cp.asnumpy(forward)
    kspace_data = cp.asnumpy(kspace_data)
    image_data = cp.asnumpy(image_data)
    val1 = np.vdot(adjoint, image_data)
    val2 = np.vdot(kspace_data, forward)
    assert allclose(abs(val1), abs(val2), rtol=mri_op.eps)


@pytest.mark.parametrize("mri_op, kspace_data, image_data",
                         zip(CONFIG, CONFIG, CONFIG),
                         ids=[make_id(val) for val in CONFIG], indirect=True)
def test_data_consistency(mri_op, kspace_data, image_data, allclose):
    """Test the data consistency operation in various settings"""
    val2 = mri_op.adj_op(mri_op.op(image_data) - kspace_data)
    val2 = cp.asnumpy(val2)
    val1 = mri_op.data_consistency(image_data, kspace_data)
    assert type(val1) == type(image_data)
    val1 = cp.asnumpy(val1)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("sampling_ratio", SAMPLING_RATIOS)
@pytest.mark.parametrize("eps", EPS)
def test_convergente_density_compensation(shape,
                                          sampling_ratio,
                                          eps, allclose):
    """Test the convergence property of the density compensation."""

    n_samples = int(np.prod(shape) * sampling_ratio)
    samples = rand11((n_samples, len(shape))).astype(np.float32) * np.pi
    density1 = MRICufiNUFFT.estimate_density(
        samples, shape, n_iter=10, eps=eps)
    density2 = MRICufiNUFFT.estimate_density(
        samples, shape, n_iter=10, eps=eps)
    assert allclose(np.sum(cp.asnumpy(density1)), np.sum(cp.asnumpy(density2)), atol=len(density1)*eps)
