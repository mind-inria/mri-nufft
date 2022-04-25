"""
Test module for the MRI operations.
"""

import pytest
from itertools import product

import numpy as np
import pycuda.gpuarray as gp
import pycuda.driver as cuda
import pycuda.autoinit

from mriCufinufft import MRICufiNUFFT
from mriCufinufft.kernels import diff_in_place
from mriCufinufft.utils import is_cuda_array

SHAPES = [(512, 512), (32,32,32)]
SAMPLING_RATIOS = [0.1, 1, 10]

DATA_LOCS = ["D", "H"]
EPS = [1e-5]


SMAPS_LOCS = ["N", "H", "D"]
SMAPS_CACHED = [True, False]
N_COILS = [4]

np.random.seed(42)

def rand11(size):
    return np.random.uniform(low=-1.0, high=1, size=size)

def product_dict(**kwargs):
    return [dict(zip(kwargs.keys(), values)) for values in product(*kwargs.values())]

MRI_OP_SETUP_MONO = product_dict(shape=SHAPES,
                                 sampling_ratio=SAMPLING_RATIOS,
                                 data_loc=DATA_LOCS,
                                 eps=EPS,
                                 n_coils=[1]
                                 )
MRI_OP_MONO_H = product_dict(shape=SHAPES,
                                 sampling_ratio=SAMPLING_RATIOS,
                                 data_loc=["H"],
                                 eps=EPS,
                                 n_coils=[1]
                                 )

MRI_OP_MONO_D = product_dict(shape=SHAPES,
                                 sampling_ratio=SAMPLING_RATIOS,
                                 data_loc=["D"],
                                 eps=EPS,
                                 n_coils=[1]
                                 )

MRI_OP_SETUP_MULTI = product_dict(shape=SHAPES,
                                  sampling_ratio=SAMPLING_RATIOS,
                                  data_loc=DATA_LOCS,
                                  eps=EPS,
                                  smaps_loc=SMAPS_LOCS,
                                  smaps_cached=SMAPS_CACHED,
                                  n_coils=N_COILS
                                  )
def make_id(val):
    return f"{val['n_coils']}{val['shape']}-{val['sampling_ratio']}-{val['data_loc']}-{val['eps']:.0e}"


def get_mri_op(param):
    n_samples = int(np.prod(param['shape']) * param['sampling_ratio'])
    shape = param['shape']
    n_coils=param['n_coils']
    samples = rand11((n_samples, len(shape))).astype(np.float32) * np.pi
    obj = MRICufiNUFFT(samples, shape,
                       n_coils=n_coils,
                       smaps=None,
                       density=False,
                       eps=param['eps'])

    return obj

def get_kspace_data(param):
    n_samples = int(np.prod(param['shape']) * param['sampling_ratio'])
    n_coils = param['n_coils']
    kspace_data = np.squeeze(rand11((n_coils, n_samples))
                              + 1j * rand11((n_coils, n_samples)))
    kspace_data = kspace_data.astype(np.complex64)
    if param['data_loc'] == "D":
        kspace_data = gp.to_gpu(kspace_data)
    return kspace_data

def get_image_data(param):
    shape = param['shape']
    image_data = rand11(shape) + 1j * rand11(shape)
    image_data = image_data.astype(np.complex64)
    if param['data_loc'] == "D":
        image_data = gp.to_gpu(image_data)
    return image_data


@pytest.mark.parametrize("param",
                         MRI_OP_MONO_H,
                         ids=[make_id(val) for val in MRI_OP_MONO_H],
                         )
def test_op_ok_value(param):

    import pycuda.autoinit
    mri_op = get_mri_op(param)
    image_data = get_image_data(param)
    ret = mri_op.op(image_data)
    image_data_d = gp.to_gpu(image_data)
    ret_d = mri_op.op(image_data_d)
    ret_d_back = ret_d.get()
    assert type(ret) == type(image_data)
    assert type(ret_d) == type(image_data_d)
    np.testing.assert_allclose(abs(ret_d), abs(ret))

@pytest.mark.parametrize("param",
                         MRI_OP_MONO_H,
                         ids=[make_id(val) for val in MRI_OP_MONO_H],
                         )
def test_adj_op_ok_value(param):
    import pycuda.autoinit
    mri_op = get_mri_op(param)
    kspace_data = get_kspace_data(param)
    ret = mri_op.adj_op(kspace_data)
    kspace_data_d = gp.to_gpu(kspace_data)
    ret_d = mri_op.adj_op(kspace_data_d)
    assert type(ret) == type(kspace_data)
    assert type(ret_d) == type(kspace_data_d)
    ret_d_back = ret_d.get()
    assert sum(abs(ret_d_back-ret).flatten())/ np.prod(param['shape']) < 1e-5


@pytest.mark.parametrize("param",
                         MRI_OP_MONO_H,
                         ids=[make_id(val) for val in MRI_OP_MONO_H],
                         )
def test_adjoint_property(param):
    """Test the adjoint property of MRICufi in various settings."""
    import pycuda.autoinit
    mri_op = get_mri_op(param)
    kspace_data = get_kspace_data(param)
    image_data = get_image_data(param)
    forward = mri_op.op(image_data)
    adjoint = mri_op.adj_op(kspace_data)
    val1 = np.vdot(adjoint.flatten(), image_data.flatten())
    val2 = np.vdot(kspace_data.flatten(), forward.flatten())
    np.testing.assert_allclose(val1, val2)

# @pytest.mark.parametrize("mri_op, kspace_data, image_data",
#                          zip(MRI_OP_SETUP_MONO, MRI_OP_SETUP_MONO, MRI_OP_SETUP_MONO),
#                          ids=[make_id(val) for val in MRI_OP_SETUP_MONO], indirect=True)
# def test_data_consistency(mri_op, kspace_data, image_data):
#     """Test the data consistency operation in various settings"""

#     val1 = mri_op.data_consistency(image_data, kspace_data)
#     if is_cuda_array(val1):
#         val1 = val1.get()
#         x = mri_op.op(image_data)
#         diff_in_place(x, kspace_data)
#         val2 = mri_op.adj_op(x)
#         val2 = val2.get()
#     else:
#         val2 = mri_op.adj_op(mri_op.op(image_data)-kspace_data)
#     np.testing.assert_allclose(val1, val2)

    # def test_convergente_density_compensation(shape, sampling_ratio, samples_loc, eps):
    #     """Test the convergence property of the density compensation."""

    #     n_samples = int(np.prod(shape)*sampling_ratio)
    #     samples = (np.random.rand(n_samples, len(shape)).astype(np.float32) * 2 - 1) * np.pi
    #     if samples_loc == "device":
    #         samples = gp.to_gpu(samples)
    #     density1 = MRICufiNUFFT.estimate_density(samples, shape, n_iter=40, eps=eps).get()
    #     density2 = MRICufiNUFFT.estimate_density(samples, shape, n_iter=60, eps=eps).get()

    #     np.testing.assert_allclose(density1, density2, atol=10*eps)
