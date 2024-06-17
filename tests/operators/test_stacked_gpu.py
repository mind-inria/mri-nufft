"""Test for the stacked NUFFT operator.

The tests compares the stacked NUFFT (which uses FFT in the z-direction)
and the fully 3D ones.
"""

import numpy as np

import numpy.testing as npt
from pytest_cases import parametrize_with_cases, parametrize, fixture
from helpers import param_array_interface, from_interface, to_interface
from mrinufft.operators.stacked import (
    MRIStackedNUFFTGPU,
    stacked2traj3d,
    traj3d2stacked,
)
from mrinufft import get_operator
from case_trajectories import CasesTrajectories


@fixture(scope="module")
@parametrize(
    "n_batchs, n_coils, sense",
    [(1, 1, False), (1, 4, False), (1, 4, True), (3, 4, False), (3, 4, True)],
)
@parametrize("z_index", ["full", "random_mask"])
@parametrize("backend", ["cufinufft"])
@parametrize_with_cases("kspace_locs, shape", cases=CasesTrajectories, glob="*2D")
def operator(request, backend, kspace_locs, shape, z_index, n_batchs, n_coils, sense):
    """Initialize the stacked and non-stacked operators."""
    shape3d = (*shape, shape[-1] - 2)  # add a 3rd dimension

    if z_index == "full":
        z_index = np.arange(shape3d[-1])
        z_index_ = z_index
    elif z_index == "random_mask":
        z_index = np.random.rand(shape3d[-1]) > 0.5
        z_index_ = np.arange(shape3d[-1])[z_index]

    kspace_locs3d = stacked2traj3d(kspace_locs, z_index_, shape3d[-1])
    # smaps support
    if sense:
        smaps = 1j * np.random.rand(n_coils, *shape3d)
        smaps += np.random.rand(n_coils, *shape3d)
    else:
        smaps = None

    # Setup the operators
    ref = get_operator(backend)(
        kspace_locs3d,
        shape=shape3d,
        n_coils=n_coils,
        n_batchs=n_batchs,
        smaps=smaps,
    )

    stacked = MRIStackedNUFFTGPU(
        samples=kspace_locs,
        shape=shape3d,
        z_index=z_index,
        n_coils=n_coils,
        n_trans=2 if n_coils > 1 else 1,
        n_batchs=n_batchs,
        smaps=smaps,
    )
    return stacked, ref


@fixture(scope="module")
def stacked_op(operator):
    """Return operator."""
    return operator[0]


@fixture(scope="module")
def ref_op(operator):
    """Return ref operator."""
    return operator[1]


@fixture(scope="module")
def image_data(stacked_op):
    """Generate a random image."""
    B, C = stacked_op.n_batchs, stacked_op.n_coils
    if stacked_op.smaps is None:
        img = np.random.randn(B, C, *stacked_op.shape).astype(
            stacked_op.cpx_dtype, copy=False
        )
    elif stacked_op.smaps is not None and stacked_op.n_coils > 1:
        img = np.random.randn(B, *stacked_op.shape).astype(
            stacked_op.cpx_dtype, copy=False
        )

    img += 1j * np.random.randn(*img.shape).astype(stacked_op.cpx_dtype, copy=False)
    return img


@fixture(scope="module")
def kspace_data(stacked_op):
    """Generate a random kspace data."""
    B, C = stacked_op.n_batchs, stacked_op.n_coils
    kspace = (1j * np.random.randn(B, C, stacked_op.n_samples)).astype(
        stacked_op.cpx_dtype, copy=False
    )
    kspace += np.random.randn(B, C, stacked_op.n_samples).astype(
        stacked_op.cpx_dtype, copy=False
    )
    return kspace


@param_array_interface
def test_stack_forward(operator, array_interface, ref_op, stacked_op, image_data):
    """Compare the stack interface to the 3D NUFFT implementation."""
    image_data_ = to_interface(image_data, array_interface)
    kspace_nufft_ = stacked_op.op(image_data_).squeeze()
    kspace_ref_ = ref_op.op(image_data_).squeeze()
    kspace_nufft = from_interface(kspace_nufft_, array_interface)
    kspace_ref = from_interface(kspace_ref_, array_interface)
    npt.assert_allclose(kspace_nufft, kspace_ref, atol=1e-4, rtol=1e-1)


@param_array_interface
def test_stack_backward(operator, array_interface, stacked_op, ref_op, kspace_data):
    """Compare the stack interface to the 3D NUFFT implementation."""
    kspace_data_ = to_interface(kspace_data, array_interface)
    image_nufft_ = stacked_op.adj_op(kspace_data_).squeeze()
    image_ref_ = ref_op.adj_op(kspace_data_).squeeze()
    image_nufft = from_interface(image_nufft_, array_interface)
    image_ref = from_interface(image_ref_, array_interface)
    npt.assert_allclose(image_nufft, image_ref, atol=1e-4, rtol=1e-1)


def test_stack_auto_adjoint(operator, stacked_op, kspace_data, image_data):
    """Test the adjoint property of the stacked NUFFT operator."""
    kspace = stacked_op.op(image_data)
    image = stacked_op.adj_op(kspace_data)
    leftadjoint = np.vdot(image, image_data)
    rightadjoint = np.vdot(kspace, kspace_data)

    npt.assert_allclose(leftadjoint.conj(), rightadjoint, atol=1e-4, rtol=1e-4)


def test_stacked2traj3d():
    """Test the conversion from stacked to 3d trajectory."""
    dimz = 64
    traj2d = np.random.randn(100, 2)
    z_index = np.random.choice(dimz, 20, replace=False)

    traj3d = stacked2traj3d(traj2d, z_index, dimz)

    traj2d, z_index2 = traj3d2stacked(traj3d, dimz)

    npt.assert_allclose(traj2d, traj2d)
    npt.assert_allclose(z_index, z_index2)
