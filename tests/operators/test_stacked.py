"""Test for the stacked NUFFT operators.

The tests compares the stacked NUFFT (which uses FFT in the z-direction)
and the fully 3D ones.
"""

import numpy as np
import numpy.testing as npt
import pytest
from pytest import fixture, mark

from helpers import (
    CasesTrajectories,
    assert_adjoint,
    case_id,
    cases_of,
    from_interface,
    image_from_op,
    kspace_from_op,
    parametrize_array_interface,
    to_interface,
)
from mrinufft import get_operator
from mrinufft.operators.stacked import (
    MRIStackedNUFFT,
    MRIStackedNUFFTGPU,
    stacked2traj3d,
    traj3d2stacked,
)


@fixture(
    scope="module",
    params=[
        pytest.param(
            (MRIStackedNUFFT, "finufft"), marks=mark.backend("finufft"), id="finufft"
        ),
        pytest.param(
            (MRIStackedNUFFTGPU, "cufinufft"),
            marks=mark.backend("cufinufft"),
            id="cufinufft-gpu",
        ),
    ],
)
def stacked_class(request):
    """Stacked operator class and the backend of its 2D NUFFT."""
    return request.param


@fixture(
    scope="module",
    params=cases_of(CasesTrajectories, "random2D", "radial2D"),
    ids=case_id,
)
def trajectory(request):
    """Sampling pattern and image shape."""
    return request.param()


@fixture(
    scope="module",
    params=[(1, 1, False), (3, 4, False), (3, 4, True)],
    ids=["single", "batch_coils", "batch_sense"],
)
def config(request):
    """Number of batches, number of coils and SENSE."""
    return request.param


@fixture(scope="module", params=["full", "random_mask"])
def z_index(request):
    """Kind of z_index: all partitions or a random subset."""
    return request.param


@fixture(scope="module")
def operator(stacked_class, trajectory, config, z_index):
    """Initialize the stacked and non-stacked operators."""
    cls, backend = stacked_class
    kspace_locs, shape = trajectory
    n_batchs, n_coils, sense = config
    shape3d = (*shape, 14)  # a few partitions keep the 3D reference cheap

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

    extra = {}
    if cls is MRIStackedNUFFTGPU:
        extra["n_trans"] = 2 if n_coils > 1 else 1
    stacked = cls(
        samples=kspace_locs,
        shape=shape3d,
        z_index=z_index,
        backend=backend,
        n_coils=n_coils,
        n_batchs=n_batchs,
        smaps=smaps,
        **extra,
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
    return image_from_op(stacked_op)


@fixture(scope="module")
def kspace_data(stacked_op):
    """Generate a random kspace data."""
    return kspace_from_op(stacked_op)


@parametrize_array_interface("numpy", "cupy")
def test_stack_forward(operator, array_interface, ref_op, stacked_op, image_data):
    """Compare the stack interface to the 3D NUFFT implementation."""
    image_data_ = to_interface(image_data, array_interface)
    kspace_nufft_ = stacked_op.op(image_data_).squeeze()
    kspace_ref_ = ref_op.op(image_data_).squeeze()
    kspace_nufft = from_interface(kspace_nufft_, array_interface)
    kspace_ref = from_interface(kspace_ref_, array_interface)
    npt.assert_allclose(kspace_nufft, kspace_ref, atol=1e-4, rtol=1e-1)


@parametrize_array_interface("numpy", "cupy")
def test_stack_backward(operator, array_interface, stacked_op, ref_op, kspace_data):
    """Compare the stack interface to the 3D NUFFT implementation."""
    kspace_data_ = to_interface(kspace_data, array_interface)
    image_nufft_ = stacked_op.adj_op(kspace_data_).squeeze()
    image_ref_ = ref_op.adj_op(kspace_data_).squeeze()
    image_nufft = from_interface(image_nufft_, array_interface)
    image_ref = from_interface(image_ref_, array_interface)
    npt.assert_allclose(image_nufft, image_ref, atol=1e-4, rtol=1e-1)


def test_stack_auto_adjoint(stacked_op):
    """Test the adjoint property of the stacked NUFFT operator."""
    assert_adjoint(stacked_op, n_draws=1, rtol=1e-4)


def test_stack_reuse(stacked_op):
    """The stacked operator can be built around an existing 2D operator."""
    nufft_2d = stacked_op.operator
    extra = {}
    if isinstance(stacked_op, MRIStackedNUFFTGPU):
        extra["n_trans"] = stacked_op.n_trans
    reuse_op = type(stacked_op)(
        backend=nufft_2d,
        shape=stacked_op.shape,
        samples=stacked_op.samples,
        z_index=stacked_op.z_index,
        n_coils=stacked_op.n_coils,
        n_batchs=stacked_op.n_batchs,
        smaps=stacked_op.smaps,
        **extra,
    )
    assert reuse_op.operator is nufft_2d


def test_stacked2traj3d():
    """Test the conversion from stacked to 3d trajectory."""
    dimz = 64
    traj2d = np.random.rand(100, 2) - 0.5
    z_index = np.random.choice(dimz, 20, replace=False)

    traj3d = stacked2traj3d(traj2d, z_index, dimz)

    traj2d_back, z_index_back = traj3d2stacked(traj3d, dimz)

    npt.assert_allclose(traj2d_back, traj2d)
    npt.assert_allclose(z_index_back, z_index)
