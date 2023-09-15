"""Test for the stacked NUFFT operator.

The tests compares the stacked NUFFT (which uses FFT in the z-direction)
and the fully 3D ones.
"""
import numpy as np

import numpy.testing as npt
from pytest_cases import parametrize_with_cases, parametrize, fixture

from mrinufft.operators.stacked import MRIStackedNUFFT
from mrinufft import get_operator
from mrinufft.operators.interfaces import proper_trajectory
from case_trajectories import CasesTrajectories


@fixture(scope="module")
@parametrize(
    "n_batchs, n_coils, sense",
    [(1, 1, False), (1, 4, False), (1, 4, True), (2, 4, False), (2, 4, True)],
)
@parametrize("z_index", ["full", "random_mask"])
@parametrize("backend", ["finufft"])
@parametrize_with_cases("kspace_locs, shape", cases=CasesTrajectories, glob="*2D")
def operators(request, backend, kspace_locs, shape, z_index, n_batchs, n_coils, sense):
    """Initialize the stacked and non-stacked operators."""
    shape3d = (*shape, shape[-1])  # add a 3rd dimension

    if z_index == "full":
        z_index = np.arange(shape3d[-1])
        z_index_ = z_index
    elif z_index == "random_mask":
        z_index = np.random.rand(shape3d[-1]) > 0.5
        z_index_ = np.arange(shape3d[-1])[z_index]

    z_kspace = (z_index_ - shape3d[-1] // 2) / shape3d[-1]
    # create the equivalent 3d trajectory
    kspace_locs_proper = proper_trajectory(kspace_locs, normalize="unit")
    nsamples = len(kspace_locs_proper)
    nz = len(z_kspace)
    kspace_locs3d = np.zeros((nsamples * nz, 3))
    # TODO use numpy api for this ?
    for i in range(nsamples):
        kspace_locs3d[i * nz : (i + 1) * nz, :2] = kspace_locs_proper[i]
        kspace_locs3d[i * nz : (i + 1) * nz, 2] = z_kspace

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

    stacked = MRIStackedNUFFT(
        samples=kspace_locs,
        shape=shape3d,
        z_index=z_index,
        backend=backend,
        n_coils=n_coils,
        n_batchs=n_batchs,
        smaps=smaps,
    )
    return stacked, ref


@fixture(scope="module")
def operator(operators):
    """Return operator."""
    return operators[0]


@fixture(scope="module")
def ref_op(operators):
    """Return ref operator."""
    return operators[1]


@fixture(scope="module")
def image_data(operator):
    """Generate a random image."""
    B, C = operator.n_batchs, operator.n_coils
    if operator.smaps is None:
        img = np.random.randn(B, C, *operator.shape).astype(operator.cpx_dtype)
    elif operator.smaps is not None and operator.n_coils > 1:
        img = np.random.randn(B, *operator.shape).astype(operator.cpx_dtype)

    img += 1j * np.random.randn(*img.shape).astype(operator.cpx_dtype)
    return img


@fixture(scope="module")
def kspace_data(operator):
    """Generate a random kspace data."""
    B, C = operator.n_batchs, operator.n_coils
    kspace = (1j * np.random.randn(B, C, operator.n_samples)).astype(operator.cpx_dtype)
    kspace += np.random.randn(B, C, operator.n_samples).astype(operator.cpx_dtype)
    return kspace


def test_stack_forward(operator, ref_op, image_data):
    """Compare Stacked and 3D NUFFT ."""
    kspace_nufft = operator.op(image_data).squeeze()
    kspace_ref = ref_op.op(image_data).squeeze()
    assert np.percentile(abs(kspace_nufft - kspace_ref), 95) < 1e-4
    assert np.max(abs(kspace_nufft - kspace_ref)) < 1e-2


def test_stack_backward(operator, kspace_data, ref_op):
    """Compare the interface to the raw NUDFT implementation."""
    image_nufft = operator.adj_op(kspace_data.copy()).squeeze()
    image_ref = ref_op.adj_op(kspace_data.copy()).squeeze()

    npt.assert_allclose(image_nufft, image_ref, atol=1e-4, rtol=1e-1)


def test_stack_auto_adjoint(operator, kspace_data, image_data):
    kspace = operator.op(image_data)
    image = operator.adj_op(kspace_data)
    leftadjoint = np.vdot(image, image_data)
    rightadjoint = np.vdot(kspace, kspace_data)

    npt.assert_allclose(leftadjoint.conj(), rightadjoint, atol=1e-4, rtol=1e-4)
