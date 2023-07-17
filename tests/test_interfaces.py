"""Test the interfaces module."""
import numpy as np
import numpy.testing as npt
from pytest_cases import parametrize_with_cases, parametrize, fixture
import pytest

from mrinufft import get_operator
from case_trajectories import CasesTrajectories

# #########################
# # Main Operator Fixture #
# #########################
#
# This fixture generate an MRINUFFT operator for every backend and every trajectory.
# All the test below will be perform on those operators.


@fixture(scope="module")
@parametrize(
    "backend",
    [
        "pynfft",
        "finufft",
        "cufinufft",
    ],
)
@parametrize_with_cases("kspace_locs, shape", cases=CasesTrajectories)
def operator(
    request,
    backend="pynfft",
    kspace_locs=None,
    shape=None,
    n_coils=1,
    smaps=None,
):
    """Generate a random image and smaps."""
    return get_operator(backend)(kspace_locs, shape, n_coils=n_coils, smaps=smaps)


@fixture(scope="module")
@parametrize("backend", ["pynfft"])
def nfft_ref_op(request, operator, backend="pynfft"):
    """Generate a NFFT operator, matching the property of the first operator."""
    return get_operator(backend)(
        operator.samples, operator.shape, n_coils=operator.n_coils, smaps=operator.smaps
    )


@fixture(scope="module")
def image_data(operator):
    """Generate a random image."""
    if operator.smaps is None:
        img = np.random.rand(operator.n_coils, *operator.shape).astype(
            operator.cpx_dtype
        )
    elif operator.smaps is not None and operator.n_coils > 1:
        img = np.random.rand(*operator.shape).astype(operator.cpx_dtype)

    img += 1j * np.random.randn(*img.shape).astype(operator.cpx_dtype)
    return img


@fixture(scope="module")
def kspace_data(operator):
    """Generate a random kspace data."""
    kspace = np.random.randn(operator.n_coils, operator.n_samples).astype(
        operator.cpx_dtype
    )
    return kspace


def test_interfaces_accuracy_forward(operator, image_data, nfft_ref_op):
    """Compare the interface to the raw NUDFT implementation."""
    kspace_nufft = operator.op(image_data).squeeze()
    kspace_ref = nfft_ref_op.op(image_data).squeeze()
    # FIXME: check with complex values ail
    assert np.percentile(abs(kspace_nufft - kspace_ref), 95) < 1e-4
    assert np.max(abs(kspace_nufft - kspace_ref)) < 1e-3


def test_interfaces_accuracy_backward(operator, kspace_data, nfft_ref_op):
    """Compare the interface to the raw NUDFT implementation."""
    image_nufft = operator.adj_op(kspace_data.copy()).squeeze()
    image_ref = nfft_ref_op.adj_op(kspace_data.copy()).squeeze()

    npt.assert_allclose(image_nufft, image_ref, atol=1e-5, rtol=5e-4)


def test_interfaces_autoadjoint(operator, kspace_data, image_data):
    """Test the adjoint property of the operator."""
    kspace = operator.op(image_data)
    image = operator.adj_op(kspace_data)
    leftadjoint = np.vdot(image, image_data)
    rightadjoint = np.vdot(kspace, kspace_data)

    npt.assert_array_almost_equal(leftadjoint.conj(), rightadjoint)
