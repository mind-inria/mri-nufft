"""Test the interfaces module."""
import numpy as np
from pytest_cases import parametrize_with_cases, parametrize, fixture

from mrinufft import get_operator
from case_trajectories import CasesTrajectories

# #########################
# # Main Operator Fixture #
# #########################
#
# This fixture generate an MRINUFFT operator for every backend and every trajectory.
# All the test below will be perform on those operators.


@fixture(scope="module")
@parametrize("backend", ["pynfft"])
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
def ndft_op(operator):
    """Generate a NDFT operator, matching the property of the first operator."""
    return get_operator("numpy")(
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


def test_interfaces_accuracy_forward(operator, image_data, ndft_op):
    """Compare the interface to the raw NUDFT implementation."""
    kspace_nufft = operator.op(image_data)
    kspace = ndft_op.op(image_data)
    # FIXME: check with complex values ail
    assert np.allclose(abs(kspace), abs(kspace_nufft))


def test_interfaces_accuracy_backward(operator, kspace_data, ndft_op):
    """Compare the interface to the raw NUDFT implementation."""
    kspace_data[0, :] = 0
    kspace_data[0, 10] = 1
    image_nufft = operator.adj_op(kspace_data.copy())
    image = ndft_op.adj_op(kspace_data.copy())
    assert np.allclose(image, image_nufft)


def test_interfaces_autoadjoint(operator, kspace_data, image_data):
    """Test the adjoint property of the operator."""
    kspace = operator.op(image_data)
    image = operator.adj_op(kspace_data)

    assert np.allclose(np.vdot(image, image_data), np.vdot(kspace, kspace_data))
