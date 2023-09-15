"""Test the interfaces module."""
import numpy as np
import numpy.testing as npt
from pytest_cases import parametrize_with_cases, parametrize, fixture

from mrinufft import get_operator
from case_trajectories import CasesTrajectories

from helpers import kspace_from_op, image_from_op

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
        "gpunufft",
    ],
)
@parametrize_with_cases("kspace_locs, shape", cases=CasesTrajectories)
def operator(
    request,
    backend="pynfft",
    kspace_locs=None,
    shape=None,
    n_coils=1,
):
    """Generate an operator."""
    return get_operator(backend)(kspace_locs, shape, n_coils=n_coils, smaps=None)


@fixture(scope="module")
@parametrize("backend", ["pynfft"])
def nfft_ref_op(request, operator, backend="pynfft"):
    """Generate a NFFT operator, matching the property of the first operator."""
    return get_operator(backend)(
        operator.samples, operator.shape, n_coils=operator.n_coils, smaps=operator.smaps
    )


@fixture(scope="module")
def image_data(operator):
    """Generate a random image. Remains constant for the module."""
    return image_from_op(operator)


@fixture(scope="module")
def kspace_data(operator):
    """Generate a random kspace. Remains constant for the module."""
    return kspace_from_op(operator)


def test_interfaces_accuracy_forward(operator, image_data, nfft_ref_op):
    """Compare the interface to the raw NUDFT implementation."""
    kspace_nufft = operator.op(image_data).squeeze()
    kspace_ref = nfft_ref_op.op(image_data).squeeze()
    # FIXME: check with complex values ail
    assert np.percentile(abs(kspace_nufft - kspace_ref) / abs(kspace_ref), 95) < 1e-1


def test_interfaces_accuracy_backward(operator, kspace_data, nfft_ref_op):
    """Compare the interface to the raw NUDFT implementation."""
    image_nufft = operator.adj_op(kspace_data.copy()).squeeze()
    image_ref = nfft_ref_op.adj_op(kspace_data.copy()).squeeze()

    assert np.percentile(abs(image_nufft - image_ref) / abs(image_ref), 95) < 1e-1


def test_interfaces_autoadjoint(operator):
    """Test the adjoint property of the operator."""
    reldiff = np.zeros(10)
    for i in range(10):
        img_data = image_from_op(operator)
        ksp_data = kspace_from_op(operator)
        kspace = operator.op(img_data)

        rightadjoint = np.vdot(kspace, ksp_data)

        image = operator.adj_op(ksp_data)
        leftadjoint = np.vdot(img_data, image)
        reldiff[i] = abs(rightadjoint - leftadjoint) / abs(leftadjoint)
    print(reldiff)
    assert np.mean(reldiff) < 1e-5


def test_data_consistency_readonly(operator, image_data, kspace_data):
    """Test that the data consistency does not modify the input parameters data."""
    kspace_tmp = kspace_data.copy()
    image_tmp = image_data.copy()
    kspace_tmp.setflags(write=False)
    image_tmp.setflags(write=False)
    operator.data_consistency(image_data, kspace_tmp)
    npt.assert_equal(kspace_tmp, kspace_data)
    npt.assert_equal(image_tmp, image_data)


def test_data_consistency(operator, image_data, kspace_data):
    """Test the data consistency operation."""
    res = operator.data_consistency(image_data, kspace_data)

    res2 = operator.adj_op(operator.op(image_data) - kspace_data)

    npt.assert_allclose(res.squeeze(), res2.squeeze(), atol=1e-4, rtol=1e-1)


def test_gradient_lipschitz(operator, image_data, kspace_data):
    """Test the gradient lipschitz constant."""
    img = image_data.copy()
    for _ in range(10):
        grad = operator.data_consistency(img, kspace_data)
        norm = np.linalg.norm(grad)
        grad /= norm
        np.copyto(img, grad)
        norm_prev = norm

    # TODO: check that the value is "not too far" from 1
    # TODO: to do the same with density compensation
    assert (norm - norm_prev) / norm_prev < 1e-3
