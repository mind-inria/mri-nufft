"""Test the interfaces module."""

import numpy as np
from pytest_cases import parametrize_with_cases, parametrize, fixture
import pytest
from mrinufft import get_operator
from case_trajectories import CasesTrajectories

from helpers import kspace_from_op, image_from_op


CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False

TORCH_AVAILABLE = True
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = torch.cuda.is_available()
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
        "bart",
        "pynfft",
        "finufft",
        "cufinufft",
        "gpunufft",
        "sigpy",
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


@fixture(scope="session", autouse=True)
def ref_backend(request):
    """Get the reference backend from the CLI."""
    return request.config.getoption("ref")


@fixture(scope="module")
def ref_operator(request, operator, ref_backend):
    """Generate a NFFT operator, matching the property of the first operator."""
    return get_operator(ref_backend)(
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


@parametrize(
    "array_interface",
    [
        "numpy",
        pytest.param(
            "cupy",
            marks=[
                pytest.mark.skipif(
                    not CUPY_AVAILABLE,
                    reason="cupy not available",
                )
            ],
        ),
        pytest.param(
            "torch",
            marks=[
                pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
            ],
        ),
    ],
)
def test_interfaces_accuracy_forward(
    operator, image_data, ref_operator, array_interface
):
    """Compare the interface to the raw NUDFT implementation."""
    image_data_ = image_data
    if array_interface == "cupy":
        image_data_ = cp.array(image_data)
    elif array_interface == "torch":
        image_data_ = torch.from_numpy(image_data).to("cuda")
    kspace_nufft = operator.op(image_data_).squeeze()
    # FIXME: check with complex values ail
    kspace_ref = ref_operator.op(image_data).squeeze()

    if array_interface == "cupy":
        kspace_nufft = kspace_nufft.get()

    if array_interface == "torch":
        kspace_nufft = kspace_nufft.to("cpu").numpy()
    assert np.percentile(abs(kspace_nufft - kspace_ref) / abs(kspace_ref), 95) < 1e-1


@parametrize(
    "array_interface",
    [
        "numpy",
        pytest.param(
            "cupy",
            marks=[
                pytest.mark.skipif(
                    not CUPY_AVAILABLE,
                    reason="cupy not available",
                )
            ],
        ),
        pytest.param(
            "torch",
            marks=[
                pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
            ],
        ),
    ],
)
def test_interfaces_accuracy_backward(
    operator, kspace_data, ref_operator, array_interface
):
    """Compare the interface to the raw NUDFT implementation."""
    image_nufft = operator.adj_op(kspace_data.copy()).squeeze()
    image_ref = ref_operator.adj_op(kspace_data.copy()).squeeze()

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
    assert np.mean(reldiff) < 5e-5


def test_interface_lipschitz(operator):
    """Test the Lipschitz constant of the operator."""
    spec_rad = operator.get_lipschitz_cst(20)

    def AHA(x):
        return operator.adj_op(operator.op(x))

    L = np.zeros(10)
    for i in range(10):
        img_data = image_from_op(operator)
        img2_data = image_from_op(operator)

        L[i] = np.linalg.norm(AHA(img2_data) - AHA(img_data)) / np.linalg.norm(
            img2_data - img_data
        )

    assert np.mean(L) < 1.1 * spec_rad
