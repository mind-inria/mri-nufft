"""Test the interfaces module."""

import numpy as np
import numpy.testing as npt
import pytest
from pytest import fixture
from mrinufft import get_operator
from case_trajectories import CasesTrajectories

from helpers import (
    backend_interfaces,
    case_id,
    cases_of,
    from_interface,
    image_from_op,
    kspace_from_op,
    param_backends,
    parametrize_array_interface,
    to_interface,
)

# #########################
# # Main Operator Fixture #
# #########################
#
# This fixture generate an MRINUFFT operator for every backend and every trajectory.
# All the test below will be perform on those operators.


@fixture(
    scope="module",
    params=param_backends(
        "bart",
        "pynfft",
        "pynufft-cpu",
        "finufft",
        "mlx-nufft",
        "cufinufft",
        "ducc0",
        "gpunufft",
        "sigpy",
        "torchkbnufft-cpu",
        "torchkbnufft-gpu",
        "tensorflow",
    ),
)
def backend(request):
    """Backend under test."""
    return request.param


@fixture(
    scope="module",
    params=cases_of(
        CasesTrajectories, "random2D", "random3D", "radial2D", "radial3D", "grid2D"
    ),
    ids=case_id,
)
def trajectory(request):
    """Sampling pattern and image shape."""
    return request.param()


@fixture(scope="module")
def operator(backend, trajectory):
    """Generate an operator."""
    kspace_locs, shape = trajectory
    if backend in ["pynfft", "sigpy"] and kspace_locs.shape[-1] == 3:
        pytest.skip("3D for slow cpu is not tested")
    return get_operator(backend)(kspace_locs, shape, n_coils=1, smaps=None)


@fixture(scope="module")
def ref_operator(operator, ref_backend):
    """Generate a NFFT operator, matching the property of the first operator."""
    samples = operator.samples
    if operator.backend == "torchkbnufft-cpu" or operator.backend == "torchkbnufft-gpu":
        # torvchkbnufft has samples as torch tensor.
        samples = samples.cpu().numpy().astype(np.float32)
    return get_operator(ref_backend)(
        samples, operator.shape, n_coils=operator.n_coils, smaps=operator.smaps
    )


@fixture(scope="module")
def image_data(operator):
    """Generate a random image. Remains constant for the module."""
    return image_from_op(operator)


@fixture(scope="module")
def kspace_data(operator):
    """Generate a random kspace. Remains constant for the module."""
    return kspace_from_op(operator)


@fixture(scope="module")
def wrong_kspace_data(operator):
    """Generate a random image. Remains constant for the module."""
    wrong_n_samples = operator.n_samples + 10
    kspace = (1j * np.random.randn(operator.n_coils, wrong_n_samples)).astype(
        operator.cpx_dtype
    )
    kspace += np.random.randn(operator.n_coils, wrong_n_samples).astype(
        operator.cpx_dtype
    )
    return kspace


@fixture(scope="module")
def wrong_image_data(operator):
    """Generate a random image. Remains constant for the module."""
    wrong_shape = (operator.shape[0] + 10, operator.shape[1] + 1)
    if operator.smaps is None:
        img = np.random.randn(operator.n_coils, *wrong_shape).astype(operator.cpx_dtype)
    elif operator.smaps is not None and operator.n_coils > 1:
        img = np.random.randn(*wrong_shape).astype(operator.cpx_dtype)
    img += 1j * np.random.randn(*img.shape).astype(operator.cpx_dtype)
    return img


@parametrize_array_interface()
def test_interfaces_array_types(
    operator, backend, array_interface, image_data, kspace_data
):
    """Outputs follow the input array type and device, and match across types."""
    ref_interface = backend_interfaces(backend)[0]
    image = to_interface(image_data, array_interface)
    kspace = to_interface(kspace_data, array_interface)
    image_ref = to_interface(image_data, ref_interface)
    kspace_ref = to_interface(kspace_data, ref_interface)
    for out, ref in [
        (operator.op(image), operator.op(image_ref)),
        (operator.adj_op(kspace), operator.adj_op(kspace_ref)),
        (
            operator.data_consistency(image, kspace),
            operator.data_consistency(image_ref, kspace_ref),
        ),
    ]:
        assert type(out) is type(image)
        assert getattr(out, "device", None) == getattr(image, "device", None)
        ref = from_interface(ref, ref_interface)
        npt.assert_allclose(
            from_interface(out, array_interface),
            ref,
            rtol=1e-3,
            atol=1e-3 * np.abs(ref).max(),
        )


@parametrize_array_interface("numpy", "cupy")
def test_interfaces_accuracy_forward(
    operator, array_interface, image_data, ref_operator
):
    """Compare the interface to the raw NUDFT implementation."""
    image_data_ = to_interface(image_data, array_interface)
    kspace_nufft = operator.op(image_data_).squeeze()
    kspace_ref = ref_operator.op(image_data).squeeze()

    kspace_nufft = from_interface(kspace_nufft, array_interface)
    assert np.percentile(abs(kspace_nufft - kspace_ref) / abs(kspace_ref), 95) < 1e-1


@parametrize_array_interface("numpy", "cupy")
def test_interfaces_accuracy_backward(
    operator, array_interface, kspace_data, ref_operator
):
    """Compare the interface to the raw NUDFT implementation."""
    kspace_data_ = to_interface(kspace_data, array_interface)

    image_nufft = operator.adj_op(kspace_data_).squeeze()
    image_ref = ref_operator.adj_op(kspace_data).squeeze()

    image_nufft = from_interface(image_nufft, array_interface)
    assert np.percentile(abs(image_nufft - image_ref) / abs(image_ref), 95) < 1e-1


@parametrize_array_interface("numpy", "cupy")
def test_interfaces_autoadjoint(operator, array_interface):
    """Test the adjoint property of the operator."""
    reldiff = np.zeros(10)
    for i in range(10):
        img_data = to_interface(image_from_op(operator), array_interface)
        ksp_data = to_interface(kspace_from_op(operator), array_interface)
        kspace = operator.op(img_data)

        rightadjoint = np.vdot(
            from_interface(kspace, array_interface),
            from_interface(ksp_data, array_interface),
        )

        image = operator.adj_op(ksp_data)
        leftadjoint = np.vdot(
            from_interface(img_data, array_interface),
            from_interface(image, array_interface),
        )
        reldiff[i] = abs(rightadjoint - leftadjoint) / abs(leftadjoint)
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


######################
# Check shape tests  #
######################


def test_check_shape_pass(operator, image_data, kspace_data):
    """Correct shapes are accepted."""
    try:
        operator.check_shape(image=image_data)
        operator.check_shape(ksp=kspace_data)
    except ValueError:
        pytest.fail("Unexpected ValueError raised")


def test_check_shape_fail_image(operator, wrong_image_data):
    """A wrong image shape is rejected."""
    with pytest.raises(ValueError):
        operator.check_shape(image=wrong_image_data)


def test_check_shape_fail_kspace(operator, wrong_kspace_data):
    """A wrong k-space shape is rejected."""
    with pytest.raises(ValueError):
        operator.check_shape(ksp=wrong_kspace_data)
