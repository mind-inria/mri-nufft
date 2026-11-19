"""Test the interfaces module."""

import numpy as np
import numpy.testing as npt
import pytest
from pytest import fixture
from mrinufft import get_operator

from helpers import (
    CasesTrajectories,
    assert_adjoint,
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


@fixture(scope="module")
def native_interface(backend):
    """First array interface accepted by the backend."""
    return backend_interfaces(backend)[0]


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


@pytest.mark.parametrize(
    "backend,name",
    [
        pytest.param(backend, name, marks=pytest.mark.backend(backend), id=backend)
        for backend, name in [
            ("finufft", "MRIfinufft"),
            ("mlx-nufft", "MRIMlxNUFFT"),
            ("cufinufft", "MRICufiNUFFT"),
            ("ducc0", "MRIDUCC0"),
            ("tensorflow", "MRITensorflowNUFFT"),
            ("pynufft-cpu", "MRIPynufft"),
            ("pynfft", "MRInfft"),
            ("gpunufft", "MRIGpuNUFFT"),
            ("torchkbnufft-cpu", "TorchKbNUFFTcpu"),
            ("torchkbnufft-gpu", "TorchKbNUFFTgpu"),
        ]
    ]
    + [("numpy", "MRInumpy")],
)
def test_get_operator(backend, name):
    """Test the get_operator function."""
    assert get_operator(backend).__name__ == name


def test_get_operator_fail():
    """Test the get_operator function."""
    with pytest.raises(ValueError):
        get_operator("unknown")


@parametrize_array_interface()
def test_interfaces_array_types(
    operator, native_interface, array_interface, image_data, kspace_data
):
    """Outputs follow the input array type and device, and match across types.

    The other tests only use the native interface of the backend.
    """
    image = to_interface(image_data, array_interface)
    kspace = to_interface(kspace_data, array_interface)
    image_ref = to_interface(image_data, native_interface)
    kspace_ref = to_interface(kspace_data, native_interface)
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
        ref = from_interface(ref, native_interface)
        npt.assert_allclose(
            from_interface(out, array_interface),
            ref,
            rtol=1e-3,
            atol=1e-3 * np.abs(ref).max(),
        )


def test_interfaces_accuracy_forward(
    operator, native_interface, image_data, ref_operator
):
    """Compare the forward operator to the reference backend."""
    kspace_nufft = operator.op(to_interface(image_data, native_interface))
    kspace_nufft = from_interface(kspace_nufft, native_interface).squeeze()
    kspace_ref = ref_operator.op(image_data).squeeze()
    assert np.percentile(abs(kspace_nufft - kspace_ref) / abs(kspace_ref), 95) < 1e-1


def test_interfaces_accuracy_backward(
    operator, native_interface, kspace_data, ref_operator
):
    """Compare the adjoint operator to the reference backend."""
    image_nufft = operator.adj_op(to_interface(kspace_data, native_interface))
    image_nufft = from_interface(image_nufft, native_interface).squeeze()
    image_ref = ref_operator.adj_op(kspace_data).squeeze()
    assert np.percentile(abs(image_nufft - image_ref) / abs(image_ref), 95) < 1e-1


def test_interfaces_autoadjoint(operator, native_interface):
    """Test the adjoint property of the operator."""
    assert_adjoint(operator, native_interface)


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


def test_check_shape():
    """Image and k-space are checked on their trailing dimensions.

    ``check_shape`` is only implemented in the base class.
    """
    samples, shape = CasesTrajectories().case_grid2D()
    operator = get_operator("numpy")(samples, shape)
    operator.check_shape(
        image=np.zeros((2, 3, *shape)), ksp=np.zeros((2, 3, operator.n_samples))
    )
    with pytest.raises(ValueError):
        operator.check_shape(image=np.zeros((shape[0] + 10, shape[1] + 1)))
    with pytest.raises(ValueError):
        operator.check_shape(ksp=np.zeros((1, operator.n_samples + 10)))
    with pytest.raises(ValueError):
        operator.check_shape()
