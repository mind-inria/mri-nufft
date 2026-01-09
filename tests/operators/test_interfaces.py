"""Test the interfaces module."""

import numpy as np
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture
from mrinufft import get_operator
from mrinufft.operators.base import get_toeplitz_kernel
from case_trajectories import CasesTrajectories
import torch
import torch.fft as fft

from helpers import (
    kspace_from_op,
    image_from_op,
    to_interface,
    from_interface,
    param_array_interface,
)

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
        "pynufft-cpu",
        "finufft",
        "cufinufft",
        "gpunufft",
        "sigpy",
        "torchkbnufft-cpu",
        "torchkbnufft-gpu",
        "tensorflow",
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
    if backend in ["pynfft", "sigpy"] and kspace_locs.shape[-1] == 3:
        pytest.skip("3D for slow cpu is not tested")
    return get_operator(backend)(kspace_locs, shape, n_coils=n_coils, smaps=None)


@fixture(scope="session", autouse=True)
def ref_backend(request):
    """Get the reference backend from the CLI."""
    return request.config.getoption("ref")


@fixture(scope="module")
def ref_operator(request, operator, ref_backend):
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


@param_array_interface
def test_interfaces_accuracy_forward(
    operator, array_interface, image_data, ref_operator
):
    """Compare the interface to the raw NUDFT implementation."""
    image_data_ = to_interface(image_data, array_interface)
    kspace_nufft = operator.op(image_data_).squeeze()
    kspace_ref = ref_operator.op(image_data).squeeze()

    kspace_nufft = from_interface(kspace_nufft, array_interface)
    assert np.percentile(abs(kspace_nufft - kspace_ref) / abs(kspace_ref), 95) < 1e-1


@param_array_interface
def test_interfaces_accuracy_backward(
    operator, array_interface, kspace_data, ref_operator
):
    """Compare the interface to the raw NUDFT implementation."""
    kspace_data_ = to_interface(kspace_data, array_interface)

    image_nufft = operator.adj_op(kspace_data_).squeeze()
    image_ref = ref_operator.adj_op(kspace_data).squeeze()

    image_nufft = from_interface(image_nufft, array_interface)
    assert np.percentile(abs(image_nufft - image_ref) / abs(image_ref), 95) < 1e-1


@param_array_interface
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


@pytest.mark.parametrize("backend", ["cufinufft", "finufft"])
@pytest.mark.parametrize("shape", [(64, 64), (32, 32, 32)]) # Test 2D et 3D
def test_toeplitz_consistency(backend, shape):
    """
    Apply A^H A x using the Toeplitz kernel
    Then check with a given backend of the NUFFT operators using its op and adj.op methods
    """
    device = "cuda" if torch.cuda.is_available() and backend == "cufinufft" else "cpu"

    # 1. Setup : random trajectories and samples
    n_samples = np.prod(shape) * 2 # oversampling to make sure the image is well reconstr (Nyquist)
    ndim = len(shape)
    ktraj = (torch.rand((n_samples, ndim), device=device) - 0.5) # [-0.5, 0.5]   

    # Initialize the NUFFT operator
    nufft_op = get_operator(backend)(
        samples=ktraj,
        shape=shape,
        backend=backend,
        device=device
    )

    # 2. Create a random testing image
    x = torch.randn(shape, dtype=torch.complex64, device=device)

    # 3. Compute the reference method (without the Toeplitz trick)
    #  A^H A x
    # We apply the standard NUFFT operator
    ref_out = nufft_op.adj(nufft_op.forward(x))

    # 4. Accelerated computation using the Toeplitz trick
    # recover the kernel from the NUFFT for the convolution in the spectral domain
    kernel = get_toeplitz_kernel(nufft_op)

    # 5. Zero-padding of image x onto the (2*N) grid to preserve linear convol (instead of circ)
    # image x is centered in the middle of the large zero-padded tensor
    pad_args = []
    for s in reversed(shape):
        pad_args += [s // 2, s // 2]
    x_padded = torch.nn.functional.pad(x, pad_args)
    
    # 6. Fourier transform of the image and multiplication in k-space (convolution)
    x_f = fft.fftn(x_padded)
    res_f = x_f * kernel
    res_padded = fft.ifftn(res_f)
    
    # 7. Crop to return to the original image size (shape)
    # we discard the padding to only keep the central area 
    slices = tuple(slice(s // 2, s // 2 + s) for s in shape)
    toeplitz_out = res_padded[slices]
    
    # 8. Check numerical consistency
    # A relative toleranve is used (float32 ~ 1e-5)
    diff = torch.norm(ref_out - toeplitz_out) / torch.norm(ref_out)
    
    print(f"Rel. Error ({backend} {ndim}D): {diff.item():.2e}")
    assert diff < 1e-4, f"Toeplitz output deviates too much from NUFFT reference: {diff.item()}"
    ## call this test manually to debug as follows:
    ## test_toeplitz_consistency("cufinufft", (128, 128))


@param_array_interface
def test_check_shape_pass(operator, array_interface, image_data, kspace_data):
    """Compare the interface to the raw NUDFT implementation."""
    image_data_ = to_interface(image_data, array_interface)
    try:
        operator.check_shape(image=image_data_)
        operator.check_shape(ksp=kspace_data)
    except ValueError:
        pytest.fail("Unexpected ValueError raised")


######################
# Check shape tests  #
######################


@param_array_interface
def test_check_shape_fail_image(operator, array_interface, wrong_image_data):
    """Compare the interface to the raw NUDFT implementation."""
    image_data_ = to_interface(wrong_image_data, array_interface)
    with pytest.raises(ValueError):
        operator.check_shape(image=image_data_)


@param_array_interface
def test_check_shape_fail_kspace(operator, array_interface, wrong_kspace_data):
    """Compare the interface to the raw NUDFT implementation."""
    kspace_data_ = to_interface(wrong_kspace_data, array_interface)
    with pytest.raises(ValueError):
        operator.check_shape(ksp=kspace_data_)
