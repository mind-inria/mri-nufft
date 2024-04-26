"""Test the autodiff functionnality."""

import numpy as np
from mrinufft.operators.interfaces.nudft_numpy import get_fourier_matrix
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture
from case_trajectories import CasesTrajectories
from mrinufft.operators import get_operator


from helpers import (
    kspace_from_op,
    image_from_op,
    to_interface,
)


TORCH_AVAILABLE = True
try:
    import torch
    import torch.testing as tt
except ImportError:
    TORCH_AVAILABLE = False


@fixture(scope="module")
@parametrize(backend=["cufinufft", "finufft"])
@parametrize(autograd=["data"])
@parametrize_with_cases(
    "kspace_loc, shape",
    cases=[
        CasesTrajectories.case_grid2D,
        CasesTrajectories.case_nyquist_radial2D,
    ],  # 2D cases only for reduced memory footprint.
)
def operator(kspace_loc, shape, backend, autograd):
    """Create NUFFT operator with autodiff capabilities."""
    kspace_loc = kspace_loc.astype(np.float32)

    nufft = get_operator(backend_name=backend, autograd=autograd)(
        samples=kspace_loc,
        shape=shape,
        smaps=None,
    )

    return nufft


@fixture(scope="module")
def ndft_matrix(operator):
    """Get the NDFT matrix from the operator."""
    return get_fourier_matrix(operator.samples, operator.shape, normalize=True)


@pytest.mark.parametrize("interface", ["torch-gpu", "torch-cpu"])
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Pytorch is not installed")
def test_adjoint_and_grad(operator, ndft_matrix, interface):
    """Test the adjoint and gradient of the operator."""
    if operator.backend == "finufft" and "gpu" in interface:
        pytest.skip("GPU not supported for finufft backend")
    ndft_matrix_torch = to_interface(ndft_matrix, interface=interface)
    ksp_data = to_interface(kspace_from_op(operator), interface=interface)
    img_data = to_interface(image_from_op(operator), interface=interface)
    ksp_data.requires_grad = True

    with torch.autograd.set_detect_anomaly(True):
        adj_data = operator.adj_op(ksp_data).reshape(img_data.shape)
        adj_data_ndft = (ndft_matrix_torch.conj().T @ ksp_data.flatten()).reshape(
            adj_data.shape
        )
        loss_nufft = torch.mean(torch.abs(adj_data) ** 2)
        loss_ndft = torch.mean(torch.abs(adj_data_ndft) ** 2)

    # Check if nufft and ndft are close in the backprop
    grad_ndft_kdata = torch.autograd.grad(loss_ndft, ksp_data, retain_graph=True)[0]
    grad_nufft_kdata = torch.autograd.grad(loss_nufft, ksp_data, retain_graph=True)[0]
    tt.assert_close(grad_ndft_kdata, grad_nufft_kdata, rtol=1, atol=1)


@pytest.mark.parametrize("interface", ["torch-gpu", "torch-cpu"])
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Pytorch is not installed")
def test_adjoint_and_gradauto(operator, ndft_matrix, interface):
    """Test the adjoint and gradient of the operator using autograd gradcheck."""
    if operator.backend == "finufft" and "gpu" in interface:
        pytest.skip("GPU not supported for finufft backend")

    ksp_data = to_interface(kspace_from_op(operator), interface=interface)
    ksp_data = torch.ones(ksp_data.shape, requires_grad=True, dtype=ksp_data.dtype)
    print(ksp_data.shape)
    # todo: tighten the tolerance
    assert torch.autograd.gradcheck(
        operator.adjoint,
        ksp_data,
        eps=1e-10,
        rtol=1,
        atol=1,
        nondet_tol=1,
        raise_exception=True,
    )


@pytest.mark.parametrize("interface", ["torch-gpu", "torch-cpu"])
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Pytorch is not installed")
def test_forward_and_grad(operator, ndft_matrix, interface):
    """Test the adjoint and gradient of the operator."""
    if operator.backend == "finufft" and "gpu" in interface:
        pytest.skip("GPU not supported for finufft backend")

    ndft_matrix_torch = to_interface(ndft_matrix, interface=interface)
    ksp_data_ref = to_interface(kspace_from_op(operator), interface=interface)
    img_data = to_interface(image_from_op(operator), interface=interface)
    img_data.requires_grad = True

    with torch.autograd.set_detect_anomaly(True):
        ksp_data = operator.op(img_data).reshape(ksp_data_ref.shape)
        ksp_data_ndft = (ndft_matrix_torch @ img_data.flatten()).reshape(ksp_data.shape)
        loss_nufft = torch.mean(torch.abs(ksp_data - ksp_data_ref) ** 2)
        loss_ndft = torch.mean(torch.abs(ksp_data_ndft - ksp_data_ref) ** 2)

    # Check if nufft and ndft are close in the backprop
    grad_ndft_kdata = torch.autograd.grad(loss_ndft, img_data, retain_graph=True)[0]
    grad_nufft_kdata = torch.autograd.grad(loss_nufft, img_data, retain_graph=True)[0]
    assert torch.allclose(grad_ndft_kdata, grad_nufft_kdata, atol=6e-3)


@pytest.mark.parametrize("interface", ["torch-gpu", "torch-cpu"])
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Pytorch is not installed")
def test_forward_and_gradauto(operator, ndft_matrix, interface):
    """Test the forward and gradient of the operator using autograd gradcheck."""
    if operator.backend == "finufft" and "gpu" in interface:
        pytest.skip("GPU not supported for finufft backend")

    img_data = to_interface(image_from_op(operator), interface=interface)
    img_data = torch.ones(img_data.shape, requires_grad=True, dtype=img_data.dtype)
    print(img_data.shape)
    # todo: tighten the tolerance
    assert torch.autograd.gradcheck(
        operator.adjoint,
        img_data,
        eps=1e-10,
        rtol=1,
        atol=1,
        nondet_tol=1,
        raise_exception=True,
    )
