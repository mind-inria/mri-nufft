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
except ImportError:
    TORCH_AVAILABLE = False


@fixture(scope="module")
@parametrize(backend=["cufinufft", "finufft", "gpunufft", "torchkbnufft"])
@parametrize_with_cases(
    "kspace_loc, shape",
    cases=[
        CasesTrajectories.case_grid2D,
        CasesTrajectories.case_nyquist_radial2D,
        CasesTrajectories.case_nyquist_radial3D_lowmem,
    ],
)
def operator(kspace_loc, shape, backend):
    """Create NUFFT operator with autodiff capabilities."""
    kspace_loc = kspace_loc.astype(np.float32)
    nufft = get_operator(backend_name=backend, wrt_data=True, wrt_traj=True)(
        samples=kspace_loc,
        shape=shape,
        smaps=None,
        squeeze_dims=False,  # Squeezing breaks dimensions !
    )
    return nufft


def ndft_matrix(operator):
    """Get the NDFT matrix from the operator."""
    return get_fourier_matrix(operator.samples, operator.shape, normalize=True)


@pytest.mark.parametrize("interface", ["torch-gpu", "torch-cpu"])
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Pytorch is not installed")
def test_adjoint_and_grad(operator, interface):
    """Test the adjoint and gradient of the operator."""
    if operator.backend == "finufft" and "gpu" in interface:
        pytest.skip("GPU not supported for finufft backend")

    if torch.is_tensor(operator.samples):
        operator.samples = operator.samples.cpu().detach().numpy()

    operator.samples = to_interface(operator.samples, interface=interface)
    ksp_data = to_interface(kspace_from_op(operator), interface=interface)
    img_data = to_interface(image_from_op(operator), interface=interface)
    ksp_data.requires_grad = True
    operator.samples.requires_grad = True

    with torch.autograd.set_detect_anomaly(True):
        adj_data = operator.adj_op(ksp_data).reshape(img_data.shape)
        adj_data_ndft = (ndft_matrix(operator).conj().T @ ksp_data.flatten()).reshape(
            adj_data.shape
        )
        loss_nufft = torch.mean(torch.abs(adj_data - img_data) ** 2)
        loss_ndft = torch.mean(torch.abs(adj_data_ndft - img_data) ** 2)

    # Check if nufft and ndft w.r.t trajectory are close in the backprop
    gradient_ndft_ktraj = torch.autograd.grad(
        loss_ndft, operator.samples, retain_graph=True
    )[0]
    gradient_nufft_ktraj = torch.autograd.grad(
        loss_nufft, operator.samples, retain_graph=True
    )[0]
    assert torch.allclose(gradient_ndft_ktraj, gradient_nufft_ktraj, atol=5e-2)

    # Check if nufft and ndft are close in the backprop
    gradient_ndft_kdata = torch.autograd.grad(loss_ndft, ksp_data, retain_graph=True)[0]
    gradient_nufft_kdata = torch.autograd.grad(loss_nufft, ksp_data, retain_graph=True)[
        0
    ]
    assert torch.allclose(gradient_ndft_kdata, gradient_nufft_kdata, atol=6e-3)


@pytest.mark.parametrize("interface", ["torch-gpu", "torch-cpu"])
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Pytorch is not installed")
def test_forward_and_grad(operator, interface):
    """Test the adjoint and gradient of the operator."""
    if operator.backend == "finufft" and "gpu" in interface:
        pytest.skip("GPU not supported for finufft backend")

    if torch.is_tensor(operator.samples):
        operator.samples = operator.samples.cpu().detach().numpy()

    operator.samples = to_interface(operator.samples, interface=interface)
    ksp_data_ref = to_interface(kspace_from_op(operator), interface=interface)
    img_data = to_interface(image_from_op(operator), interface=interface)

    img_data.requires_grad = True
    operator.samples.requires_grad = True

    with torch.autograd.set_detect_anomaly(True):
        ksp_data = operator.op(img_data).reshape(ksp_data_ref.shape)
        ksp_data_ndft = (ndft_matrix(operator) @ img_data.flatten()).reshape(
            ksp_data.shape
        )
        loss_nufft = torch.mean(torch.abs(ksp_data - ksp_data_ref) ** 2)
        loss_ndft = torch.mean(torch.abs(ksp_data_ndft - ksp_data_ref) ** 2)

    # Check if nufft and ndft w.r.t trajectory are close in the backprop
    gradient_ndft_ktraj = torch.autograd.grad(
        loss_ndft, operator.samples, retain_graph=True
    )[0]
    gradient_nufft_ktraj = torch.autograd.grad(
        loss_nufft, operator.samples, retain_graph=True
    )[0]
    assert torch.allclose(gradient_ndft_ktraj, gradient_nufft_ktraj, atol=5e-2)

    # Check if nufft and ndft are close in the backprop
    gradient_ndft_kdata = torch.autograd.grad(loss_ndft, img_data, retain_graph=True)[0]
    gradient_nufft_kdata = torch.autograd.grad(loss_nufft, img_data, retain_graph=True)[
        0
    ]
    assert torch.allclose(gradient_ndft_kdata, gradient_nufft_kdata, atol=6e-3)
