"""Test the autodiff functionnality."""

import numpy as np
from mrinufft.operators.interfaces.nudft_numpy import get_fourier_matrix
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture
from case_trajectories import CasesTrajectories
from mrinufft.operators import get_operator
import warnings

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
@parametrize(backend=["cufinufft", "finufft", "gpunufft"])
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
        squeeze_dims=False,  # Squeezing breaks dimensions !
    )

    return nufft


def proper_trajectory_torch(trajectory, normalize="pi"):
    """Normalize the trajectory to be used by NUFFT operators on device."""
    if not torch.is_tensor(trajectory):
        raise ValueError("trajectory should be a torch.Tensor")

    new_traj = trajectory.clone()
    new_traj = new_traj.view(-1, trajectory.shape[-1])

    if normalize == "pi" and torch.max(torch.abs(new_traj)) - 1e-4 < 0.5:
        warnings.warn(
            "Samples will be rescaled to [-pi, pi), assuming they were in [-0.5, 0.5)"
        )
        new_traj *= 2 * torch.pi
    elif normalize == "unit" and torch.max(torch.abs(new_traj)) - 1e-4 > 0.5:
        warnings.warn(
            "Samples will be rescaled to [-0.5, 0.5), assuming they were in [-pi, pi)"
        )
        new_traj /= 2 * torch.pi

    if normalize == "unit" and torch.max(new_traj) >= 0.5:
        new_traj = (new_traj + 0.5) % 1 - 0.5

    return new_traj


def get_fourier_matrix_torch(ktraj, shape, dtype=torch.complex64, normalize=False):
    """Get the NDFT Fourier Matrix which is calculated on device."""
    device = ktraj.device
    ktraj = proper_trajectory_torch(ktraj, normalize="unit")
    n = np.prod(shape)
    ndim = len(shape)

    r = [torch.linspace(-s / 2, s / 2 - 1, s, device=device) for s in shape]

    grid_r = torch.meshgrid(r, indexing="ij")
    grid_r = torch.reshape(torch.stack(grid_r), (ndim, n)).to(device)

    traj_grid = torch.matmul(ktraj, grid_r)
    matrix = torch.exp(-2j * np.pi * traj_grid).to(dtype).to(device).clone()

    if normalize:
        matrix /= torch.sqrt(torch.tensor(np.prod(shape), device=device)) * torch.pow(
            torch.sqrt(torch.tensor(2, device=device)), ndim
        )

    return matrix


def ndft_matrix_ktraj(operator, k_traj):
    """Get the NDFT matrix from the operator."""
    return get_fourier_matrix_torch(k_traj, operator.shape, normalize=True)


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
        loss_nufft = torch.mean(torch.abs(adj_data - img_data) ** 2)
        loss_ndft = torch.mean(torch.abs(adj_data_ndft - img_data) ** 2)

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

    ktraj = to_interface(np.copy(operator.samples), interface=interface)

    ksp_data_ref = to_interface(kspace_from_op(operator), interface=interface)
    img_data = to_interface(image_from_op(operator), interface=interface)

    img_data.requires_grad = True
    ktraj.requires_grad = True

    with torch.autograd.set_detect_anomaly(True):
        ksp_data = operator.op(img_data, ktraj).reshape(ksp_data_ref.shape)
        ndft_matrix_torch = ndft_matrix_ktraj(operator, ktraj)
        ksp_data_ndft = (ndft_matrix_torch @ img_data.flatten()).reshape(ksp_data.shape)
        loss_nufft = torch.mean(torch.abs(ksp_data - ksp_data_ref) ** 2)
        loss_ndft = torch.mean(torch.abs(ksp_data_ndft - ksp_data_ref) ** 2)

    # Check if nufft and ndft w.r.t trajectory are close in the backprop
    gradient_ndft_ktraj = torch.autograd.grad(loss_ndft, ktraj, retain_graph=True)[0]
    gradient_nufft_ktraj = torch.autograd.grad(loss_nufft, ktraj, retain_graph=True)[  #
        0
    ]
    assert torch.allclose(gradient_ndft_ktraj, gradient_nufft_ktraj, atol=6e-3)

    # Check if nufft and ndft are close in the backprop
    gradient_ndft_kdata = torch.autograd.grad(loss_ndft, img_data, retain_graph=True)[0]
    gradient_nufft_kdata = torch.autograd.grad(loss_nufft, img_data, retain_graph=True)[
        0
    ]
    assert torch.allclose(gradient_ndft_kdata, gradient_nufft_kdata, atol=6e-3)
