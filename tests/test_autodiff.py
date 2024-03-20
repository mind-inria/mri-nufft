"""Test the autodiff functionnality """
import numpy as np
try:
    import torch
except ImportError:
    AUTOGRAD_AVAILABLE = False
from mrinufft.operators.interfaces.nudft_numpy import get_fourier_matrix
import pytest
from pytest_cases import parametrize_with_cases
from case_trajectories import CasesTrajectories
from mrinufft.operators import get_operator

from helpers import (
    kspace_from_op,
    image_from_op,
    to_interface,
)

if not AUTOGRAD_AVAILABLE:
    raise ValueError("Autograd not available, ensure torch is installed.")

@parametrize_with_cases("kspace_locs, shape", cases=CasesTrajectories.case_grid2D)
@pytest.mark.parametrize("interface", ["torch-gpu", "torch-cpu"])
def test_adjoint_and_gradients(kspace_locs,shape,interface):
    """. """ 
    operator = get_operator("cufinufft",kspace_locs, shape, n_coils=1, smaps=None,autograd="data")
    kdata = kspace_from_op(operator)
    kdata_ = to_interface(kdata,interface=interface)
    Idata = operator.adj_op(kdata_)
    kdata_.requires_grad = True
    breakpoint()
    ktraj = kspace_locs + 0.01 * np.random.uniform(shape) * 2 * np.pi 
    with torch.autograd.set_detect_anomaly(True):
        operator_n = get_operator("cufinufft",ktraj, shape, n_coils=1, smaps=None,autograd="data")
        I_nufft = operator_n.adj_op(kdata_)
        A = get_fourier_matrix(ktraj,shape)
        A = to_interface(A,interface=interface).type(torch.complex64)
        I_ndft = (((A.conj()).T )@ kdata_.flatten()).unsqueeze(0).view(I_nufft.shape)
        loss_nufft = torch.mean(torch.abs(Idata - I_nufft)**2)
        loss_nudft = torch.mean(torch.abs(Idata - I_ndft)**2)

    # Test if the NUFFT and NDFT operations are close
    assert torch.quantile(abs(I_nufft - I_ndft) / abs(I_ndft), 0.95) < 1e-1

    # Test gradients with respect to kdata
    gradient_ndft_kdata = torch.autograd.grad(loss_nudft, kdata_,retain_graph=True)[0]
    gradient_nufft_kdata = torch.autograd.grad(loss_nufft, kdata_,retain_graph=True)[0]
    assert torch.allclose(gradient_ndft_kdata, gradient_nufft_kdata, atol=6e-3)

@parametrize_with_cases("kspace_locs, shape", cases=CasesTrajectories.case_grid2D)
@pytest.mark.parametrize("interface", ["torch-gpu", "torch-cpu"])
def test_forward_and_gradients(kspace_locs,shape,interface):
    """. """
    operator = get_operator("cufinufft",kspace_locs, shape, n_coils=1, smaps=None,autograd="data")
    image = image_from_op(operator)
    image = to_interface(image,interface=interface)
    kdata = operator.op(image)
    image.requires_grad=True
    
    ktraj = kspace_locs + 0.01 * np.random.uniform(shape) * 2 * np.pi 
    with torch.autograd.set_detect_anomaly(True):
        operator_n = get_operator("cufinufft",ktraj, shape, n_coils=1, smaps=None,autograd="data")
        kdata_nufft = operator_n.op(image)
        A = get_fourier_matrix(ktraj,shape)
        A = to_interface(A,interface=interface).type(torch.complex64)
        
        kdata_ndft = A @ image.flatten()
        loss_nufft = torch.mean(torch.abs(kdata - kdata_nufft)**2)
        loss_ndft = torch.mean(torch.abs(kdata - kdata_ndft)**2)

    assert torch.quantile(abs(kdata_ndft - kdata_ndft) / abs(kdata_ndft), 0.95) < 1e-1

    # Test gradients with respect to image
    gradient_ndft_kdata = torch.autograd.grad(loss_nufft, image,retain_graph=True)[0]
    gradient_nufft_kdata = torch.autograd.grad(loss_ndft, image,retain_graph=True)[0]
    assert torch.allclose(gradient_ndft_kdata, gradient_nufft_kdata, atol=6e-3)

