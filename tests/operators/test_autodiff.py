"""Test the autodiff functionnality."""

import numpy as np
from numpy.testing import assert_allclose
from mrinufft.operators.interfaces.nudft_numpy import get_fourier_matrix
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture
from case_trajectories import CasesTrajectories
from mrinufft.operators import get_operator

from helpers import kspace_from_op, image_from_op, to_interface, assert_almost_allclose


TORCH_AVAILABLE = True
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False


@fixture(scope="module")
@parametrize(backend=["cufinufft", "finufft", "gpunufft"])
@parametrize(
    "n_coils, n_trans, sense",
    [
        (1, 1, False),
        (4, 1, True),
        (4, 1, False),
        (4, 2, True),
        (4, 2, False),
    ],
)
@parametrize_with_cases(
    "kspace_loc, shape",
    cases=[
        CasesTrajectories.case_grid2D,
        CasesTrajectories.case_nyquist_radial2D,
        CasesTrajectories.case_nyquist_lowmem_radial3D,
    ],
)
def operator(request, kspace_loc, shape, n_coils, sense, n_trans, backend):
    """Create NUFFT operator with autodiff capabilities."""
    if n_trans != 1 and backend == "gpunufft":
        pytest.skip("Duplicate case.")
    if sense:
        smaps = 1j * np.random.rand(n_coils, *shape)
        smaps += np.random.rand(n_coils, *shape)
        smaps = smaps.astype(np.complex64)
        smaps /= np.linalg.norm(smaps, axis=0)
    else:
        smaps = None
    kspace_loc = kspace_loc.astype(np.float32)
    nufft = get_operator(backend_name=backend, wrt_data=True, wrt_traj=True)(
        samples=kspace_loc,
        shape=shape,
        n_coils=n_coils,
        n_trans=1,
        smaps=smaps,
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

    if "gpu" in interface:
        operator.samples = operator.samples.to("cuda")
    else:
        operator.samples = operator.samples.cpu()
    ksp_data = to_interface(kspace_from_op(operator), interface=interface)
    img_data = to_interface(image_from_op(operator), interface=interface)

    ksp_data.requires_grad = True
    with torch.autograd.set_detect_anomaly(True):
        adj_data = operator.adj_op(ksp_data).reshape(img_data.shape)
        if operator.smaps is not None:
            smaps = torch.from_numpy(operator.smaps).to(img_data.device)
            adj_data_ndft_smaps = torch.cat(
                [
                    (ndft_matrix(operator).conj().T @ ksp_data[i].flatten()).reshape(
                        img_data.shape
                    )[None, ...]
                    for i in range(ksp_data.shape[0])
                ],
                dim=0,
            )
            adj_data_ndft = torch.sum(smaps.conj() * adj_data_ndft_smaps, dim=0)
        else:
            adj_data_ndft = torch.matmul(
                ndft_matrix(operator).conj().T, ksp_data.T
            ).T.reshape(img_data.shape)
        loss_nufft = torch.mean(torch.abs(adj_data - img_data) ** 2)
        loss_ndft = torch.mean(torch.abs(adj_data_ndft - img_data) ** 2)

    assert_almost_allclose(
        adj_data.cpu().detach(),
        adj_data_ndft.cpu().detach(),
        atol=1e-1,
        rtol=1e-1,
        mismatch=20,
    )

    # Check if nufft and ndft w.r.t trajectory are close in the backprop
    gradient_ndft_ktraj = torch.autograd.grad(
        loss_ndft, operator.samples, retain_graph=True
    )[0]
    gradient_nufft_ktraj = torch.autograd.grad(
        loss_nufft, operator.samples, retain_graph=True
    )[0]
    assert_almost_allclose(
        gradient_ndft_ktraj.cpu().numpy(),
        gradient_nufft_ktraj.cpu().numpy(),
        atol=1e-2,
        rtol=1e-2,
        mismatch=20,
    )

    # Check if nufft and ndft are close in the backprop
    gradient_ndft_kdata = torch.autograd.grad(loss_ndft, ksp_data, retain_graph=True)[0]
    gradient_nufft_kdata = torch.autograd.grad(loss_nufft, ksp_data, retain_graph=True)[
        0
    ]
    assert_allclose(
        gradient_ndft_kdata.cpu().numpy(), gradient_nufft_kdata.cpu().numpy(), atol=1e-2
    )


@pytest.mark.parametrize("interface", ["torch-gpu", "torch-cpu"])
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Pytorch is not installed")
def test_forward_and_grad(operator, interface):
    """Test the adjoint and gradient of the operator."""
    if operator.backend == "finufft" and "gpu" in interface:
        pytest.skip("GPU not supported for finufft backend")

    if "gpu" in interface:
        operator.samples = operator.samples.to("cuda")
    else:
        operator.samples = operator.samples.cpu()
    ksp_data_ref = to_interface(kspace_from_op(operator), interface=interface)
    img_data = to_interface(image_from_op(operator), interface=interface)
    img_data.requires_grad = True

    with torch.autograd.set_detect_anomaly(True):
        if operator.smaps is not None and operator.n_coils > 1:
            img_data = img_data[None, ...]
        ksp_data = operator.op(img_data).reshape(ksp_data_ref.shape)
        if operator.smaps is not None:
            smaps = torch.from_numpy(operator.smaps).to(ksp_data_ref.device)
            img_data_smaps = smaps * img_data
            ksp_data_ndft = torch.cat(
                [
                    (ndft_matrix(operator) @ img_data_smaps[i].flatten())[None, ...]
                    for i in range(img_data_smaps.shape[0])
                ],
                dim=0,
            )  # fft for each coil
        else:
            ksp_data_ndft = torch.matmul(
                ndft_matrix(operator),
                img_data.reshape(operator.n_coils, -1).squeeze().T,
            ).T.reshape(ksp_data.shape)

        loss_nufft = torch.mean(torch.abs(ksp_data - ksp_data_ref) ** 2)
        loss_ndft = torch.mean(torch.abs(ksp_data_ndft - ksp_data_ref) ** 2)

    # FIXME: This check can be tighter for Nyquist cases
    assert_almost_allclose(
        ksp_data.cpu().detach(),
        ksp_data_ndft.cpu().detach(),
        atol=1e-1,
        rtol=1e-1,
        mismatch=20,
    )

    # Check if nufft and ndft w.r.t trajectory are close in the backprop
    gradient_ndft_ktraj = torch.autograd.grad(
        loss_ndft, operator.samples, retain_graph=True
    )[0]
    gradient_nufft_ktraj = torch.autograd.grad(
        loss_nufft, operator.samples, retain_graph=True
    )[0]
    assert_allclose(
        gradient_ndft_ktraj.cpu().numpy(), gradient_nufft_ktraj.cpu().numpy(), atol=5e-1
    )

    # Check if nufft and ndft are close in the backprop
    gradient_ndft_kdata = torch.autograd.grad(loss_ndft, img_data, retain_graph=True)[0]
    gradient_nufft_kdata = torch.autograd.grad(loss_nufft, img_data, retain_graph=True)[
        0
    ]
    assert_allclose(
        gradient_ndft_kdata.cpu().numpy(), gradient_nufft_kdata.cpu().numpy(), atol=6e-3
    )
