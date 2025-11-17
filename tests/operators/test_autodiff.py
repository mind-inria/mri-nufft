"""Test the autodiff functionnality."""

import numpy as np
from numpy.typing import NDArray
from numpy.testing import assert_allclose
from mrinufft.operators.interfaces.nudft_numpy import get_fourier_matrix
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture
from case_trajectories import CasesTrajectories
from mrinufft.operators import get_operator

from helpers import (
    kspace_from_op,
    image_from_op,
    to_interface,
    assert_almost_allclose,
    batched_smaps_from_op,
)


TORCH_AVAILABLE = True
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False


@fixture(scope="module")
@parametrize(backend=["cufinufft", "finufft", "gpunufft"])
@parametrize(paired_batch_size=[0, 1, 3])
@parametrize(
    "n_coils, n_batchs, sense",
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
def operator(
    request,
    kspace_loc: NDArray,
    shape: tuple[int, ...],
    n_coils: int,
    sense: bool,
    n_batchs: int,
    backend: str,
    paired_batch_size: bool,
):
    """Create NUFFT operator with autodiff capabilities."""
    if not sense and paired_batch_size:
        pytest.skip("Not relevant to test.")
    if sense:
        smaps = 1j * np.random.rand(n_coils, *shape)
        smaps += np.random.rand(n_coils, *shape)
        smaps = smaps.astype(np.complex64)
        smaps /= np.linalg.norm(smaps, axis=0)
    else:
        smaps = None
    kspace_loc = kspace_loc.astype(np.float32)
    nufft = get_operator(
        backend_name=backend,
        wrt_data=True,
        wrt_traj=True,
        paired_batch=paired_batch_size,  # a slight abuse of attribute.
    )(
        samples=kspace_loc,
        shape=shape,
        n_batchs=n_batchs,
        n_coils=n_coils,
        smaps=smaps,
        squeeze_dims=False,
    )
    return nufft


def ndft_matrix(operator):
    """Get the NDFT matrix from the operator."""
    return get_fourier_matrix(operator.samples, operator.shape, normalize=True)


def get_data(operator, interface):
    """Generate k-space and image data based on the interface."""
    if operator.paired_batch:
        ksp_data = to_interface(
            np.stack([kspace_from_op(operator) for _ in range(operator.paired_batch)]),
            interface=interface,
        )
        img_data = to_interface(
            np.stack([image_from_op(operator) for _ in range(operator.paired_batch)]),
            interface=interface,
        )
    else:
        ksp_data = to_interface(kspace_from_op(operator), interface=interface)
        img_data = to_interface(image_from_op(operator), interface=interface)
    return ksp_data, img_data


def compute_adjoint(operator, ksp_data, img_data_ref):
    """Compute adjoint imgs for non-batched mode."""
    adj_data = operator.adj_op(ksp_data).reshape(img_data_ref.shape)

    adj_matrix = ndft_matrix(operator).conj().T

    if operator.smaps is not None:
        smaps = torch.from_numpy(operator.smaps).to(img_data_ref.device)
        adj_data_ndft_smaps = torch.einsum(
            "nm,...m -> ...n", adj_matrix, ksp_data
        ).reshape((operator.n_batchs, operator.n_coils, *operator.shape))
        adj_data_ndft = torch.einsum(
            "c...,bc...->b...", smaps.conj(), adj_data_ndft_smaps
        )[:, None]
    else:
        adj_data_ndft = torch.einsum("nm,...m->...n", adj_matrix, ksp_data).reshape(
            operator.img_full_shape
        )
    return adj_data, adj_data_ndft


def compute_adjoint_batched(operator, ksp_data, img_data_ref):
    """Compute adjoint imgs for batched mode."""
    # paired batched mode will be tested only if sense is True
    smaps = batched_smaps_from_op(operator)

    adj_matrix = ndft_matrix(operator).conj().T
    adj_data = operator.adj_op(ksp_data, smaps=smaps).reshape(img_data_ref.shape)
    smaps = torch.from_numpy(smaps).to(img_data_ref.device)

    adj_data_ndft_smaps = torch.einsum("nm,...m -> ...n", adj_matrix, ksp_data).reshape(
        (operator.paired_batch, operator.n_batchs, operator.n_coils, *operator.shape)
    )
    adj_data_ndft = torch.einsum(
        "vc...,vbc...->vb...", smaps.conj(), adj_data_ndft_smaps
    )[:, :, None]
    return adj_data, adj_data_ndft


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

    ksp_data, img_data_ref = get_data(operator, interface)
    ksp_data.requires_grad = True

    is_batched = getattr(operator, "paired_batch", 0)

    with torch.autograd.set_detect_anomaly(True):
        if is_batched:
            adj_data, adj_data_ndft = compute_adjoint_batched(
                operator, ksp_data, img_data_ref
            )
        else:
            adj_data, adj_data_ndft = compute_adjoint(operator, ksp_data, img_data_ref)
        loss_nufft = torch.mean(torch.abs(adj_data - img_data_ref) ** 2)
        loss_ndft = torch.mean(torch.abs(adj_data_ndft - img_data_ref) ** 2)

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
    # FIXME: atol=5e-1 is too loose?
    assert_almost_allclose(
        gradient_ndft_ktraj.cpu().numpy(),
        gradient_nufft_ktraj.cpu().numpy(),
        atol=5e-1,
        rtol=5e-1,
        mismatch=20,
    )
    # Check if nufft and ndft are close in the backprop
    grad_ndft_kdata = torch.autograd.grad(loss_ndft, ksp_data, retain_graph=True)[0]
    grad_nufft_kdata = torch.autograd.grad(loss_nufft, ksp_data, retain_graph=True)[0]
    assert_allclose(
        grad_ndft_kdata.cpu().numpy(),
        grad_nufft_kdata.cpu().numpy(),
        atol=6e-3,
        rtol=6e-3,
    )


def compute_forward(operator, ksp_data_ref, img_data):
    """Compute ksps for non-batched mode."""
    ksp_data = operator.op(img_data).reshape(ksp_data_ref.shape)
    if operator.uses_sense:
        smaps = torch.from_numpy(operator.smaps).to(ksp_data_ref.device)
        img_data_smaps = smaps * img_data
        ksp_data_ndft = torch.einsum(
            "mn,...n->...m",
            ndft_matrix(operator),
            img_data_smaps.reshape(operator.n_batchs, operator.n_coils, -1),
        )
    else:
        ksp_data_ndft = torch.einsum(
            "mn,...n->...m",
            ndft_matrix(operator),
            img_data.reshape(operator.n_batchs, operator.n_coils, -1),
        )
    return ksp_data, ksp_data_ndft


def compute_forward_batched(operator, ksp_data_ref, img_data):
    """Compute ksps for batched mode."""
    smaps = batched_smaps_from_op(operator)
    ksp_data = operator.op(img_data, smaps=smaps).reshape(ksp_data_ref.shape)
    smaps = torch.from_numpy(smaps).to(ksp_data_ref.device)
    img_data_smaps = smaps[:, None] * img_data
    ksp_data_ndft = torch.einsum(
        "mn,...n->...m",
        ndft_matrix(operator),
        img_data_smaps.reshape(
            operator.paired_batch, operator.n_batchs, operator.n_coils, -1
        ),
    )
    return ksp_data, ksp_data_ndft


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

    ksp_data_ref, img_data = get_data(operator, interface)
    img_data.requires_grad = True

    is_batched = getattr(operator, "paired_batch", 0)
    with torch.autograd.set_detect_anomaly(True):
        if is_batched:
            ksp_data, ksp_data_ndft = compute_forward_batched(
                operator, ksp_data_ref, img_data
            )
        else:
            ksp_data, ksp_data_ndft = compute_forward(operator, ksp_data_ref, img_data)
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

    # Check if nufft and ndft w.r.t image  are close in the backprop
    grad_ndft_kdata = torch.autograd.grad(loss_ndft, img_data, retain_graph=True)[0]
    grad_nufft_kdata = torch.autograd.grad(loss_nufft, img_data, retain_graph=True)[0]
    assert_allclose(
        grad_ndft_kdata.cpu().numpy(),
        grad_nufft_kdata.cpu().numpy(),
        atol=6e-3,
    )

    # Check if nufft and ndft w.r.t trajectory are close in the backprop
    grad_ndft_ktraj = torch.autograd.grad(
        loss_ndft, operator.samples, retain_graph=True
    )[0]
    grad_nufft_ktraj = torch.autograd.grad(
        loss_nufft, operator.samples, retain_graph=True
    )[0]

    assert_allclose(
        grad_ndft_ktraj.cpu().numpy(),
        grad_nufft_ktraj.cpu().numpy(),
        atol=5e-1,
    )
