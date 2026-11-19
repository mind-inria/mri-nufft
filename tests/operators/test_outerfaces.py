"""Tests for the adapters exposing operators to external libraries."""

import numpy as np
import numpy.testing as npt
import pytest
from pytest import fixture, mark
from scipy.sparse.linalg import LinearOperator

from mrinufft import get_operator
from mrinufft._array_compat import DEEPINV_AVAILABLE, MRPRO_AVAILABLE

from helpers import (
    CasesBatch,
    CasesTrajectories,
    case_id,
    cases_of,
    image_from_op,
    kspace_from_op,
    random_smaps,
)

try:
    import torch
except ImportError:
    pass


@fixture(
    scope="module",
    params=cases_of(CasesTrajectories, "grid2D", "nyquist_lowmem_radial3D"),
    ids=case_id,
)
def trajectory(request):
    """Sampling pattern and image shape."""
    return request.param()


@fixture(
    scope="module",
    params=cases_of(CasesBatch, "single", "batch_coils", "batch_sense"),
    ids=case_id,
)
def config(request):
    """Batch, coils and SENSE configuration."""
    return request.param()


@fixture(scope="module", params=[True, False], ids=["squeeze", "full"])
def squeeze_dims(request):
    """Whether the operator squeezes its singleton dimensions."""
    return request.param


@fixture(scope="module")
def nufft(trajectory, config, squeeze_dims):
    """NUFFT operator, the adapters do not depend on the backend."""
    samples, shape = trajectory
    n_batchs, n_coils, n_trans, sense = config
    return get_operator("finufft")(
        samples.astype(np.float32),
        shape,
        n_batchs=n_batchs,
        n_coils=n_coils,
        n_trans=n_trans,
        smaps=random_smaps(n_coils, shape) if sense else None,
        squeeze_dims=squeeze_dims,
    )


def test_scipy_linop(nufft):
    """The linear operator applies the NUFFT on flattened arrays."""
    linop = nufft.make_linops()
    x, y = image_from_op(nufft), kspace_from_op(nufft)

    assert isinstance(linop, LinearOperator)
    assert linop.shape == (y.size, x.size)
    assert linop.dtype == nufft.cpx_dtype
    # finufft is not bitwise deterministic across calls.
    tol = dict(rtol=1e-5, atol=1e-5)
    npt.assert_allclose(linop @ x.ravel(), nufft.op(x).ravel(), **tol)
    npt.assert_allclose(linop.H @ y.ravel(), nufft.adj_op(y).ravel(), **tol)


@mark.skipif(not MRPRO_AVAILABLE, reason="mrpro not available")
def test_mrpro(nufft):
    """MRPro operator matches the NUFFT, and differentiates through its adjoint."""
    from mrpro.operators import LinearOperator

    operator = nufft.make_mrpro()
    assert isinstance(operator, LinearOperator)

    x = torch.from_numpy(image_from_op(nufft)).requires_grad_()
    y = torch.from_numpy(kspace_from_op(nufft))
    v = torch.randn_like(x)
    (forward,) = operator(x)
    (adjoint,) = operator.H(y)
    torch.testing.assert_close(forward, nufft.op(x.detach()).reshape(forward.shape))
    torch.testing.assert_close(adjoint, nufft.adj_op(y).reshape(adjoint.shape))

    # grad ||Ax||^2 = 2 A^H A x, and its derivative along v is 2 A^H A v.
    (grad,) = torch.autograd.grad(forward.abs().square().sum(), x, create_graph=True)
    (hvp,) = torch.autograd.grad(grad, x, v)
    torch.testing.assert_close(grad, 2 * operator.H(forward)[0])
    torch.testing.assert_close(hvp, 2 * operator.H(operator(v)[0])[0])


@mark.skipif(not MRPRO_AVAILABLE, reason="mrpro not available")
def test_mrpro_errors():
    """MRPro rejects density compensation and wrongly shaped inputs."""
    nufft = get_operator("finufft")(
        np.zeros((3, 2), dtype=np.float32), (4, 4), density=np.ones(3)
    )
    with pytest.raises(ValueError, match="density compensation"):
        nufft.make_mrpro()

    nufft.density = None
    operator = nufft.make_mrpro()
    with pytest.raises(ValueError, match="image shape"):
        operator(torch.zeros((4, 4), dtype=torch.complex64))
    with pytest.raises(ValueError, match="k-space shape"):
        operator.H(torch.zeros(3, dtype=torch.complex64))

    nufft.density = np.ones(3)
    with pytest.raises(ValueError, match="density compensation"):
        operator(torch.zeros(nufft.img_full_shape, dtype=torch.complex64))


@mark.skipif(not DEEPINV_AVAILABLE, reason="deepinv not available")
@mark.parametrize("viewed_as_real", [False, True])
def test_deepinv(nufft, viewed_as_real):
    """DeepInv physics matches the NUFFT, in complex or real view."""
    if nufft.squeeze_dims:
        pytest.skip("Autodiff does not support squeeze_dims.")
    from mrinufft.operators import DeepInvPhyNufft, image_as_real, kspace_as_real
    from mrinufft.operators.autodiff import DeepInvPhyNufft as legacy

    assert legacy is DeepInvPhyNufft
    physics = nufft.make_deepinv_phy(viewed_as_real=viewed_as_real)
    assert isinstance(physics, DeepInvPhyNufft)

    x = torch.from_numpy(image_from_op(nufft))
    y = torch.from_numpy(kspace_from_op(nufft))
    expected = nufft.op(x), nufft.adj_op(y), nufft.pinv_solver(y)
    if viewed_as_real:
        x, y = image_as_real(x), kspace_as_real(y)
        expected = kspace_as_real(expected[0]), *map(image_as_real, expected[1:])

    x.requires_grad_()
    torch.testing.assert_close(physics.A(x), expected[0])
    torch.testing.assert_close(physics.A_adjoint(y), expected[1])
    torch.testing.assert_close(physics.A_dagger(y), expected[2])
    physics.A(x).abs().square().sum().backward()
    assert x.grad is not None
