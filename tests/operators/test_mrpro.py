"""Tests for the MRPro linear operator interface."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
mrpro = pytest.importorskip("mrpro")

from mrinufft.operators import check_backend, get_operator


@pytest.mark.skipif(not check_backend("finufft"), reason="finufft is not installed")
@pytest.mark.parametrize(
    "shape,n_batchs,n_coils,sense,squeeze_dims",
    [
        ((4, 4), 1, 1, False, True),
        ((4, 4), 2, 2, False, False),
        ((4, 4), 2, 2, True, True),
        ((3, 4, 4), 1, 2, True, False),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mrpro_operator(shape, n_batchs, n_coils, sense, squeeze_dims, dtype):
    rng = np.random.default_rng(42)
    samples = rng.uniform(-0.4, 0.4, (12, len(shape))).astype(dtype)
    smaps = None
    if sense:
        smaps = (rng.standard_normal((n_coils, *shape)) + 1j).astype(
            np.complex64 if dtype == np.float32 else np.complex128
        )
    nufft = get_operator("finufft")(
        samples,
        shape,
        n_batchs=n_batchs,
        n_coils=n_coils,
        smaps=smaps,
        squeeze_dims=squeeze_dims,
    )
    operator = nufft.make_mrpro()
    assert isinstance(operator, mrpro.operators.LinearOperator)
    torch_dtype = torch.complex64 if dtype == np.float32 else torch.complex128
    x = torch.randn(nufft.img_full_shape, dtype=torch_dtype, requires_grad=True)
    y = torch.randn(nufft.ksp_full_shape, dtype=torch_dtype, requires_grad=True)

    (forward,) = operator(x)
    (adjoint,) = operator.H(y)
    assert forward.shape == nufft.ksp_full_shape
    assert adjoint.shape == nufft.img_full_shape
    torch.testing.assert_close(forward, nufft.op(x.detach()).reshape(forward.shape))
    torch.testing.assert_close(adjoint, nufft.adj_op(y.detach()).reshape(adjoint.shape))
    torch.testing.assert_close(
        torch.vdot(forward.flatten(), y.flatten()),
        torch.vdot(x.flatten(), adjoint.flatten()),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(operator.gram(x)[0], operator.H(forward)[0])
    grad_x = torch.autograd.grad(forward, x, y, retain_graph=True)[0]
    grad_y = torch.autograd.grad(adjoint, y, x)[0]
    torch.testing.assert_close(grad_x, adjoint.detach())
    torch.testing.assert_close(grad_y, forward.detach())


@pytest.mark.skipif(not check_backend("finufft"), reason="finufft is not installed")
def test_mrpro_density_and_shapes():
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
    with pytest.raises(ValueError, match="density compensation"):
        operator.H(torch.zeros(nufft.ksp_full_shape, dtype=torch.complex64))
