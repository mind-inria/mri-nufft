"""Tests for external operator interfaces."""

import subprocess
import sys

import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator

from mrinufft.operators import check_backend, get_operator
from mrinufft.operators import outerfaces


@pytest.mark.parametrize("sense", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scipy_linop(sense, dtype):
    samples = np.zeros((5, 2), dtype=dtype)
    smaps = np.ones((2, 4, 4), dtype=np.complex64) if sense else None
    nufft = get_operator("numpy")(samples, (4, 4), n_coils=2, smaps=smaps)
    linop = nufft.make_linops()
    x = np.arange(np.prod(nufft.img_full_shape), dtype=nufft.cpx_dtype)
    y = np.arange(np.prod(nufft.ksp_full_shape), dtype=nufft.cpx_dtype)

    assert isinstance(linop, LinearOperator)
    assert linop._nufft is nufft
    assert linop.shape == (y.size, x.size)
    assert linop.dtype == nufft.cpx_dtype
    np.testing.assert_allclose(
        linop @ x, nufft.op(x.reshape(nufft.img_full_shape)).ravel()
    )
    np.testing.assert_allclose(
        linop.H @ y, nufft.adj_op(y.reshape(nufft.ksp_full_shape)).ravel()
    )


def test_scipy_linop_missing_cupy(monkeypatch):
    nufft = get_operator("numpy")(np.zeros((3, 2), dtype=np.float32), (4, 4))
    monkeypatch.setattr(outerfaces, "CUPY_AVAILABLE", False)
    with pytest.raises(ValueError, match="cupy is not available"):
        nufft.make_linops(cupy=True)


def test_outerfaces_without_optional_frameworks():
    code = """
import numpy as np
import mrinufft._array_compat as compat
compat.DEEPINV_AVAILABLE = compat.MRPRO_AVAILABLE = False
from mrinufft.operators import get_operator, outerfaces
assert not hasattr(outerfaces, 'DeepInvPhyNufft')
assert not hasattr(outerfaces, 'MRProNUFFT')
get_operator('numpy')(np.zeros((3, 2), dtype=np.float32), (4, 4)).make_linops()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_mrpro_missing_dependency(monkeypatch):
    nufft = get_operator("numpy")(np.zeros((3, 2), dtype=np.float32), (4, 4))
    monkeypatch.setattr("mrinufft.operators.base.MRPRO_AVAILABLE", False)
    with pytest.raises(ImportError, match="pip install mrpro"):
        nufft.make_mrpro()


def test_autodiff_without_deepinv():
    pytest.importorskip("torch")
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'deepinv' or name.startswith('deepinv.'):
        raise ImportError('deepinv blocked')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from mrinufft.operators.autodiff import MRINufftAutoGrad, image_as_cpx
from mrinufft.operators import image_as_real, check_backend, get_operator
if check_backend('finufft'):
    import numpy as np
    import torch
    nufft = get_operator('finufft')(
        np.zeros((3, 2), dtype=np.float32), (4, 4), squeeze_dims=False
    )
    x = torch.randn(nufft.img_full_shape, dtype=torch.complex64, requires_grad=True)
    nufft.make_autograd().op(x).abs().square().sum().backward()
    assert x.grad is not None
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.skipif(not check_backend("finufft"), reason="finufft is not installed")
@pytest.mark.parametrize("viewed_as_real", [False, True])
def test_deepinv_interface(viewed_as_real, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("deepinv")
    from deepinv.physics.forward import LinearPhysics
    from mrinufft.operators import DeepInvPhyNufft as exported
    from mrinufft.operators.autodiff import (
        DeepInvPhyNufft as legacy,
        image_as_real,
        kspace_as_real,
    )
    from mrinufft.operators.outerfaces import DeepInvPhyNufft

    assert exported is legacy is DeepInvPhyNufft
    samples = np.zeros((5, 2), dtype=np.float32)
    nufft = get_operator("finufft")(samples, (4, 4), squeeze_dims=False)
    physics = nufft.make_deepinv_phy(viewed_as_real=viewed_as_real)
    assert isinstance(physics, (DeepInvPhyNufft, LinearPhysics))
    x = torch.randn(nufft.img_full_shape, dtype=torch.complex64)
    y = torch.randn(nufft.ksp_full_shape, dtype=torch.complex64)
    expected_forward = nufft.op(x)
    expected_adjoint = nufft.adj_op(y)
    monkeypatch.setattr(nufft, "pinv_solver", lambda data: nufft.adj_op(data))
    if viewed_as_real:
        x, y = image_as_real(x), kspace_as_real(y)
        expected_forward = kspace_as_real(expected_forward)
        expected_adjoint = image_as_real(expected_adjoint)
    x.requires_grad_()
    torch.testing.assert_close(physics.A(x), expected_forward)
    torch.testing.assert_close(physics.A_adjoint(y), expected_adjoint)
    torch.testing.assert_close(physics.A_dagger(y), expected_adjoint)
    physics.A(x).abs().square().sum().backward()
    assert x.grad is not None
