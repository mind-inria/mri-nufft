"""Adapters exposing MRI-NUFFT operators to external libraries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mrinufft._array_compat import CUPY_AVAILABLE, DEEPINV_AVAILABLE, MRPRO_AVAILABLE

if DEEPINV_AVAILABLE:
    from deepinv.physics.forward import LinearPhysics

    from .autodiff import MRINufftAutoGrad, complex_view_wrapper
if MRPRO_AVAILABLE:
    from mrpro.operators import LinearOperator as MRProLinearOperator

if TYPE_CHECKING:
    from torch import Tensor


def make_scipy_linop(nufft, *, cupy=False):
    """Create a SciPy or CuPy linear operator wrapping a NUFFT operator."""
    if cupy and not CUPY_AVAILABLE:
        raise ValueError("cupy is not available")
    elif cupy:
        from cupyx.scipy.sparse.linalg import LinearOperator
    else:
        from scipy.sparse.linalg import LinearOperator

    linop = LinearOperator(
        shape=(np.prod(nufft.ksp_full_shape), np.prod(nufft.img_full_shape)),
        matvec=lambda x: nufft.op(x.reshape(nufft.img_full_shape)).ravel(),  # type: ignore
        rmatvec=lambda x: nufft.adj_op(x.reshape(nufft.ksp_full_shape)).ravel(),  # type: ignore
        dtype=nufft.cpx_dtype,
    )
    linop._nufft = nufft  # type: ignore
    return linop


if DEEPINV_AVAILABLE:

    class DeepInvPhyNufft(LinearPhysics):
        """Expose an MRINufftAutoGrad as a DeepInv Physics Operator."""

        def __init__(self, autograd_nufft, viewed_as_real=False):
            if not isinstance(autograd_nufft, MRINufftAutoGrad):
                raise ValueError(
                    "autograd_nufft should be an instance of MRINufftAutoGrad"
                )
            super().__init__()
            # since autograd_nufft is a nn.Module, we need to set it this way
            # to avoid registering it as a sub-module / parameter.
            self.__dict__["_operator"] = autograd_nufft
            self.viewed_as_real = viewed_as_real

        @complex_view_wrapper(in_space="img", out_space="ksp")
        def A(self, x: Tensor, **kwargs) -> Tensor:
            """Forward operation."""
            return self._operator.op(x, **kwargs)

        @complex_view_wrapper(in_space="ksp", out_space="img")
        def A_adjoint(self, y: Tensor, **kwargs) -> Tensor:
            """Adjoint operation."""
            return self._operator.adj_op(y, **kwargs)

        @complex_view_wrapper(in_space="ksp", out_space="img")
        def A_dagger(self, y: Tensor, **kwargs) -> Tensor:
            """Pseudo-inverse operation."""
            return self._operator.nufft_op.pinv_solver(y, **kwargs)

        def __getattr__(self, name):
            """Get attribute."""
            try:
                return super().__getattr__(name)
            except AttributeError:
                operator = super().__getattr__("_operator")
                return getattr(operator, name)


if MRPRO_AVAILABLE:

    class MRProNUFFT(MRProLinearOperator, adjoint_as_backward=True):
        """Expose an MRI-NUFFT operator as an MRPro linear operator."""

        def __init__(self, nufft):
            super().__init__()
            self.nufft = nufft
            self._check_density()

        def _check_density(self):
            if self.nufft.uses_density:
                raise ValueError(
                    "MRPro requires an operator without density compensation."
                )

        def forward(self, x: Tensor) -> tuple[Tensor]:
            """Apply the forward transform."""
            self._check_density()
            if x.shape != self.nufft.img_full_shape:
                raise ValueError(f"Expected image shape {self.nufft.img_full_shape}.")
            return (self.nufft.op(x).reshape(self.nufft.ksp_full_shape),)

        def adjoint(self, x: Tensor) -> tuple[Tensor]:
            """Apply the adjoint transform."""
            self._check_density()
            if x.shape != self.nufft.ksp_full_shape:
                raise ValueError(f"Expected k-space shape {self.nufft.ksp_full_shape}.")
            return (self.nufft.adj_op(x).reshape(self.nufft.img_full_shape),)
