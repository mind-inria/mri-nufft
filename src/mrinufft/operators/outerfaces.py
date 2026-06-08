"""Framework wrapper interfaces (Deepinv, MRPro).

This module provides thin wrapper classes that expose mri-nufft operators
as first-class citizens in popular imaging frameworks.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mrinufft.operators.base import FourierOperatorBase
from mrinufft._array_compat import (
    CUPY_AVAILABLE,
    MRPRO_AVAILABLE,
    NP2TORCH,
    _array_to_torch,
    AUTOGRAD_AVAILABLE,
    DEEPINV_AVAILABLE,
)

if TYPE_CHECKING:
    import torch


class ScipyLinearOperatorInterface:
    """Expose an mri-nufft operator as a SciPy/Cupy LinearOperator.

    Parameters
    ----------
    nufft_op : FourierOperatorBase
        A non-differentiable mri-nufft operator instance.

    Notes
    -----
    This wrapper allows the mri-nufft operator to be used with SciPy's sparse
    linear algebra routines, such as iterative solvers.
    """

    def __init__(self, nufft_op, cupy: bool = False):
        if not isinstance(nufft_op, FourierOperatorBase):
            raise ValueError("nufft_op should be an instance of FourierOperatorBase")
        self._nufft = nufft_op

        if cupy and not CUPY_AVAILABLE:
            raise ValueError("cupy is not available")
        elif cupy:
            from cupyx.scipy.sparse.linalg import LinearOperator
        else:
            from scipy.sparse.linalg import LinearOperator

        # Build input shape: (batch, coil, *shape)
        ishape = self._nufft.img_full_shape
        # Build output shape: (batch, coil, n_samples)
        oshape = self._nufft.ksp_full_shape

        # Initialize SciPy LinearOperator
        LinearOperator.__init__(
            self,
            dtype=np.complex64,
            shape=(np.prod(oshape), np.prod(ishape)),
            matvec=self._matvec,
            rmatvec=self._rmatvec,
        )
        self._nufft = nufft_op

    def _matvec(self, x):
        """Forward operation (image -> k-space)."""
        x_reshaped = x.reshape(self._nufft.img_full_shape)
        return self._nufft.op(x_reshaped).ravel()

    def _rmatvec(self, y):
        """Adjoint operation (k-space -> image)."""
        y_reshaped = y.reshape(self._nufft.ksp_full_shape)
        return self._nufft.adj_op(y_reshaped).ravel()


class DeepInvPhyNufft:
    """Expose an MRINufftAutoGrad as a DeepInv Physics Operator.

    Parameters
    ----------
    autograd_nufft : MRINufftAutoGrad
        An mri-nufft operator wrapped with autograd support.

    Notes
    -----
    Since `autograd_nufft` is a ``nn.Module``, it is stored via ``__dict__``
    to avoid registering it as a sub-module / parameter of the DeepInv
    physics object.
    """

    def __init__(self, autograd_nufft):
        try:
            from deepinv.physics.forward import LinearPhysics
        except ImportError:
            raise ImportError(
                "DeepInv is not installed. Install it with: pip install deepinv"
            ) from None

        from mrinufft.operators.autodiff import MRINufftAutoGrad

        if not isinstance(autograd_nufft, MRINufftAutoGrad):
            raise ValueError("autograd_nufft should be an instance of MRINufftAutoGrad")

        LinearPhysics.__init__(self)

        # Avoid nn.Module registration
        self.__dict__["_operator"] = autograd_nufft

    def A(self, x, **kwargs):
        """Forward operation (image -> k-space)."""
        return self._operator.op(x, **kwargs)

    def A_adjoint(self, y, **kwargs):
        """Adjoint operation (k-space -> image)."""
        return self._operator.adj_op(y, **kwargs)

    def A_dagger(self, y, **kwargs):
        """Pseudo-inverse operation (k-space -> image)."""
        return self._operator.nufft_op.pinv_solver(y, **kwargs)

    def __getattr__(self, name):
        """Delegate attribute access to the underlying autograd operator."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            operator = super().__getattr__("_operator")
            return getattr(operator, name)


class MRProNufftInterface:
    """Expose an mri-nufft operator as an MRPro LinearOperator.

    Parameters
    ----------
    nufft_op : FourierOperatorBase
        A non-differentiable mri-nufft operator instance.
    wrt_data : bool, default True
        If True, enable gradient computation with respect to image data.
    wrt_traj : bool, default False
        If True, enable gradient computation with respect to trajectory samples.
    paired_batch : bool, default False
        If True, support paired batched data/smaps.

    Notes
    -----
    MRPro's ``LinearOperator`` inherits from ``Operator[Tin, tuple[Tout]]``
    with ``adjoint_as_backward=True``. Since MRPro's ``Operator.forward()``
    expects the output to be returned as a tuple, this wrapper automatically
    wraps the k-space output in a single-element tuple.
    """

    def __init__(
        self,
        nufft_op: FourierOperatorBase,
        wrt_data: bool = True,
        wrt_traj: bool = False,
        paired_batch: bool = False,
    ):
        if not MRPRO_AVAILABLE:
            raise ImportError(
                "MRPro is not installed. Install it with: pip install mrpro"
            ) from None

        if not AUTOGRAD_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed. Install it with: pip install torch"
            ) from None

        if not nufft_op.autograd_available:
            raise ValueError(
                f"Backend '{nufft_op.backend}' does not support auto-differentiation."
            )

        from mrinufft.operators.autodiff import MRINufftAutoGrad

        self._nufft = nufft_op
        self._autograd = nufft_op.make_autograd(
            wrt_data=wrt_data, wrt_traj=wrt_traj, paired_batch=paired_batch
        )

        # MRPro LinearOperator base class
        from mrpro.algorithms.linear_operator import LinearOperator

        # Build input shape: (batch, coil, *shape)
        ishape = self._nufft.img_full_shape
        # Build output shape: (batch, coil, n_samples)
        oshape = self._nufft.ksp_full_shape

        # Initialize MRPro LinearOperator
        LinearOperator.__init__(
            self,
            ishape=ishape,
            oshape=oshape,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Forward operation (image -> k-space).

        Parameters
        ----------
        x : torch.Tensor
            Image data.

        Returns
        -------
        tuple[torch.Tensor]
            A single-element tuple containing k-space data.
        """
        kspace = self._autograd.op(x)
        return (kspace,)

    def adjoint(self, y: tuple[torch.Tensor]) -> torch.Tensor:
        """Adjoint operation (k-space -> image).

        Parameters
        ----------
        y : tuple[torch.Tensor]
            A single-element tuple containing k-space data.

        Returns
        -------
        torch.Tensor
            Reconstructed image data.
        """
        kspace = y[0]
        return self._autograd.adj_op(kspace)

    @property
    def ishape(self):
        """Input shape (image space)."""
        return self._nufft.img_full_shape

    @property
    def oshape(self):
        """Output shape (k-space)."""
        return self._nufft.ksp_full_shape
