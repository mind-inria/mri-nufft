"""Cartesian Operator for MRI reconstruction.

This module defines the MRICartesianOperator class, which implements the forward
and adjoint operations for Cartesian MRI reconstruction.
"""

from __future__ import annotations
from typing import Literal

from numpy.typing import NDArray

from mrinufft._array_compat import _to_interface, get_array_module, with_numpy_cupy
from mrinufft.extras.cartesian import fft, ifft

from .base import FourierOperatorSimple


class RawMRICartesianOperator:
    """Raw Cartesian Operator for MRI reconstruction.

    This class implements the forward and adjoint operations for Cartesian MRI
    reconstruction without any additional functionality. It serves as a
    low-level interface for the MRICartesianOperator class.
    """

    def __init__(self, mask: NDArray, shape: tuple[int, ...]):
        self.mask = mask
        self.shape = shape

    def op(self, ret, x):
        """Forward operation for Cartesian MRI reconstruction."""
        xp = get_array_module(ret)
        mask = _to_interface(self.mask, xp)
        xp.copyto(ret, fft(x, dims=len(self.shape))[:, mask])
        return ret

    def adj_op(self, y, ret):
        """Adjoint operation for Cartesian MRI reconstruction."""
        xp = get_array_module(ret)
        mask = _to_interface(self.mask, xp)
        y_ = xp.zeros((y.shape[0], *mask.shape), dtype=y.dtype)
        y_[:, mask] = y
        xp.copyto(ret, ifft(y_, dims=len(self.shape)))

        return ret


class MRICartesianOperator(FourierOperatorSimple):
    """Cartesian Operator for MRI reconstruction.

    This class implements the forward and adjoint operations for Cartesian MRI
    reconstruction. It inherits from the FourierOperatorBase class, which
    provides common functionality for Fourier-based operators.
    """

    backend = "cartesian"
    available = True

    def __init__(
        self,
        mask: NDArray,
        shape: tuple[int, ...],
        density: Literal[False] = False,
        n_coils: int = 1,
        n_batchs: int = 1,
        n_trans: int = 1,
        smaps: NDArray | None = None,
        squeeze_dims: bool = True,
    ):
        """Initialize the MRICartesianOperator.

        Args:
            shape (tuple): The shape of the input image.
            field_map (torch.Tensor, optional): The field map for the operator.
                If None, a default field map will be used. Defaults to None.
        """
        xp = get_array_module(mask)

        super().__init__(
            samples=xp.zeros(
                (1, len(shape)), dtype=xp.float32
            ),  # dummy samples, not used in Cartesian
            shape=shape,
            density=False,
            n_coils=n_coils,
            n_batchs=n_batchs,
            n_trans=n_trans,
            raw_op=RawMRICartesianOperator(mask, shape),
            squeeze_dims=squeeze_dims,
        )
        self.op = with_numpy_cupy(self.op)
        self.adj_op = with_numpy_cupy(self.adj_op)
        self.data_consistency = with_numpy_cupy(self.data_consistency)

    @property
    def mask(self):
        """Mask of the Cartesian sampling pattern."""
        return self.raw_op.mask

    @property
    def n_samples(self):
        """Number of samples in the Cartesian sampling pattern."""
        return sum(self.mask.flatten())  # count non-zeros element in the mask

    def compute_density(self, method=None):
        """Density is not supported for Cartesian Operator."""
        return None
