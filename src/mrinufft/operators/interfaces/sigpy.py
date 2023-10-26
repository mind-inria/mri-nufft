"""Sigpy NUFFT interface.

The SigPy NUFFT is fully implemented in Python.
"""

import numpy as np
from ..base import FourierOperatorCPU, proper_trajectory


SIGPY_AVAILABLE = True
try:
    import sigpy.fourier as sgf
except ImportError:
    SIGPY_AVAILABLE = False


class RawSigpyNUFFT:
    """Raw interface to SigPy NUFFT."""

    def __init__(self, samples, shape, oversamp=1.25, width=4, **kwargs):
        shape = np.array(shape)
        # scale in FOV/2 units
        self.samples = samples * shape

        self.shape = shape
        self._oversamp = oversamp
        self._width = width

    def op(self, coeffs_data, grid_data):
        """Forward Operator."""
        ret = sgf.nufft(
            grid_data,
            self.samples,
            oversamp=self._oversamp,
            width=self._width,
        )
        np.copyto(coeffs_data, ret)
        return coeffs_data

    def adj_op(self, coeffs_data, grid_data):
        """Adjoint Operator."""
        ret = sgf.nufft_adjoint(
            coeffs_data,
            self.samples,
            oshape=self.shape,
            oversamp=self._oversamp,
            width=self._width,
        )
        np.copyto(grid_data, ret)
        return grid_data


class MRISigpyNUFFT(FourierOperatorCPU):
    """NUFFT using SigPy.

    This is a wrapper around the SigPy NUFFT operator.
    """

    backend = "sigpy"
    available = SIGPY_AVAILABLE

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        n_batchs=1,
        smaps=None,
        **kwargs,
    ):
        samples_ = proper_trajectory(samples, normalize="unit")

        super().__init__(
            samples_,
            shape,
            density=density,
            n_coils=n_coils,
            n_batchs=n_batchs,
            n_trans=1,
            smaps=smaps,
        )

        self.raw_op = RawSigpyNUFFT(samples_, shape, **kwargs)

    @property
    def norm_factor(self):
        """Normalization factor of the operator."""
        return np.sqrt(2 ** len(self.shape))
