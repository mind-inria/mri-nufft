"""Sigpy NUFFT interface.

The SigPy NUFFT is fully implemented in Python.
"""

import warnings

import numpy as np
from mrinufft._utils import proper_trajectory
from mrinufft.operators.base import FourierOperatorCPU


SIGPY_AVAILABLE = True
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import sigpy.fourier as sgf
except ImportError:
    SIGPY_AVAILABLE = False


class RawSigpyNUFFT:
    """Raw interface to SigPy output /= width**ndim NUFFT.

    Parameters
    ----------
    samples: np.ndarray
        the kspace sample locations in the Fourier domain,
        normalized between -0.5 and 0.5
    shape: tuple of int
        shape of the image
    oversamp: float default 1.25
        oversampling factor
    width: int default 4
        interpolation kernel width (usually 3 to 7)
    upsampfac: float, default 1.25
        Same as oversamp
    eps: float, default 1e-4
        Other way of specifiying width.
    """

    def __init__(
        self,
        samples,
        shape,
        oversamp=1.25,
        width=4,
        eps=None,
        upsampfac=None,
        n_trans=1,
        **kwargs,
    ):
        if upsampfac is not None:
            oversamp = upsampfac
        if eps is not None:
            width = -int(np.log10(eps))

        self.shape = shape
        shape = np.array(shape)
        # scale in FOV/2 units
        self.samples = samples * shape
        self.n_trans = n_trans
        self._oversamp = oversamp
        self._width = width

    def op(self, coeffs_data, grid_data):
        """Forward Operator."""
        grid_data_ = grid_data.reshape(self.n_trans, *self.shape)
        ret = sgf.nufft(
            grid_data_,
            self.samples,
            oversamp=self._oversamp,
            width=self._width,
        )
        ret = ret.reshape(self.n_trans, len(self.samples))
        np.copyto(coeffs_data, ret)
        return coeffs_data

    def adj_op(self, coeffs_data, grid_data):
        """Adjoint Operator."""
        coeffs_data_ = coeffs_data.reshape(self.n_trans, len(self.samples))
        ret = sgf.nufft_adjoint(
            coeffs_data_,
            self.samples,
            oshape=(self.n_trans, *self.shape),
            oversamp=self._oversamp,
            width=self._width,
        )
        ret = ret.reshape(self.n_trans, *self.shape)
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
        n_trans=1,
        smaps=None,
        squeeze_dims=True,
        **kwargs,
    ):
        samples_ = proper_trajectory(samples, normalize="unit")
        raw_op = RawSigpyNUFFT(samples_, shape, n_trans=n_trans, **kwargs)

        super().__init__(
            samples_,
            shape,
            density=density,
            n_coils=n_coils,
            n_batchs=n_batchs,
            n_trans=n_trans,
            smaps=smaps,
            raw_op=raw_op,
            squeeze_dims=squeeze_dims,
        )

    @property
    def norm_factor(self):
        """Normalization factor of the operator."""
        return np.sqrt(2 ** len(self.shape))
