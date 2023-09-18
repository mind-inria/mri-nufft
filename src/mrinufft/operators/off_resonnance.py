"""Off Resonance correction Operator wrapper.

Based on the implementation of Guillaume Daval-Frérot in pysap-mri:
https://github.com/CEA-COSMIC/pysap-mri/blob/master/mri/operators/fourier/orc_wrapper.py
"""
import numpy as np

from .base import FourierOperatorBase
from .interfaces.utils import is_cuda_array

CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False


class MRIFourierCorrected(FourierOperatorBase):
    """Fourier Operator with B0 Inhomogeneities compensation.

    This is a wrapper around the Fourier Operator to compensate for the
    B0 inhomogeneities  in the  k-space.

    Parameters
    ----------
    fourier_op: object of class FourierBase
        the fourier operator to wrap
    B: numpy.ndarray
    C: numpy.ndarray
    indices: numpy.ndarray
    backend: str, default 'cpu'
        the backend to use for computations. Either 'cpu' or 'gpu'.
    """

    def __init__(self, fourier_op, B, C, indices, backend="cpu"):
        if backend == "gpu" and not CUPY_AVAILABLE:
            raise RuntimeError("Cupy is required for gpu computations.")
        if backend == "gpu":
            self.xp = cp
        elif backend == "cpu":
            self.xp = np
        else:
            raise ValueError("Unsupported backend.")
        self._fourier_op = fourier_op

        if not fourier_op.uses_sense:
            raise ValueError("please use smaps.")

        self.n_samples = fourier_op.n_samples
        self.n_coils = fourier_op.n_coils
        self.shape = fourier_op.shape
        self.smaps = fourier_op.smaps
        self.n_interpolators = len(C)
        self.B = self.xp.array(B)
        self.B = self.xp.tile(self.B, (self._fourier_op.n_samples // len(B), 1))
        self.C = self.xp.array(C)
        self.indices = indices

    def op(self, data, *args):
        """Compute Forward Operation with off-resonnances effect.

        Parameters
        ----------
        x: numpy.ndarray or cupy.ndarray
            N-D input image

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            masked distorded N-D k-space
        """
        y = self.xp.zeros((self.n_coils, self.n_samples), dtype=np.complex64)
        data_d = self.xp.asarray(data)
        for idx in range(self.n_interpolators):
            y += self.B[..., idx] * self._fourier_op.op(
                self.C[idx, self.indices] * data_d, *args
            )
        if self.xp.__name__ == "cupy" and is_cuda_array(data):
            return y
        return y.get()

    def adj_op(self, coeffs, *args):
        """
        Compute Adjoint Operation with off-resonnance effect.

        Parameters
        ----------
        x: numpy.ndarray or cupy.ndarray
            masked distorded N-D k-space

        Returns
        -------
            inverse Fourier transform of the distorded input k-space.
        """
        y = self.xp.zeros(self.shape, dtype=np.complex64)
        coeffs_d = self.xp.array(coeffs)
        for idx in range(self.n_interpolators):
            y += cp.conj(self.C[idx, self.indices]) * self._fourier_op.adj_op(
                cp.conj(self.B[..., idx]) * coeffs_d, *args
            )
        if self.xp.__name__ == "cupy" and is_cuda_array(coeffs):
            return y
        return y.get()

    def data_consistency(self, image_data, obs_data):
        """Compute the data consistency error.

        Parameters
        ----------
        image_data: numpy.ndarray or cupy.ndarray
            N-D input image
        obs_data: numpy.ndarray or cupy.ndarray
            N-D observed k-space

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            data consistency error in image space.
        """
        return self.adj_op(self.op(image_data) - obs_data)
