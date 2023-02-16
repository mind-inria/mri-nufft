"""Off Resonance correction Operator wrapper.


Based on the implementation of Guillaume Daval-Frérot in pysap-mri:
https://github.com/CEA-COSMIC/pysap-mri/blob/master/mri/operators/fourier/orc_wrapper.py
"""
import numpy as np

from .interfaces.base import FourierOperatorBase
from .interfaces.gpu.utils import is_cuda_array

CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False


class MRIFourierCorrected(FourierOperatorBase):
    """Fourier Operator with B0 Inhomogeneities compensation."""

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

        self._uses_sense = fourier_op.uses_sense
        if not self._uses_sense:
            raise ValueError("please use smaps.")

        self.n_samples = fourier_op.n_samples
        self.n_coils = fourier_op.n_coils
        self.shape = fourier_op.shape
        self.n_interpolators = len(C)
        self.B = self.xp.array(B)
        self.B = self.xp.tile(self.B, (self._fourier_op.n_samples // len(B), 1))
        self.C = self.xp.array(C)
        self.indices = indices

    def op(self, data, *args):
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
        x: numpy.ndarray
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
        return self.adj_op(self.op(image_data) - obs_data)
