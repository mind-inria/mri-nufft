"""
Off Resonance correction Operator wrapper.
"""
import cupy as cp
import numpy as np

from .base import FourierOperatorBase
from ..utils  import is_cuda_array

class MRIFourierCorrected(FourierOperatorBase):
    """Fourier Operator with B0 Inhomogeneities compensation."""

    def __init__(self, fourier_op, B, C, indices):
        self._fourier_op = fourier_op

        self.uses_sense = fourier_op.uses_sense
        if not self.uses_sense:
            raise ValueError("please use smaps.")

        self.n_samples = fourier_op.n_samples
        self.n_coils = fourier_op.n_coils
        self.shape = fourier_op.shape

        self.B = cp.array(B)
        self.C = cp.array(C)
        self.indices = indices

    def op(self, data, *args):
        y = cp.zeros(self.n_coils, self.n_samples, dtype=np.complex64)
        data_d = cp.asarray(data)
        for l in range(len(self.C)):
            y += self.B[..., l] * self.fourier_op.op(
                self.C[l, self.indices] * data_d, *args
            )
        if is_cuda_array(data):
            return y
        return y.get()


    def adj_op(self, coeffs, *args):
        """
        This method calculates an inverse masked Fourier
        transform of a distorded N-D k-space.

        Parameters
        ----------
        x: numpy.ndarray
            masked distorded N-D k-space
        Returns
        -------
            inverse Fourier transform of the distorded input k-space.
        """
        y = cp.zeros(self.shape, dtype=np.complex64)
        coeffs_d = cp.asatrray(coeffs)
        for l in range(self.num_interpolators):
            y += cp.conj(self.C[l, self.indices]) * self.fourier_op.adj_op(
                cp.conj(self.B[..., l]) * coeffs_d,
                *args
            )
        if is_cuda_array(coeffs):
            return y
        return y.get()

    def data_consistency(self, image_data, obs_data):
        return self.adj_op(self.op(image_data) - obs_data)
