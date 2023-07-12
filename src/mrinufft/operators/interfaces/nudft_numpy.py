"""An implementation of the NUDFT using numpy."""

from .base import FourierOperatorCPU

import numpy as np


class RawNDFT:
    def __init__(self, samples, shape, explicit_matrix=False):
        self.samples = samples
        self.shape = shape
        self.n_samples = len(samples)
        self.ndim = len(shape)

        if explicit_matrix:
            self._fourier_matrix = self.get_fourier_matrix()
            self.op = lambda x: self._fourier_matrix @ x
            self.adj_op = lambda x: self._fourier_matrix.conj() @ x
        else:
            self.op = self._op_sum
            self.adj_op = self._adj_op_sum

    def _op_sum(self, x):
        """Compute the type 2 NUDFT."""

        y = np.zeros(self.n_samples, dtype=x.dtype)
        for i in range(self.n_samples):
            for j in range(self.ndim):
                y[i] += (
                    np.exp(
                        -1j * 2 * np.pi * self.samples[i, j] * np.arange(self.shape[j])
                    )
                    @ x[j, :]
                )
        return y

    def _adj_op_sum(self, x):
        """Compute the type 1 NUDFT."""
            y = np.zeros((self.ndim, np.prod(self.shape)), dtype=x.dtype)
            for i in range(self.n_samples):
                for j in range(self.ndim):
                    y[j, :] += (
                        np.exp(
                            1j * 2 * np.pi * self.samples[i, j] * np.arange(self.shape[j])
                        )
                        * x[i]
                    )
            return y

    def get_fourier_matrix(self):
        """Get the NDFT Fourier Matrix"""
        r = [
            np.linspace(-self.shape[i] / 2, self.shape[i] / 2 - 1, self.shape[i])
            for i in range(self.ndim)
        ]
        grid_r = np.reshape(
            np.meshgrid(*r, indexing="ij"), (self.ndim, np.prod(self.shape))
        )
        traj_grid = self.samples @ grid_r
        A = np.exp(-1j * traj_grid)
        scale = np.sqrt(np.prod(self.shape)) * np.power(np.sqrt(2), self.ndim)
        A = A / scale
        return A


class MRInumpy(FourierOperatorCPU):
    def __init_(self, samples, shape, n_coils=1, smaps=None, explicit_matrix=False):
        super().__init__(samples, shape, density=False, n_coils=n_coils, smaps=smaps)

        self.raw_op = RawNDFT(samples, shape, explicit_matrix=explicit_matrix)
