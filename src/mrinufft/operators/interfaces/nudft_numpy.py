"""An implementation of the NUDFT using numpy."""

from .base import FourierOperatorCPU

import numpy as np


def get_fourier_matrix(ktraj, shape, ndim, do_ifft=False):
    """Get the NDFT Fourier Matrix."""
    n = np.prod(shape)
    matrix = np.zeros((n, n), dtype=complex)
    r = [np.arange(shape[i]) for i in range(ndim)]
    grid_r = np.reshape(np.meshgrid(*r, indexing="ij"), (ndim, np.prod(shape)))
    traj_grid = ktraj @ grid_r
    matrix = np.exp(-2j * np.pi * traj_grid)
    if do_ifft:
        matrix = matrix.conj().T
    return matrix / np.sqrt(n)


class RawNDFT:
    """Implementation of the NUDFT using numpy."""

    def __init__(self, samples, shape, explicit_matrix=False):
        self.samples = samples
        self.shape = shape
        self.n_samples = len(samples)
        self.ndim = len(shape)

        if explicit_matrix:
            self._fourier_matrix = get_fourier_matrix(
                self.samples, self.shape, self.ndim
            )
            self.op = lambda x: self._fourier_matrix @ x.flatten()
            self.adj_op = lambda x: np.reshape(
                (self._fourier_matrix.conj() @ x), self.shape
            )
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


class MRInumpy(FourierOperatorCPU):
    """MRI operator using numpy NUDFT backend.

    For testing purposes only, as it is very slow.
    """

    def __init_(self, samples, shape, n_coils=1, smaps=None, explicit_matrix=False):
        super().__init__(samples, shape, density=False, n_coils=n_coils, smaps=smaps)

        self.raw_op = RawNDFT(samples, shape, explicit_matrix=explicit_matrix)
