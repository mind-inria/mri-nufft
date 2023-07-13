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

    def __init__(self, samples, shape):
        self.samples = samples
        self.shape = shape
        self.n_samples = len(samples)
        self.ndim = len(shape)

        self._fourier_matrix = get_fourier_matrix(self.samples, self.shape, self.ndim)

    def op(self, image, coeffs):
        """Compute the forward NUDFT."""
        coeffs = self._fourier_matrix.T @ image.flatten()
        return coeffs

    def adj_op(self, coeffs, image):
        """Compute the adjoint NUDFT."""
        image = np.reshape((self._fourier_matrix.conj() @ coeffs), self.shape)
        return image


class MRInumpy(FourierOperatorCPU):
    """MRI operator using numpy NUDFT backend.

    For testing purposes only, as it is very slow.
    """

    def __init_(self, samples, shape, n_coils=1, smaps=None):
        rndft = RawNDFT(samples, shape)

        assert rndft is not None
        self.raw_op = rndft
        super().__init__(
            samples, shape, density=False, n_coils=n_coils, smaps=smaps, raw_op=rndft
        )
