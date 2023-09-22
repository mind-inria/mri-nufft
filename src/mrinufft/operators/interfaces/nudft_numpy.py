"""An implementation of the NUDFT using numpy."""

import warnings

import numpy as np
import scipy as sp

from ..base import FourierOperatorCPU


def get_fourier_matrix(ktraj, shape):
    """Get the NDFT Fourier Matrix."""
    n = np.prod(shape)
    ndim = len(shape)
    matrix = np.zeros((len(ktraj), n), dtype=complex)
    r = [np.arange(shape[i]) for i in range(ndim)]
    grid_r = np.reshape(np.meshgrid(*r, indexing="ij"), (ndim, np.prod(shape)))
    traj_grid = ktraj @ grid_r
    matrix = np.exp(-2j * np.pi * traj_grid)
    return matrix


def implicit_type2_ndft(ktraj, image, shape):
    """Compute the NDFT using the implicit type 2 (image -> kspace) algorithm."""
    r = [np.arange(s) for s in shape]
    grid_r = np.reshape(
        np.meshgrid(*r, indexing="ij"), (len(shape), np.prod(image.shape))
    )
    res = np.zeros(len(ktraj), dtype=image.dtype)
    for j in range(np.prod(image.shape)):
        res += image[j] * np.exp(-2j * np.pi * ktraj @ grid_r[:, j])
    return res


def implicit_type1_ndft(ktraj, coeffs, shape):
    """Compute the NDFT using the implicit type 1 (kspace -> image) algorithm."""
    r = [np.arange(s) for s in shape]
    grid_r = np.reshape(np.meshgrid(*r, indexing="ij"), (len(shape), np.prod(shape)))
    res = np.zeros(np.prod(shape), dtype=coeffs.dtype)
    for i in range(len(ktraj)):
        res += coeffs[i] * np.exp(2j * np.pi * ktraj[i] @ grid_r)
    return res


def get_implicit_matrix(ktraj, shape):
    """Get the NDFT Fourier Matrix as implicit operator.

    This is more memory efficient than the explicit matrix.
    """
    return sp.sparse.linalg.LinearOperator(
        (len(ktraj), np.prod(shape)),
        matvec=lambda x: implicit_type2_ndft(ktraj, x, shape),
        rmatvec=lambda x: implicit_type1_ndft(ktraj, x, shape),
    )


class RawNDFT:
    """Implementation of the NUDFT using numpy."""

    def __init__(self, samples, shape, explicit_matrix=True):
        self.samples = samples
        self.shape = shape
        self.n_samples = len(samples)
        self.ndim = len(shape)
        if explicit_matrix:
            try:
                self._fourier_matrix = sp.sparse.linalg.aslinearoperator(
                    get_fourier_matrix(self.samples, self.shape)
                )
            except MemoryError:
                warnings.warn("Not enough memory, using an implicit definition anyway")
                self._fourier_matrix = get_implicit_matrix(self.samples, self.shape)
        else:
            self._fourier_matrix = get_implicit_matrix(self.samples, self.shape)

    def op(self, coeffs, image):
        """Compute the forward NUDFT."""
        np.copyto(coeffs, self._fourier_matrix @ image.flatten())
        return coeffs

    def adj_op(self, coeffs, image):
        """Compute the adjoint NUDFT."""
        np.copyto(
            image,
            (self._fourier_matrix.adjoint() @ coeffs.flatten()).reshape(self.shape),
        )
        return image


class MRInumpy(FourierOperatorCPU):
    """MRI operator using numpy NUDFT backend.

    For testing purposes only, as it is very slow.
    """

    backend = "numpy"
    available = True

    def __init__(self, samples, shape, n_coils=1, smaps=None):
        super().__init__(
            samples,
            shape,
            density=False,
            n_coils=n_coils,
            smaps=smaps,
            raw_op=RawNDFT(samples, shape),
        )
