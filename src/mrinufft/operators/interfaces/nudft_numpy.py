"""An implementation of the NUDFT using numpy."""

import warnings

import numpy as np
import scipy as sp
from ..base import FourierOperatorCPU
from mrinufft._utils import proper_trajectory, get_array_module


def get_fourier_matrix(ktraj, shape, dtype=np.complex64, normalize=False):
    """Get the NDFT Fourier Matrix.

    Parameters
    ----------
    ktraj: array_like
        The k-space coordinates for the Fourier transformation.
    shape: tuple of int
        The dimensions of the output Fourier matrix.
    dtype: data-type, optional
        The data type of the Fourier matrix, default is np.complex64.
    normalize : bool, optional
        If True, normalizes the matrix to maintain numerical stability.

    Returns
    -------
    matrix
        The NDFT Fourier Matrix.
    """
    xp = get_array_module(ktraj)
    ktraj = proper_trajectory(ktraj, normalize="unit")
    n = np.prod(shape)
    ndim = len(shape)
    dtype = xp.complex64
    device = getattr(ktraj, "device", None)

    r = [xp.linspace(-s / 2, s / 2 - 1, s) for s in shape]
    if xp.__name__ == "torch":
        r = [x.to(device) for x in r]

    grid_r = xp.meshgrid(*r, indexing="ij")
    grid_r = xp.reshape(xp.stack(grid_r), (ndim, n))
    traj_grid = xp.matmul(ktraj, grid_r)
    matrix = xp.exp(-2j * xp.pi * traj_grid)
    if xp.__name__ == "torch":
        matrix = matrix.to(dtype=dtype, device=device, copy=True)

    if normalize:
        norm_factor = np.sqrt(np.prod(shape)) * np.power(np.sqrt(2), ndim)
        if xp.__name__ == "torch":
            norm_factor = xp.tensor(norm_factor, device=device)
        matrix /= norm_factor

    return matrix


def implicit_type2_ndft(ktraj, image, shape, normalize=False):
    """Compute the NDFT using the implicit type 2 (image -> kspace) algorithm."""
    r = [np.linspace(-s / 2, s / 2 - 1, s) for s in shape]
    grid_r = np.reshape(
        np.meshgrid(*r, indexing="ij"), (len(shape), np.prod(image.shape))
    )
    res = np.zeros(len(ktraj), dtype=image.dtype)
    for j in range(np.prod(image.shape)):
        res += image[j] * np.exp(-2j * np.pi * ktraj @ grid_r[:, j])
    if normalize:
        res /= np.sqrt(np.prod(shape)) * np.power(np.sqrt(2), len(shape))
    return res


def implicit_type1_ndft(ktraj, coeffs, shape, normalize=False):
    """Compute the NDFT using the implicit type 1 (kspace -> image) algorithm."""
    r = [np.linspace(-s / 2, s / 2 - 1, s) for s in shape]
    grid_r = np.reshape(np.meshgrid(*r, indexing="ij"), (len(shape), np.prod(shape)))
    res = np.zeros(np.prod(shape), dtype=coeffs.dtype)
    for i in range(len(ktraj)):
        res += coeffs[i] * np.exp(2j * np.pi * ktraj[i] @ grid_r)
    if normalize:
        res /= np.sqrt(np.prod(shape)) * np.power(np.sqrt(2), len(shape))
    return res


def get_implicit_matrix(ktraj, shape, normalize=False):
    """Get the NDFT Fourier Matrix as implicit operator.

    This is more memory efficient than the explicit matrix.
    """
    return sp.sparse.linalg.LinearOperator(
        (len(ktraj), np.prod(shape)),
        matvec=lambda x: implicit_type2_ndft(ktraj, x, shape, normalize),
        rmatvec=lambda x: implicit_type1_ndft(ktraj, x, shape, normalize),
    )


class RawNDFT:
    """Implementation of the NUDFT using numpy."""

    def __init__(self, samples, shape, explicit_matrix=True, normalize=False):
        self.samples = samples
        self.shape = shape
        self.n_samples = len(samples)
        self.ndim = len(shape)
        if explicit_matrix:
            try:
                self._fourier_matrix = sp.sparse.linalg.aslinearoperator(
                    get_fourier_matrix(self.samples, self.shape, normalize=normalize)
                )
            except MemoryError:
                warnings.warn("Not enough memory, using an implicit definition anyway")
                self._fourier_matrix = get_implicit_matrix(
                    self.samples, self.shape, normalize
                )
        else:
            self._fourier_matrix = get_implicit_matrix(
                self.samples, self.shape, normalize
            )

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
