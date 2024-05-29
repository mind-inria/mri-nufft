"""An implementation of the NUDFT using numpy."""

import warnings

import numpy as np
import scipy as sp
import torch
from ..base import FourierOperatorCPU
from mrinufft._utils import proper_trajectory, get_array_module


def get_fourier_matrix(ktraj, shape, dtype=np.complex64, normalize=False):
    #FIXME: check if trajectory is torch. get_array_module : torch. Basically output must be torch if input is torch and everything must be done in torch. 
    """Get the NDFT Fourier Matrix."""
    
    module = get_array_module(ktraj)
    ktraj = proper_trajectory(ktraj, normalize="unit")
    n = np.prod(shape)
    ndim = len(shape)

    if module.__name__ == "torch":
        device = ktraj.device
        dtype = torch.complex64
        r = [torch.linspace(-s / 2, s / 2 - 1, s, device=device) for s in shape]
        grid_r = torch.meshgrid(r, indexing="ij")
        grid_r = torch.reshape(torch.stack(grid_r), (ndim, n)).to(device)
        traj_grid = torch.matmul(ktraj, grid_r)
        matrix = torch.exp(-2j * np.pi * traj_grid).to(dtype).to(device).clone()

    else:
        r = [np.linspace(-s / 2, s / 2 - 1, s) for s in shape]
        grid_r = np.reshape(np.meshgrid(*r, indexing="ij"), (ndim, np.prod(shape)))
        traj_grid = ktraj @ grid_r
        matrix = np.exp(-2j * np.pi * traj_grid, dtype=dtype)

    if normalize:
        matrix /= (
            (
                torch.sqrt(torch.tensor(np.prod(shape), device=device))
                * torch.pow(torch.sqrt(torch.tensor(2, device=device)), ndim)
            )
            if module.__name__ == "torch"
            else (np.sqrt(np.prod(shape)) * np.power(np.sqrt(2), len(shape)))
        )

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
