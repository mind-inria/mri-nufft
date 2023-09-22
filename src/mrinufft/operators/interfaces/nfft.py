"""An implementation of the NUDFT using numpy."""

import numpy as np

from ..base import FourierOperatorCPU

PYNFFT_AVAILABLE = True
try:
    import pynfft
except ImportError:
    PYNUFFT_AVAILABLE = False


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


class RawPyNFFT:
    """Implementation of the NUDFT using numpy."""

    def __init__(self, samples, shape):
        self.samples = samples
        self.shape = shape
        self.ndim = len(shape)
        self.plan = pynfft.NFFT(N=shape, M=len(samples))
        self.plan.x = self.samples
        self.plan.precompute()
        self.shape = shape

    def op(self, coeffs, image):
        """Compute the forward NUDFT."""
        self.plan.f_hat = image
        np.copyto(coeffs, self.plan.trafo())
        return coeffs

    def adj_op(self, coeffs, image):
        """Compute the adjoint NUDFT."""
        self.plan.f = coeffs
        np.copyto(image, self.plan.adjoint())
        return image


class MRInfft(FourierOperatorCPU):
    """MRI operator using numpy NUDFT backend.

    For testing purposes only, as it is very slow.
    """

    backend = "pynfft"
    available = PYNFFT_AVAILABLE

    def __init__(self, samples, shape, n_coils=1, smaps=None):
        super().__init__(
            samples,
            shape,
            n_coils=n_coils,
            smaps=smaps,
            raw_op=None,  # is set later, after normalizing samples.
        )
        self.raw_op = RawPyNFFT(self.samples, shape)
