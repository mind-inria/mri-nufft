"""An implementation of the NUDFT using numpy."""

import numpy as np

from ..base import FourierOperatorCPU

PYNFFT_AVAILABLE = True
try:
    import pyNFFT3 as pynfft3
except ImportError:
    PYNFFT_AVAILABLE = False


class RawPyNFFT3:
    """Binding for the pyNFFT3 package."""

    def __init__(self, samples, shape):
        self.samples = samples
        self.shape = shape
        self.plan = pynfft3.NFFT(N=np.array(shape, dtype="int32"), M=len(samples))
        self.plan.x = self.samples

    def op(self, coeffs, image):
        """Compute the forward NUDFT."""
        self.plan.fhat = image.ravel()
        self.plan.trafo()
        np.copyto(coeffs, self.plan.f.reshape(-1))
        return coeffs

    def adj_op(self, coeffs, image):
        """Compute the adjoint NUDFT."""
        self.plan.f = coeffs.ravel()
        self.plan.adjoint()
        np.copyto(image, self.plan.fhat.reshape(self.shape))
        return image


class MRInfft(FourierOperatorCPU):
    """MRI operator using numpy NUDFT backend.

    For testing purposes only, as it is very slow.
    """

    backend = "pynfft"
    available = PYNFFT_AVAILABLE

    def __init__(
        self, samples, shape, n_coils=1, n_batchs=1, smaps=None, density=False
    ):
        super().__init__(
            samples,
            shape,
            n_coils=n_coils,
            n_batchs=n_batchs,
            n_trans=1,
            smaps=smaps,
            density=density,
            raw_op=None,  # is set later, after normalizing samples.
        )
        self.raw_op = RawPyNFFT3(self.samples, shape)
