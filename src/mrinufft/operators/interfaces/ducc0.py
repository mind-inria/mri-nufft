"""An implementation of the NUDFT using numpy."""

import numpy as np

from ..base import FourierOperatorCPU

DUCC0_AVAILABLE = True
try:
    import ducc0
except ImportError:
    DUCC0_AVAILABLE = False


class RawDUCC0:
    """Binding for the ducc0 package."""

    def __init__(self, samples, shape, eps=1e-6):
        self.samples = samples
        self.shape = shape
        self.plan = ducc0.nufft.plan(
            nu2u=False,  # can be used for both directions
            coord=samples,
            grid_shape=shape,
            epsilon=eps,
            # nthreads=?,
            periodicity=1,
            fft_order=False,
        )

    def op(self, coeffs, image):
        """Compute the forward NUDFT."""
        for i in range(coeffs.shape[0]):
            self.plan.u2nu(forward=True, grid=image[i], out=coeffs[i])
        return coeffs

    def adj_op(self, coeffs, image):
        """Compute the adjoint NUDFT."""
        for i in range(coeffs.shape[0]):
            self.plan.nu2u(forward=False, points=coeffs[i], out=image[i])
        return image


class MRIDUCC0(FourierOperatorCPU):
    """MRI operator using ducc0 backend."""

    backend = "ducc0"
    available = DUCC0_AVAILABLE

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
        self.raw_op = RawDUCC0(self.samples, shape)
