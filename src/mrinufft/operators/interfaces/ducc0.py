"""An implementation of the NUDFT using numpy."""

import numpy as np

from ..base import FourierOperatorCPU

DUCC0_AVAILABLE = True
try:
    import ducc0
except ImportError:
    DUCC0_AVAILABLE = False


class RawDUCC0:
    """Binding for the ducc0 package.

    Parameters
    ----------
    samples: array
        The samples location of shape ``Nsamples x N_dimensions``.
        It should be C-contiguous.
    shape: tuple
        Shape of the image space.
    eps: float, default=1e-6
        Requested accuracy of the transform.
        Useful values go from 1e-2 down to 1e-13 (for double precision).
    nthreads: int, default=1
        Number of threads to use for the transforms.
        0 uses as many threads as there are virtual CPU cores on the system.
    """

    def __init__(self, samples, shape, eps=1e-6, nthreads=1, **kwargs):
        self.samples = samples
        self.shape = shape
        self.plan = ducc0.nufft.plan(
            nu2u=False,  # can be used for both directions
            coord=samples,
            grid_shape=shape,
            epsilon=eps,
            nthreads=nthreads,
            periodicity=1.0,  # must be 1, otherwise conventions don't match
            fft_order=False,  # must be False, otherwise conventions don't match
            **kwargs,
        )

    def op(self, coeffs, image):
        """Compute the forward NUDFT."""
        return self.plan.u2nu(forward=True, grid=image, out=coeffs)

    def adj_op(self, coeffs, image):
        """Compute the adjoint NUDFT."""
        return self.plan.nu2u(forward=False, points=coeffs, out=image)


class MRIDUCC0(FourierOperatorCPU):
    """MRI operator using ducc0 backend."""

    backend = "ducc0"
    available = DUCC0_AVAILABLE

    def __init__(
        self, samples, shape, n_coils=1, n_batchs=1, smaps=None, density=False, **kwargs
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
        self.raw_op = RawDUCC0(self.samples, shape, **kwargs)
