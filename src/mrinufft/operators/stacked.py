"""Stacked Operator for NUFFT.""""


import numpy as np
import scipy as sp

from .interfaces.base import FourierOperatorBase, proper_trajectory
from .interfaces import get_operator

class MRIStackedNUFFT(FourierOperatorBase):
    """Stacked NUFFT Operator for MRI.

    Parameters
    ----------
    samples : array-like
        Sample locations in k-space 2D
    z_index: array-like
        Z index of masked plan.
    backend: str
        Backend to use.
    smaps: array-like
        Sensitivity maps.
    n_coils: int
        Number of coils.
    n_batchs: int
        Number of batchs.
    **kwargs: dict
        Additional arguments to pass to the backend.


    TODO: Add  support for GPU Arrays.
    """

    def __init__(self, samples, z_index, backend, smaps, n_coils=1, n_batchs=1, **kwargs):
        self.samples = samples
        self.shape = shape

        if z_index is None:
            z_index = np.arange(samples.shape[-1])
        elif z_index.dtype == np.bool:
            z_index = np.where(z_index)

        self.n_coils = n_coils
        self.n_batchs = n_batchs

        self.z_index = z_index
        self.smaps = smaps
        self.operator = get_operator(backend)()

    def op(self, data):
        # Do SENSE Stuff if needed.
        # Apply the FFT on z-axis
        # Apply the NUFFT on the selected plans
        #

    def adj_op(self, data):
        # DO NUFFT adjoint
        # Apply the FFT on z-axis
        # Do SENSE Stuff if needed.

        pass
