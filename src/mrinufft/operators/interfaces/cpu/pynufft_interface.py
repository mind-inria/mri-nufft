"""PyNUFFT CPU Interface."""

import numpy as np

from .base import AbstractMRIcpuNUFFT


PYNUFFT_CPU_AVAILABLE = True
try:
    import pynufft
except ImportError:
    PYNUFFT_CPU_AVAILABLE = False


class RawPyNUFFT:
    """Wrapper around PyNUFFT object."""

    def __init__(self, samples, shape, osf=2, interpolator_shape=6):
        self._nufft_obj = pynufft.NUFFT()
        self._nufft_obj.plan(
            samples,
            shape,
            tuple(osf * s for s in shape),
            tuple([interpolator_shape] * len(shape)),
        )

    def op(self, coeffs_data, grid_data):
        """Forward Operator."""
        coeffs_data = self._nufft_obj.forward(grid_data)
        return coeffs_data

    def adj_op(self, coeffs_data, grid_data):
        """Adjoint Operator."""
        grid_data = self._nufft_obj.backward(coeffs_data)
        return grid_data


class MRIPynufft(AbstractMRIcpuNUFFT):
    """PyNUFFT implementation of MRI NUFFT transform."""

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        smaps=None,
        osf=2,
        **kwargs,
    ):
        if not PYNUFFT_CPU_AVAILABLE:
            raise RuntimeError("Pynufft is not available.")
        super().__init__(samples, shape, density, n_coils, smaps)

        self.raw_op = RawPyNUFFT(samples, shape, osf, **kwargs)

    @classmethod
    def estimate_density(cls, samples, shape, n_iter=10, **kwargs):
        """Estimate the density compensation array."""
        oper = cls(samples, shape, density=False, **kwargs)
        density = np.ones(len(samples), dtype=oper._cpx_dtype)
        update = np.empty_like(density, dtype=oper._cpx_dtype)
        img = np.empty(shape, dtype=oper._cpx_dtype)
        for _ in range(n_iter):
            oper._adj_op(density, img)
            oper._op(img, update)
            density /= np.abs(update)
        return density.real
