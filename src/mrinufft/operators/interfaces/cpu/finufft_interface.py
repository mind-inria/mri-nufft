"""Finufft interface."""

import numpy as np

from .base import AbstractMRIcpuNUFFT

FINUFFT_AVAILABLE = True
try:
    from finufft._interfaces import Plan
except ImportError:
    FINUFFT_AVAILABLE = False


class RawFinufftPlan:
    """Light wrapper around the guru interface of finufft."""

    def __init__(
        self,
        samples,
        shape,
        n_trans=1,
        eps=1e-6,
        **kwargs,
    ):
        self.shape = shape
        self.samples = samples
        self.ndim = len(shape)
        self.eps = float(eps)
        self.n_trans = n_trans

        # the first element is dummy to index type 1 with 1
        # and type 2 with 2.
        self.plans = [None, None, None]

        for i in [1, 2]:
            self._make_plan(i, **kwargs)
            self._set_pts(i)

    def _make_plan(self, typ, **kwargs):
        self.plans[typ] = Plan(
            typ,
            self.shape,
            self.n_trans,
            self.eps,
            dtype="complex64" if self.samples.dtype == "float32" else "complex128",
            **kwargs,
        )

    def _set_pts(self, typ):
        fpts_axes = [None, None, None]
        for i in range(self.ndim):
            fpts_axes[i] = np.array(self.samples[:, i], dtype=self.samples.dtype)
        self.plans[typ].setpts(*fpts_axes)

    def adj_op(self, coeff_data, grid_data):
        """Type 1 transform. Non Uniform to Uniform."""
        return self.plans[1].execute(coeff_data, grid_data)

    def op(self, coeff_data, grid_data):
        """Type 2 transform. Uniform to non-uniform."""
        return self.plans[2].execute(grid_data, coeff_data)


class MRIfinufft(AbstractMRIcpuNUFFT):
    """MRI Transform Operator using finufft.

    Parameters
    ----------
    samples: array
        The samples location of shape ``Nsamples x N_dimensions``.
        It should be C-contiguous.
    shape: tuple
        Shape of the image space.
    n_coils: int
        Number of coils.
    density: bool or array
       Density compensation support.
        - If a Tensor, it will be used for the density.
        - If True, the density compensation will be automatically estimated,
          using the fixed point method.
        - If False, density compensation will not be used.
    smaps: array
    """

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        smaps=None,
        **kwargs,
    ):
        if not FINUFFT_AVAILABLE:
            raise RuntimeError("finufft is not available.")
        super().__init__(samples, shape, density, n_coils, smaps)
        # Initialise NUFFT plans
        self.raw_op = RawFinufftPlan(
            self.samples,
            tuple(shape),
            n_trans=n_coils,
            **kwargs,
        )

    @classmethod
    def estimate_density(cls, samples, shape, n_iter=1, **kwargs):
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
