"""Finufft interface."""

import numpy as np

from mrinufft._utils import proper_trajectory
from mrinufft.operators.base import FourierOperatorCPU, FourierOperatorBase

FINUFFT_AVAILABLE = True
try:
    from finufft._interfaces import Plan
except ImportError:
    FINUFFT_AVAILABLE = False

DTYPE_R2C = {"float32": "complex64", "float64": "complex128"}


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
        self.ndim = len(shape)
        self.eps = float(eps)
        self.n_trans = n_trans
        self.n_samples = len(samples)
        # the first element is dummy to index type 1 with 1
        # and type 2 with 2.
        self.plans = [None, None, None]
        self.grad_plan = None

        for i in [1, 2]:
            self._make_plan(i, samples, **kwargs)
            self._set_pts(i, samples)

    def _make_plan(self, typ, samples, **kwargs):
        self.plans[typ] = Plan(
            typ,
            self.shape,
            self.n_trans,
            self.eps,
            dtype=DTYPE_R2C[str(samples.dtype)],
            **kwargs,
        )

    def _set_pts(self, typ, samples):
        fpts_axes = [None, None, None]
        for i in range(self.ndim):
            fpts_axes[i] = np.array(samples[:, i], dtype=samples.dtype)
        plan = self.grad_plan if typ == "grad" else self.plans[typ]
        plan.setpts(*fpts_axes)

    def adj_op(self, coeffs_data, grid_data):
        """Type 1 transform. Non Uniform to Uniform."""
        if self.n_trans == 1:
            grid_data = grid_data.reshape(self.shape)
            coeffs_data = coeffs_data.reshape(self.n_samples)
        return self.plans[1].execute(coeffs_data, grid_data)

    def op(self, coeffs_data, grid_data):
        """Type 2 transform. Uniform to non-uniform."""
        if self.n_trans == 1:
            grid_data = grid_data.reshape(self.shape)
            coeffs_data = coeffs_data.reshape(self.n_samples)
        return self.plans[2].execute(grid_data, coeffs_data)

    def toggle_grad_traj(self):
        """Toggle between the gradient trajectory and the plan for type 1 transform."""
        self.plans[2], self.grad_plan = self.grad_plan, self.plans[2]


class MRIfinufft(FourierOperatorCPU):
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
    n_batchs: int
        Number of batchs .
    n_trans: int
        Number of parallel transform
    density: bool or array
       Density compensation support.
        - If a Tensor, it will be used for the density.
        - If True, the density compensation will be automatically estimated,
          using the fixed point method.
        - If False, density compensation will not be used.
    smaps: array
        Sensitivity maps of shape ``N_coils x *shape``.
    squeeze_dims: bool
        If True, the dimensions of size 1 for the coil
        and batch dimension will be squeezed.
    """

    backend = "finufft"
    available = FINUFFT_AVAILABLE
    autograd_available = True

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        n_batchs=1,
        n_trans=1,
        smaps=None,
        squeeze_dims=True,
        **kwargs,
    ):
        samples = proper_trajectory(np.asfortranarray(samples), normalize="pi")
        self.raw_op = RawFinufftPlan(
            samples,
            shape,
            n_trans=n_trans,
            **kwargs,
        )
        super().__init__(
            samples,
            shape,
            density,
            n_coils=n_coils,
            n_batchs=n_batchs,
            n_trans=n_trans,
            smaps=smaps,
            raw_op=self.raw_op,
            squeeze_dims=squeeze_dims,
        )

    @FourierOperatorBase.samples.setter
    def samples(self, new_samples):
        """Update the plans when changing the samples."""
        self._samples = proper_trajectory(
            np.asfortranarray(new_samples), normalize="pi"
        )
        for typ in [1, 2, "grad"]:
            if typ == "grad" and not self._grad_wrt_traj:
                continue
            self.raw_op._set_pts(typ, self._samples)
        self.compute_density(self._density_method)

    def _make_plan_grad(self, **kwargs):
        self.raw_op.grad_plan = Plan(
            2,
            self.raw_op.shape,
            self.raw_op.n_trans,
            self.raw_op.eps,
            dtype=DTYPE_R2C[str(self.samples.dtype)],
            isign=1,
            **kwargs,
        )
        self.raw_op._set_pts(typ="grad", samples=self.samples)

    def toggle_grad_traj(self):
        """Toggle between the gradient trajectory and the plan for type 1 transform."""
        if self.uses_sense:
            self.smaps = self.smaps.conj()
        self.raw_op.toggle_grad_traj()
