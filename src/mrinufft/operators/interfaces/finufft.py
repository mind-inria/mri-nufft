"""Finufft interface."""

import numpy as np

from mrinufft._utils import proper_trajectory
from mrinufft.operators.base import (
    FourierOperatorCPU,
    FourierOperatorBase,
    _ToggleGradPlanMixin,
)
from mrinufft._array_compat import _array_to_numpy

FINUFFT_AVAILABLE = True
try:
    import finufft
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

        if finufft.__version__ >= "2.6.0":
            kwargs["allow_eps_too_small"] = 1

        self.plan = Plan(
            2,
            self.shape,
            self.n_trans,
            self.eps,
            dtype=DTYPE_R2C[str(samples.dtype)],
            **kwargs,
        )
        self.grad_plan: Plan

        self._set_pts(samples)

    def _set_pts(self, samples, typ=2):
        fpts_axes = [None, None, None]
        for i in range(self.ndim):
            fpts_axes[i] = np.array(samples[:, i], dtype=samples.dtype)
        if typ == 2:
            self.plan.setpts(*fpts_axes)
        elif typ == "grad":
            self.grad_plan.setpts(*fpts_axes)

    def adj_op(self, coeffs_data, grid_data):
        """Type 1 transform. Non Uniform to Uniform."""
        if self.n_trans == 1:
            grid_data = grid_data.reshape(self.shape)
            coeffs_data = coeffs_data.reshape(self.n_samples)
        return self.plan.execute_adjoint(coeffs_data, grid_data)

    def op(self, coeffs_data, grid_data):
        """Type 2 transform. Uniform to non-uniform."""
        if self.n_trans == 1:
            grid_data = grid_data.reshape(self.shape)
            coeffs_data = coeffs_data.reshape(self.n_samples)
        return self.plan.execute(grid_data, coeffs_data)

    def toggle_grad_traj(self):
        """Toggle between the gradient trajectory and the plan for type 1 transform."""
        self.plan, self.grad_plan = self.grad_plan, self.plan


class MRIfinufft(FourierOperatorCPU, _ToggleGradPlanMixin):
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
        samples = _array_to_numpy(proper_trajectory(samples, normalize="pi"))
        samples = np.asarray(samples, order="F")
        raw_op = RawFinufftPlan(
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
            raw_op=raw_op,
            squeeze_dims=squeeze_dims,
        )

        self.raw_op: RawFinufftPlan

    def update_samples(self, new_samples, unsafe=False):
        """Update the samples of the NUFFT operator.

        Parameters
        ----------
        new_samples: np.ndarray or GPUArray
            The new samples location of shape ``Nsamples x N_dimensions``.
        unsafe: bool, default False
            If True, the original array is used directly without any checks.
            This should be used with caution as it might lead to unexpected behavior.

        Notes
        -----
        If unsafe is True, the new_samples should be of shape (Nsamples, N_dimensions),
        F-ordered (column-major) and in the range [-pi, pi]. If not, this will lead to
        unexpected behavior. You have been warned.

        If unsafe is False, this is automatically handled.
        """
        if not unsafe:
            self._samples = np.asarray(
                proper_trajectory(new_samples, normalize="pi"), order="F"
            )

        else:
            self._samples = new_samples

        for typ in [2, "grad"]:
            if typ == "grad" and not self._grad_wrt_traj:
                continue
            self.raw_op._set_pts(self._samples, typ=typ)
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

    @classmethod
    def pipe(
        cls,
        kspace_loc,
        volume_shape,
        max_iter=10,
        osf=2,
        normalize=True,
        **kwargs,
    ):
        """Compute the density compensation weights for a given set of kspace locations.

        Parameters
        ----------
        kspace_loc: np.ndarray
            the kspace locations
        volume_shape: np.ndarray
            the volume shape
        max_iter: int default 10
            the number of iterations for density estimation
        osf: float or int
            The oversampling factor the volume shape
        normalize: bool
            Whether to normalize the density compensation.
            We normalize such that the energy of PSF = 1
        """
        if FINUFFT_AVAILABLE is False:
            raise ValueError(
                "finufft is not available, cannot estimate the density compensation"
            )
        grid_op = MRIfinufft(
            samples=kspace_loc,
            shape=volume_shape,
            upsampfac=osf,
            spreadinterponly=1,
            spread_kerevalmeth=0,
            **kwargs,
        )
        density_comp = np.ones(kspace_loc.shape[0], dtype=grid_op.cpx_dtype)
        for _ in range(max_iter):
            density_comp /= abs(
                grid_op.op(
                    grid_op.adj_op(density_comp.astype(grid_op.cpx_dtype))
                ).squeeze()
            )
        if normalize:
            test_op = MRIfinufft(samples=kspace_loc, shape=volume_shape, **kwargs)
            test_im = np.ones(volume_shape, dtype=test_op.cpx_dtype)
            test_im_recon = test_op.adj_op(density_comp * test_op.op(test_im))
            density_comp /= np.mean(np.abs(test_im_recon))
        return abs(density_comp.squeeze())
