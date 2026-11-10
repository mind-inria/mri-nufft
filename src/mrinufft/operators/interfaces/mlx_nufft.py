"""Interface for the MLX NUFFT operator."""

import numpy as np

from mrinufft.operators.interfaces.finufft import MRIfinufft

MLX_NUFFT_AVAILABLE = True
try:
    import mlx_nufft
except ImportError:
    MLX_NUFFT_AVAILABLE = False


class RawMlxNufftPlan:
    """Wrapper around the FINUFFT-compatible MLX plan interface."""

    def __init__(self, samples, shape, n_trans=1, eps=1e-6, **kwargs):
        if not MLX_NUFFT_AVAILABLE:
            raise RuntimeError("mlx-nufft is not installed.")
        self.shape = shape
        self.ndim = len(shape)
        self.n_trans = n_trans
        self.plan = mlx_nufft.Plan(
            2,
            shape,
            n_trans=n_trans,
            eps=eps,
            dtype=f"complex{samples.dtype.itemsize * 16}",
            **kwargs,
        )
        self._set_pts(samples)

    def _set_pts(self, samples, typ=2):
        if typ == 2:
            self.plan.setpts(*(np.asarray(samples[:, i]) for i in range(self.ndim)))

    def adj_op(self, coeffs_data, grid_data):
        """Type 1 transform. Non Uniform to Uniform."""
        grid_data[...] = self.plan.execute_adjoint(coeffs_data)

    def op(self, coeffs_data, grid_data):
        """Type 2 transform. Uniform to non-uniform."""
        coeffs_data[...] = self.plan.execute(grid_data)

    def toggle_grad_traj(self):
        """Trajectory gradients are not available for mlx-nufft."""
        raise NotImplementedError("mlx-nufft does not support trajectory gradients.")


class MRIMlxNUFFT(MRIfinufft):
    """MRI transform operator using mlx-nufft on Apple GPUs."""

    backend = "mlx-nufft"
    available = MLX_NUFFT_AVAILABLE
    autograd_available = False
    raw_plan_class = RawMlxNufftPlan
