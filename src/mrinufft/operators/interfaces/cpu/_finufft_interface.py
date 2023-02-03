"""Low level interface for finufft."""
import numpy as np

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

    def type1(self, coeff_data, grid_data):
        """Type 1 transform. Non Uniform to Uniform."""
        return self.plans[1].execute(coeff_data, grid_data)

    def type2(self, coeff_data, grid_data):
        """Type 2 transform. Uniform to non-uniform."""
        return self.plans[2].execute(grid_data, coeff_data)
