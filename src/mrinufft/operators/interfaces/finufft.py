"""Finufft interface."""

import numpy as np
import warnings

from .base import FourierOperatorBase, proper_trajectory

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


class MRIfinufft(FourierOperatorBase):
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
    """

    backend = "finufft"

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        n_batchs=1,
        n_trans=1,
        smaps=None,
        squeeze_dims=False,
    ):
        super().__init__()

        if not FINUFFT_AVAILABLE:
            raise RuntimeError("finufft is not available.")
        self.shape = shape

        # we will access the samples by their coordinate first.
        self.samples = proper_trajectory(np.asfortranarray(samples), normalize="pi")
        self.dtype = self.samples.dtype

        # Density Compensation Setup
        if density is True:
            self.density = self.estimate_density(self.samples, shape)
        elif isinstance(density, np.ndarray):
            if len(density) != len(self.samples):
                raise ValueError(
                    "Density array and samples array should have the same length."
                )
            self.density = np.asfortranarray(density)
        else:
            self.density = None
        # Multi Coil Setup
        if n_coils < 1:
            raise ValueError("n_coils should be ≥ 1")
        self.n_coils = n_coils
        self.smaps = smaps
        self.n_batchs = n_batchs
        self.n_trans = n_trans
        self.squeeze_dims = squeeze_dims
        # Initialise NUFFT plans
        self.raw_op = RawFinufftPlan(
            self.samples,
            tuple(shape),
            n_trans=n_trans,
        )

    def op(self, data, ksp=None):
        r"""Non Cartesian MRI forward operator.

        Parameters
        ----------
        data: np.ndarray
        The uniform (2D or 3D) data in image space.

        Returns
        -------
        Results array on the same device as data.

        Notes
        -----
        this performs for every coil \ell:
        ..math:: \mathcal{F}\mathcal{S}_\ell x
        """
        if data.dtype != self.cpx_dtype:
            warnings.warn(
                f"Data should be of dtype {self.cpx_dtype}. Casting it for you."
            )
            data = data.astype(self.cpx_dtype)
        # sense
        if self.uses_sense:
            ret = self._op_sense(data, ksp)
        # calibrationless or monocoil.
        else:
            ret = self._op_calibless(data, ksp)
        ret /= self.norm_factor
        if not self.squeeze_dims:
            return ret
        else:  # squeeze the batch and coil dimensions.
            return ret.squeeze(axis=(0, 1))

    def _op_sense(self, data, ksp_d=None):
        if self.n_batchs > 1:
            raise ValueError("Sense cannot be used with batchs.")
        coil_img = np.empty((self.n_coils, *self.shape), dtype=data.dtype)
        ksp = np.zeros((self.n_coils, self.n_samples), dtype=data.dtype)
        coil_img = data * self.smaps
        self._op(coil_img)
        return ksp

    def _op_calibless(self, data, ksp=None):
        ksp = ksp or np.empty(
            (self.n_batchs * self.n_coils, self.n_samples), dtype=data.dtype
        )
        dataf = np.reshape(data, (self.n_batchs * self.n_coils, *self.shape))
        if self.n_trans == 1:
            for i in range(self.n_coils * self.n_batchs):
                self._op(dataf[i], ksp[i])
        else:
            for i in range((self.n_coils * self.n_batchs) // self.n_trans):
                self._op(
                    dataf[i * self.n_trans : (i + 1) * self.n_trans],
                    ksp[i * self.n_trans : (i + 1) * self.n_trans],
                )
        ksp = ksp.reshape((self.n_batchs, self.n_coils, self.n_samples))
        return ksp

    def _op(self, image, coeffs):
        self.raw_op.op(coeffs, np.asfortranarray(image))

    def adj_op(self, coeffs, img=None):
        """Non Cartesian MRI adjoint operator.

        Parameters
        ----------
        coeffs: np.array or GPUArray

        Returns
        -------
        Array in the same memory space of coeffs. (ie on cpu or gpu Memory).
        """
        if coeffs.dtype != self.cpx_dtype:
            warnings.warn(
                f"coeffs should be of dtype {self.cpx_dtype}. Casting it for you."
            )
            coeffs = coeffs.astype(self.cpx_dtype)
        if self.uses_sense:
            ret = self._adj_op_sense(coeffs, img)
        # calibrationless or monocoil.
        else:
            ret = self._adj_op_calibless(coeffs, img)
        ret /= self.norm_factor
        if not self.squeeze_dims:
            return ret
        else:
            return ret.squeeze(axis=(0, 1))

    def _adj_op_sense(self, coeffs, img=None):
        if self.n_batchs > 1:
            raise ValueError("Sense cannot be used with batchs.")
        coil_img = np.empty(self.shape, dtype=coeffs.dtype)
        if img is None:
            img = np.zeros(self.shape, dtype=coeffs.dtype)
        self._adj_op(coeffs, coil_img)
        img = np.sum(coil_img * self.smaps.conjugate(), axis=0)
        return img

    def _adj_op_calibless(self, coeffs, img=None):
        img = img or np.empty(
            (self.n_batchs * self.n_coils, *self.shape), dtype=coeffs.dtype
        )
        coeffs_f = np.reshape(coeffs, (self.n_batchs * self.n_coils, self.n_samples))
        for i in range((self.n_batchs * self.n_coils) // self.n_trans):
            self._adj_op(coeffs_f[i], img[i])

        img = img.reshape((self.n_batchs, self.n_coils, *self.shape))
        return img

    def _adj_op(self, coeffs, image):
        if self.density is not None:
            coeffs2 = coeffs.copy()
            for i in range(self.n_trans):
                coeffs2[i * self.n_samples : (i + 1) * self.n_samples] *= self.density
        else:
            coeffs2 = coeffs
        self.raw_op.adj_op(coeffs2, image)

    @property
    def norm_factor(self):
        """Norm factor of the operator."""
        return np.sqrt(np.prod(self.shape) * (2 ** len(self.shape)))
