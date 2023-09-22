"""Finufft interface."""

import numpy as np
import warnings

from ..base import FourierOperatorBase, proper_trajectory

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
        Sensitivity maps of shape ``N_coils x *shape``.
    squeeze_dims: bool
        If True, the dimensions of size 1 for the coil
        and batch dimension will be squeezed.
    """

    backend = "finufft"
    available = FINUFFT_AVAILABLE

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
                f"Data should be of dtype {self.cpx_dtype} (is {data.dtype}). "
                "Casting it for you."
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

    def _op_sense(self, data, ksp=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        dataf = data.reshape((B, *XYZ))
        if ksp is None:
            ksp = np.empty((B * C, K), dtype=self.cpx_dtype)
        for i in range(B * C // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            coil_img = self.smaps[idx_coils].copy().reshape((T, *XYZ))
            coil_img *= dataf[idx_batch]
            self._op(coil_img, ksp[i * T : (i + 1) * T])
        ksp = ksp.reshape((B, C, K))
        return ksp

    def _op_calibless(self, data, ksp=None):
        if ksp is None:
            ksp = np.empty(
                (self.n_batchs * self.n_coils, self.n_samples), dtype=self.cpx_dtype
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
        if self.n_trans == 1:
            image = image.reshape(self.shape)
            coeffs = coeffs.reshape(self.n_samples)
        self.raw_op.op(coeffs, image)

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
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        if img is None:
            img = np.zeros((B, *XYZ), dtype=self.cpx_dtype)
        coeffs_flat = coeffs.reshape((B * C, K))
        img_batched = np.zeros((T, *XYZ), dtype=self.cpx_dtype)
        for i in range(B * C // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            self._adj_op(coeffs_flat[i * T : (i + 1) * T], img_batched)
            img_batched *= self.smaps[idx_coils].conj()
            for t, b in enumerate(idx_batch):
                img[b] += img_batched[t]
        img = img.reshape((B, 1, *XYZ))
        return img

    def _adj_op_calibless(self, coeffs, img=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        if img is None:
            img = np.empty((B * C, *XYZ), dtype=self.cpx_dtype)
        coeffs_f = np.reshape(coeffs, (B * C, K))
        for i in range((B * C) // T):
            self._adj_op(coeffs_f[i * T : (i + 1) * T], img[i * T : (i + 1) * T])

        img = img.reshape((B, C, *XYZ))
        return img

    def _adj_op(self, coeffs, image):
        if self.density is not None:
            coeffs2 = coeffs.copy()
            for i in range(self.n_trans):
                coeffs2[i * self.n_samples : (i + 1) * self.n_samples] *= self.density
        else:
            coeffs2 = coeffs
        if self.n_trans == 1:
            image = image.reshape(self.shape)
            coeffs2 = coeffs2.reshape(self.n_samples)
        self.raw_op.adj_op(coeffs2, image)

    @property
    def norm_factor(self):
        """Norm factor of the operator."""
        return np.sqrt(np.prod(self.shape) * (2 ** len(self.shape)))

    def data_consistency(self, image_data, obs_data):
        """Compute the gradient estimation directly on gpu.

        This mixes the op and adj_op method to perform F_adj(F(x-y))
        on a per coil basis. By doing the computation coil wise,
        it uses less memory than the naive call to adj_op(op(x)-y)

        Parameters
        ----------
        image: array
            Image on which the gradient operation will be evaluated.
            N_coil x Image shape is not using sense.
        obs_data: array
            Observed data.
        """
        if self.uses_sense:
            return self._data_consistency_sense(image_data, obs_data)
        return self._data_consistency_calibless(image_data, obs_data)

    def _data_consistency_sense(self, image_data, obs_data):
        img = np.empty_like(image_data)
        coil_img = np.empty(self.shape, dtype=image_data.dtype)
        coil_ksp = np.empty(self.n_samples, dtype=obs_data.dtype)
        for i in range(self.n_coils):
            np.copyto(coil_img, img)
            coil_img *= self._smap
            self._op(coil_img, coil_ksp)
            coil_ksp /= self.norm_factor
            coil_ksp -= obs_data[i]
            if self.uses_density:
                coil_ksp *= self.density_d
            self._adj_op(coil_ksp, coil_img)
            coil_img /= self.norm_factor
            img += coil_img * self._smaps[i].conjugate()
        return img

    def _data_consistency_calibless(self, image_data, obs_data):
        img = np.empty((self.n_coils, *self.shape), dtype=image_data.dtype)
        ksp = np.empty(self.n_samples, dtype=obs_data.dtype)
        for i in range(self.n_coils):
            self._op(image_data[i], ksp)
            ksp /= self.norm_factor
            ksp -= obs_data[i]
            if self.uses_density:
                ksp *= self.density_d
            self._adj_op(ksp, img[i])
        return img / self.norm_factor
