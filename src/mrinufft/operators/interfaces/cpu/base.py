"""Base interface for the CPU backends."""
import warnings

import numpy as np

from ..base import FourierOperatorBase

from .cpu_kernels import sense_adj_mono


class AbstractMRIcpuNUFFT(FourierOperatorBase):
    """Base class for CPU-based NUFFT operator."""

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        smaps=None,
    ):
        self.shape = shape
        self.n_samples = len(samples)
        if samples.max() > np.pi:
            warnings.warn("samples will be normalized in [-pi, pi]")
            samples *= np.pi / samples.max()
        # we will access the samples by their coordinate first.
        self.samples = np.asfortranarray(samples)

        self._dtype = self.samples.dtype
        self._cpx_dtype = np.complex128 if self._dtype == "float64" else np.complex64
        self._uses_sense = False

        # Density Compensation Setup
        if density is True:
            self.density = self.estimate_density(samples, shape)
        elif isinstance(density, np.ndarray):
            if len(density) != len(samples):
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
        if smaps is not None:
            self._uses_sense = True
            if isinstance(smaps, np.ndarray):
                raise ValueError("Smaps should be either a C-ordered ndarray")
            self._smaps = smaps
        else:
            self._uses_sense = False

        # Raw_op should be instantiated by subclasses.
        self.raw_op = None

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
        if data.dtype != self._cpx_dtype:
            warnings.warn(
                f"Data should be of dtype {self._cpx_dtype}. Casting it for you."
            )
            data = data.astype(self._cpx_dtype)
        # monocoil
        if self.n_coils == 1:
            ret = self._op_mono(data, ksp)
        # sense
        elif self.uses_sense:
            ret = self._op_sense(data, ksp)
        # calibrationless, data on device
        else:
            ret = self._op_calibless(data, ksp)
        return ret

    def _op_mono(self, data, ksp=None):
        if ksp is None:
            ksp = np.empty(self.n_samples, dtype=data.dtype)
        self._op(data, ksp)
        return ksp

    def _op_sense(self, data, ksp_d=None):
        coil_img = np.empty((self.n_coils, *self.shape), dtype=data.dtype)
        ksp = np.zeros((self.n_coils, self.n_samples), dtype=data.dtype)
        coil_img = data * self._smaps
        self._op(coil_img)
        return ksp

    def _op_calibless(self, data, ksp=None):
        if ksp is None:
            ksp = np.empty((self.n_coils, self.n_samples), dtype=data.dtype)
        for i in range(self.n_coils):
            self._op(data[i], ksp[i])
        return ksp

    def _op(self, image, coeffs):
        return self.raw_op.op(coeffs, image)

    def adj_op(self, coeffs, img=None):
        """Non Cartesian MRI adjoint operator.

        Parameters
        ----------
        coeffs: np.array or GPUArray

        Returns
        -------
        Array in the same memory space of coeffs. (ie on cpu or gpu Memory).
        """
        if coeffs.dtype != self._cpx_dtype:
            warnings.warn(
                f"coeffs should be of dtype {self._cpx_dtype}. Casting it for you."
            )
            coeffs = coeffs.astype(self._cpx_dtype)
        if self.n_coils == 1:
            ret = self._adj_op_mono(coeffs, img)
        # sense
        elif self.uses_sense:
            ret = self._adj_op_sense(coeffs, img)
        # calibrationless
        else:
            ret = self._adj_op_calibless(coeffs, img)
        return ret

    def _adj_op_mono(self, coeffs, img=None):
        if img is None:
            img = np.empty(self.shape, dtype=coeffs.dtype)
        self._adj_op(coeffs, img)
        return img

    def _adj_op_sense(self, coeffs, img=None):
        coil_img = np.empty(self.shape, dtype=coeffs.dtype)
        if img is None:
            img = np.zeros(self.shape, dtype=coeffs.dtype)
        self._adj_op(coeffs, coil_img)
        img = np.sum(coil_img * self._smaps.conjugate(), axis=0)
        return img

    def _adj_op_calibless(self, coeffs, img=None):
        if img is None:
            img = np.empty((self.n_coils, *self.shape), dtype=coeffs.dtype)
        self._adj_op(coeffs, img)
        return img

    def _apply_dc(self, coeffs):
        if self.density is not None:
            return coeffs * self.density
        return coeffs

    def _adj_op(self, coeffs, image):
        return self.raw_op.adj_op(self._apply_dc(coeffs), image)

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
        if self.n_coils == 1:
            return self._data_consistency_mono(image_data, obs_data)
        if self.uses_sense:
            return self._data_consistency_sense(image_data, obs_data)
        return self._data_consistency_calibless(image_data, obs_data)

    def _data_consistency_mono(self, image_data, obs_data):
        ksp = np.empty(self.n_samples, dtype=obs_data.dtype)
        img = np.empty(self.shape, dtype=image_data.dtype)
        self._op(image_data, ksp)
        ksp -= obs_data
        self._adj_op(ksp, img)
        return img

    def _data_consistency_sense(self, image_data, obs_data):
        img = np.empty_like(image_data)
        coil_img = np.empty(self.shape, dtype=image_data.dtype)
        coil_ksp = np.empty(self.n_samples, dtype=obs_data.dtype)
        for i in range(self.n_coils):
            np.copyto(coil_img, img)
            coil_img *= self._smap
            self._op(coil_img, coil_ksp)
            coil_ksp -= obs_data[i]
            if self.uses_density:
                coil_ksp *= self.density_d
            self._adj_op(coil_ksp, coil_img)
            sense_adj_mono(img, coil_img, self._smaps[i])
        return img

    def _data_consistency_calibless(self, image_data, obs_data):
        img = np.empty((self.n_coils, *self.shape), dtype=image_data.dtype)
        ksp = np.empty(self.n_samples, dtype=obs_data.dtype)
        for i in range(self.n_coils):
            self._op(image_data[i], ksp)
            ksp -= obs_data[i]
            if self.uses_density:
                ksp *= self.density_d
            self._adj_op(ksp, img[i])
        return img

    @property
    def eps(self):
        """Return the underlying precision parameter."""
        return self.raw_op.eps

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

    def __repr__(self):
        """Return info about the MRICufiNUFFT Object."""
        return (
            "MRICufiNUFFT(\n"
            f"  shape: {self.shape}\n"
            f"  n_coils: {self.n_coils}\n"
            f"  n_samples: {self.n_samples}\n"
            f"  uses_density: {self.uses_density}\n"
            f"  uses_sense: {self._uses_sense}\n"
            f"  smaps_cached: {self.smaps_cached}\n"
            f"  plan_setup: {self.plan_setup}\n"
            f"  eps:{self.raw_op.eps:.0e}\n"
            ")"
        )
