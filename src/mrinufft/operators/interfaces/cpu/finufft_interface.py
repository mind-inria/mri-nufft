"""Finufft interface."""
import warnings

import numpy as np

from ..base import FourierOperatorBase
from ._finufft_interface import FINUFFT_AVAILABLE, RawFinufft


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
        verbose=False,
        plan_setup="persist",
        **kwargs,
    ):
        if not FINUFFT_AVAILABLE:
            raise RuntimeError("finufft is not available.")

        self.shape = shape
        self.n_samples = len(samples)
        if samples.max() > np.pi:
            warnings.warn("samples will be normalized in [-pi, pi]")
            samples *= np.pi / samples.max()
            samples = samples.astype(np.float32)
        # we will access the samples by their coordinate first.
        self.samples = np.asfortranarray(samples)

        if density is True:
            self.density = MRIfinufft.estimate_density(samples, shape)
            self.uses_density = True
        elif isinstance(density, np.ndarray):
            if len(density) != len(samples):
                raise ValueError(
                    "Density array and samples array should have the same length."
                )
            self.uses_density = True
            self.density = np.asfortranarray(density)
        else:
            self.density = None
            self.uses_density = False
        self._uses_sense = False

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

        # Initialise NUFFT plans
        if plan_setup not in ["persist", "multicoil", "single"]:
            raise ValueError(
                "plan_setup should be either 'persist'," "'multicoil' or 'single'"
            )
        self.plan_setup = plan_setup
        self.raw_op = RawFinufft(
            self.samples, tuple(shape), init_plans=plan_setup == "persist", **kwargs
        )

    def op(self, data, ksp=None):
        r"""Non Cartesian MRI forward operator.

        Parameters
        ----------
        data: np.ndarray or GPUArray
        The uniform (2D or 3D) data in image space.

        Returns
        -------
        Results array on the same device as data.

        Notes
        -----
        this performs for every coil \ell:
        ..math:: \mathcal{F}\mathcal{S}_\ell x
        """
        # monocoil
        if self.plan_setup == "multicoil":
            self.raw_op._make_plan(2)
            self.raw_op._set_pts(2)
        if self.n_coils == 1:
            ret = self._op_mono(data, ksp)
        # sense
        elif self.uses_sense:
            ret = self._op_sense(data, ksp)
        # calibrationless, data on device
        else:
            ret = self._op_calibless(data, ksp)

        if self.plan_setup == "multicoil":
            self.raw_op._destroy_plan(2)
        return ret

    def _op_mono(self, data, ksp=None):
        if ksp is None:
            ksp = np.empty(self.n_samples, dtype=np.complex64)
        self.__op(data, ksp)
        return ksp

    def _op_sense(self, data, ksp_d=None):
        coil_img = np.empty(self.shape, dtype=np.complex64)
        ksp = np.zeros((self.n_coils, self.n_samples), dtype=np.complex64)
        for i in range(self.n_coils):
            np.copyto(coil_img, data)
            coil_img *= self._smaps[i]  # sense forward
            self.__op(coil_img, ksp[i])
        return ksp

    def _op_calibless(self, data, ksp=None):
        if ksp is None:
            ksp = np.empty((self.n_coils, self.n_samples), dtype=np.complex64)
        for i in range(self.n_coils):
            self.__op(data[i], ksp[i])
        return ksp

    def __op(self, image, coeffs):
        # ensure everything is pointers before going to raw level.
        return self.raw_op.type2(coeffs, image)

    def adj_op(self, coeffs, img=None):
        """Non Cartesian MRI adjoint operator.

        Parameters
        ----------
        coeffs: np.array or GPUArray

        Returns
        -------
        Array in the same memory space of coeffs. (ie on cpu or gpu Memory).
        """
        if self.plan_setup == "multicoil":
            self.raw_op._make_plan(1)
            self.raw_op._set_pts(1)
        if self.n_coils == 1:
            ret = self._adj_op_mono(coeffs, img)
        # sense
        elif self.uses_sense:
            ret = self._adj_op_sense(coeffs, img)
        # calibrationless
        else:
            ret = self._adj_op_calibless(coeffs, img)

        if self.plan_setup == "multicoil":
            self.raw_op._destroy_plan(1)

        return ret

    def _adj_op_mono(self, coeffs, img=None):
        if img is None:
            img = np.empty(self.shape, dtype=np.complex64)
        if self.uses_density:
            coil_ksp = np.copy(coeffs)
            coil_ksp *= self.density  # density preconditionning
        else:
            coil_ksp = coeffs
        self.__adj_op(coil_ksp, img)
        return img

    def _adj_op_sense(self, coeffs, img=None):
        coil_img = np.empty(self.shape, dtype=np.complex64)
        if img is None:
            img = np.zeros(self.shape, dtype=np.complex64)
        coil_ksp = np.empty(self.n_samples, dtype=np.complex64)
        for i in range(self.n_coils):
            if self.uses_density:
                coil_ksp = np.copy(coeffs[i])
                coil_ksp *= self.density
            else:
                coil_ksp = coeffs[i]
            self.__adj_op(coil_ksp, coil_img)
            sense_adj_mono(img, coil_img, self._smaps[i])
        return img

    def _adj_op_calibless(self, coeffs, img_d=None):
        if img_d is None:
            img_d = cp.empty((self.n_coils, *self.shape), dtype=np.complex64)
        coil_ksp_d = cp.empty(self.n_samples, dtype=np.complex64)
        for i in range(self.n_coils):
            if self.uses_density:
                cp.copyto(coil_ksp_d, coeffs[i])
                coil_ksp_d *= self.density_d
                self.__adj_op(coil_ksp_d.data.ptr, img_d.data.ptr + i * self.img_size)
            else:
                self.__adj_op(
                    coeffs.data.ptr + i * self.ksp_size,
                    img_d.data.ptr + i * self.img_size,
                )
        return img_d

    def __adj_op(self, coeffs, image):
        return self.raw_op.type1(coeffs, image)

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
        img_d = cp.array(image_data, copy=True)
        obs_d = cp.asarray(obs_data)
        ksp_d = cp.empty(self.n_samples, dtype=np.complex64)
        self.__op(img_d, ksp_d)
        ksp_d -= obs_d
        self.__adj_op(ksp_d, img_d)
        if is_cuda_array(image_data):
            return img_d
        return img_d.get()

    def _data_consistency_sense(self, image_data, obs_data):
        img_d = cp.array(image_data, copy=True)
        coil_img_d = cp.empty(self.shape, dtype=np.complex64)
        coil_ksp_d = cp.empty(self.n_samples, dtype=np.complex64)
        if is_host_array(obs_data):
            coil_obs_data = cp.empty(self.n_samples, dtype=np.complex64)
            obs_data_pinned = pin_memory(obs_data)
            for i in range(self.n_coils):
                cp.copyto(coil_img_d, img_d)
                if self.smaps_cached:
                    coil_img_d *= self._smaps_d[i]
                else:
                    self._smap_d.set(self._smaps[i])
                    coil_img_d *= self._smap_d
                self.__op(coil_img_d.data.ptr, coil_ksp_d.data.ptr)
                coil_obs_data = cp.asarray(obs_data_pinned[i])
                coil_ksp_d -= coil_obs_data
                if self.uses_density:
                    coil_ksp_d *= self.density_d
                self.__adj_op(coil_ksp_d.data.ptr, coil_img_d.data.ptr)
                if self.smaps_cached:
                    sense_adj_mono(img_d, coil_img_d, self._smaps_d[i])
                else:
                    sense_adj_mono(img_d, coil_img_d, self._smap_d)
            del obs_data_pinned
            return img_d.get()
        for i in range(self.n_coils):
            cp.copyto(coil_img_d, img_d)
            if self.smaps_cached:
                coil_img_d *= self._smaps_d[i]
            else:
                self._smap_d.set(self._smaps[i])
                coil_img_d *= self._smap_d
            self.__op(coil_img_d.data.ptr, coil_ksp_d.data.ptr)
            coil_ksp_d -= obs_data[i]
            if self.uses_density:
                coil_ksp_d *= self.density_d
            self.__adj_op(coil_ksp_d.data.ptr, coil_img_d.data.ptr)
            if self.smaps_cached:
                sense_adj_mono(img_d, coil_img_d, self._smaps_d[i])
            else:
                sense_adj_mono(img_d, coil_img_d, self._smap_d)
        return img_d

    def _data_consistency_calibless(self, image_data, obs_data):
        if is_cuda_array(image_data):
            img_d = cp.empty((self.n_coils, *self.shape), dtype=np.complex64)
            ksp_d = cp.empty(self.n_samples, dtype=np.complex64)
            for i in range(self.n_coils):
                self.__op(image_data.data.ptr + i * self.img_size, ksp_d.data.ptr)
                ksp_d -= obs_data[i]
                if self.uses_density:
                    ksp_d *= self.density_d
                self.__adj_op(ksp_d.data.ptr, img_d.data.ptr + i * self.img_size)
            return img_d

        img_d = cp.empty(self.shape, dtype=np.complex64)
        img = np.zeros((self.n_coils, *self.shape), dtype=np.complex64)
        ksp_d = cp.empty(self.n_samples, dtype=np.complex64)
        obs_d = cp.empty(self.n_samples, dtype=np.complex64)
        for i in range(self.n_coils):
            img_d.set(image_data[i])
            obs_d.set(obs_data[i])
            self.__op(img_d.data.ptr, ksp_d.data.ptr)
            ksp_d -= obs_d
            if self.uses_density:
                ksp_d *= self.density_d
            self.__adj_op(ksp_d.data.ptr, img_d.data.ptr)
            cp.asnumpy(img_d, out=img[i])
        return img

    @property
    def eps(self):
        """Return the underlying precision parameter."""
        return self.raw_op.eps

    @classmethod
    def estimate_density(cls, samples, shape, n_iter=10, **kwargs):
        """Estimate the density compensation array."""
        oper = cls(samples, shape, density=False, **kwargs)

        density = np.ones(len(samples), dtype=np.complex64)
        update = np.empty_like(density)
        img = np.empty(shape, dtype=np.complex64)
        for _ in range(n_iter):
            oper.__adj_op(density, img)
            oper.__op(img, update)
            update_density(density, update)
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
