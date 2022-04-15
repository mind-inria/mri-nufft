"""Provides Operator for MR Image processing on gpu."""
import warnings

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gp

from .kernels import update_density, sense_adj_mono, sense_forward
from .raw_operator import RawCufinufft
from .utils import ensure_on_gpu, is_c_array, is_cuda_array


class MRICufi:
    """MRI Transform operator, build around cufinufft.

    This operator adds density estimation and compensation (preconditioning)
    and multicoil support.

    Parameters
    ----------
    samples: np.ndarray or GPUArray.
        The samples location of shape ``Nsamples x N_dimensions``.
    shape: tuple
        Shape of the image space.
    n_coils: int
        Number of coils.
    density: bool or array
       Density compensation support.
        - If array, use this for density compensation
        - If True, the density compensation will be automatically estimated,
          using the fixed point method.
        - If False, density compensation will not be used.
    smaps: np.ndarray or GPUArray , optional
        - If None: no Smaps wil be used.
        - If np.ndarray: Smaps will be copied on the device,
          according to `smaps_cached`.
        - If GPUArray, the smaps are already cached.
    smaps_cached: bool, default False
        - If False the smaps are copied on device and free at each iterations.
        - If True, the smaps are copied on device and stay on it.
    kwargs :
        Extra kwargs for the raw cufinufft operator


    Notes
    -----
    TODO: Add concurrency for multicoil operations
    TODO: Add device context for multi gpu support.

    See Also
    --------
    cufinufft.raw_operator.RawCufinufft
    """

    def __init__(self, samples, shape, density=False, n_coils=1, smaps=None,
                 smaps_cached=False, **kwargs):
        self.shape = shape
        self.n_samples = len(samples)
        if is_c_array(samples):
            samples_d = gp.to_gpu(samples)
        elif is_cuda_array(samples):
            samples_d = samples
        else:
            raise ValueError("Samples should be either a C-ordered ndarray, "
                             "or a GPUArray.")

        # density compensation support
        if density is True:
            self.density_d = MRICufi.estimate_density(samples_d, shape)
            self.uses_density = True
        elif is_c_array(density) or is_cuda_array(density):
            if len(density) != len(samples):
                raise ValueError("Density array and samples array should "
                                 "have the same length.")
            self.uses_density = True

            if is_c_array(density):
                self.density_d = gp.to_gpu(density)
            elif is_cuda_array(density):
                self.density_d = density
        else:
            self.density_d = None
            self.uses_density = False

        self._uses_sense = False
        self.smaps_cached = False
        # Smaps support
        if n_coils < 1:
            raise ValueError("n_coils should be ≥ 1")
        self.n_coils = n_coils
        if smaps is not None:
            self._uses_sense = True
            if not(is_c_array(smaps) and is_cuda_array(smaps)):
                raise ValueError("Smaps should be either a C-ordered ndarray, "
                                 "or a GPUArray.")
            if is_cuda_array(smaps):
                self._smaps_d = smaps
            elif smaps_cached and is_c_array(smaps):
                warnings.warn(f"{smaps.nbytes/2**30} GiB will be used on gpu.")
                self._smaps_d = gp.to_gpu(smaps)
                self.smaps_cached = True
                self._smaps = smaps

            else:
                # allocate device memory
                self._smap_d = gp.empty(shape, dtype=np.complex64)
                # move smaps to pinned memory
                self._smaps = cuda.register_host_memory(smaps)  # pylint: disable=E1101 # noqa: E501
        else:
            self._uses_sense = False
            self._smaps = None
        # Initialize data holders on device.
        if self.uses_sense or self.n_coils == 1:
            self.image_data_d = gp.empty(self.shape, dtype=np.complex64)
        else:
            self.image_data_d = gp.empty((self.n_coils, *self.shape),
                                         dtype=np.complex64)
        self.kspace_data_d = gp.empty((self.n_coils, *self.shape),
                                      dtype=np.complex64)
        # Initialise NUFFT plans
        self.raw_op = RawCufinufft(samples_d, shape, **kwargs)

        # Usefull data sizes:
        self.image_coil_offset = np.prod(self.shape) * np.complex64.itemsize
        self.kspace_coil_offset = self.n_samples * np.complex64.itemsize

    def op(self, data):
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
        if self.n_coils == 1:
            return self._op(data)

        if self.uses_sense:
            for i in range(self.n_coils):
                if self.smaps_cached and is_cuda_array(data):
                    self._smap_d = self._smaps_d[i]
                    self.image_data_d = data[i]
                elif self.smaps_cached and is_c_array(data):
                    self._smap_d = self._smaps[i]
                    self.image_data_d.set(data[i])
                elif not self.smaps_cached and is_cuda_array(data):
                    self._smap_d = self._smaps[i]
                    self.image_data_d = data[i]
                else:  # everything is on cpu currently
                    self.image_data_d.set(data[i])
                    self._smap_d.set(self._smaps[i])
            sense_forward(self.image_data_d, self._smap_d)
            self.__op(self.image_data_d.ptr,
                      self.kspace_data_d.ptr + i * self.kspace_coil_offset)
            if is_cuda_array(data):
                return self.kspace_data_d
            return self.kspace_data_d.get()
        if is_cuda_array(data):
            for i in range(self.n_coils):
                self.__op(
                    data.ptr + i * self.image_coil_offset,
                    self.kspace_data_d.ptr + i * self.kspace_coil_offset)
            return self.kspace_data_d
        for i in range(self.n_coils):
            self.image_data_d.set(data[i])
            self.__op(
                self.image_data_d.ptr,
                self.kspace_data_d.ptr + i * self.kspace_coil_offset)
        return self.kspace_data_d.get()

    def _op(self, data, coeff_d=None):
        coeff_d = self.kspace_data_d if coeff_d is None else coeff_d
        data_d = ensure_on_gpu(data)
        self.__op(coeff_d, data_d)
        if is_c_array(data):
            return coeff_d.get()
        return coeff_d

    def __op(self, image_d, coeffs_d):
        if not isinstance(image_d, int):
            return self.raw_op.type2(coeffs_d.ptr, image_d.ptr)
        return self.raw_op.type2(coeffs_d, image_d)

    def adj_op(self, coeffs):
        """Non Cartesian MRI adjoint operator.

        Parameters
        ----------
        coeffs: np.array or GPUArray

        Returns
        -------
        Array in the same memory space of coeffs. (ie on cpu or gpu Memory).
        """
        if self.n_coils == 1:
            return self._adj_op(coeffs)
        if is_cuda_array(coeffs):
            self.kspace_data_d = coeffs
        else:
            self.kspace_data_d.set(coeffs)

        if self.uses_sense:
            coil_image_d = gp.empty(self.shape, np.complex64)
            for i in range(self.n_coils):
                if self.smaps_cached:
                    self.__adj_op(
                        self.kspace_data_d.ptr + i * self.kspace_coil_offset,
                        coil_image_d.ptr)
                    sense_adj_mono(self.image_data_d,
                                   coil_image_d, self._smaps_d[i])
                else:
                    self._smap_d.set(self._smaps[i])
                    self.__adj_op(
                        self.kspace_data_d.ptr + i * self.kspace_coil_offset,
                        coil_image_d.ptr)
                    sense_adj_mono(self.image_data_d,
                                   coil_image_d, self._smap_d)
        else:
            for i in range(self.n_coils):
                self.__adj_op(
                    self.kspace_data_d.ptr + i * self.kspace_coil_offset,
                    self.image_data_d.ptr + i * self.image_coil_offset)

        if is_cuda_array(coeffs):
            return self.image_data_d
        return self.image_data_d.get()

    def _adj_op(self, coeffs, image_d=None):
        """Non Cartesian MRI  single coil adjoint operator."""
        image_d = self.image_data_d if image_d is None else image_d

        coeffs_d = ensure_on_gpu(coeffs)
        if self.uses_density:
            coeffs_d *= self.density_d
        self.__adj_op(coeffs_d, image_d)

        if is_c_array(coeffs):
            return image_d.get()
        # image_data has been updated.
        return None

    def __adj_op(self, coeffs_d, image_d):
        if not isinstance(coeffs_d, int) and isinstance(image_d, int):
            return self.raw_op.type1(coeffs_d.ptr, image_d.ptr)
        return self.raw_op.type1(coeffs_d, image_d)

    def get_device_memory_size(self):
        """Get the size in bytes of allocated device memory for this object."""
        raise NotImplementedError

    def _get_spec_rad(self):
        pass

    @property
    def spec_rad(self):
        return None

    @property
    def uses_sense(self):
        """Return True if the transform uses the SENSE method, else False."""
        return self._uses_sense

    @classmethod
    def estimate_density(cls, samples, shape, n_iter=10, **kwargs):
        """Estimate the density compensation array."""
        oper = cls(samples, shape, density=False, **kwargs)

        density = gp.empty(samples.shape[0], dtype=np.complex64)
        density.fill(np.ones((), dtype=np.complex64))
        update = gp.empty_like(density)
        img = gp.empty(shape, dtype=np.complex64)
        for _ in range(n_iter):
            oper.__adj_op(density, img)
            oper.__op(img, update)
            update_density(density, update)
        return density.real
