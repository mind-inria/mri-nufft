"""Provides Operator for MR Image processing on gpu."""

import warnings
import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as pg

from .raw_operator import RawCufinufft
from .kernels import update_density
from .utils import is_cuda_array, is_c_array, ensure_on_gpu


class MRICufi:
    """MRI Transform operator, build around cufinufft.

    This provides extra step to reproduce the MR acquisition process.

    Parameters
    ----------
    samples :
    shape : tuple
    n_coils : int
        Number of coils
    density : bool or array
       Density compensation support.
        - If array, use this for density compensation
        - If True, the density compensation will be automatically estimated,
          using the fixed point method.
    smaps : array
        Sensitivity maps.
    smap_cache : bool, default True
        If True, the Smaps are cached on the gpu. This may be very expensive for 3D data.
        If False, the Smaps are only copied coil wised when needed.
    kwargs :
        Extra kwargs for the raw cufinufft operator

    See Also
    --------
    cufinufft.raw_operator.RawCufinufft
    """

    def __init__(self, samples, shape,
                 n_coils=1, density=False, **kwargs):

        self.shape = shape
        self.n_samples = len(samples)
        if is_c_array(samples):
            samples_d = pg.to_gpu(samples)
        elif is_cuda_array(samples):
            samples_d = samples
        else:
            raise ValueError("Samples should be either a C-ordered ndarray, "
                             "or a GPUArray.")
        self.n_coils = n_coils

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
                self.density_d = pg.to_gpu(density)
            elif is_cuda_array(density):
                self.density_d = density
        else:
            self.density_d = None
            self.uses_density = False

        self.raw_op = RawCufinufft(samples_d, shape, **kwargs)

    def op(self, data, coeffs_d=None):
        """Non Cartesian MRI operator."""
        data_d = ensure_on_gpu(data)
        if coeffs_d is None:
            coeffs_d = pg.GPUArray(self.n_samples, self.raw_op.complex_dtype)
        self.raw_op.type2(coeffs_d, data_d)

        if is_cuda_array(data):
            return coeffs_d.get()
        return coeffs_d

    def adj_op(self, coeffs, data_d=None):
        """Non Cartesian MRI adjoint operator."""
        return self._adj_op(coeffs, data_d)

    def _adj_op(self, coeffs, data_d=None):
        """Non Cartesian MRI  single coil adjoint operator."""
        coeffs_d = ensure_on_gpu(coeffs)
        if data_d is None:
            data_d = pg.GPUArray(self.shape, self.raw_op.complex_dtype)

        if self.uses_density:
            coeffs_d *= self.density_d

        self.raw_op.type1(coeffs_d, data_d)

        if is_cuda_array(coeffs):
            return data_d.get()
        return data_d

    @classmethod
    def estimate_density(cls, samples, shape, n_iter=10, **kwargs):
        """Estimate the density compensation array."""
        oper = cls(samples, shape, density=False, n_coils=1, **kwargs)

        density = pg.empty(samples.shape[0], dtype=np.complex64)
        density.fill(np.ones((), dtype=np.complex64))
        update = pg.empty_like(density)
        img = pg.empty(shape, dtype=np.complex64)
        for i in range(n_iter):
            oper.adj_op(density, img)
            oper.op(img, update)
            update_density(density, update)
        return density.real


class MRICufiAsync(MRICufi):
    """MRI Cufinufft with Async Memory transfers and preallocated buffers."""

    def __init__(self, samples, shape, n_coils, concurrent,  **kwargs):
        super().__init__(samples, shape, n_coils, **kwargs)
