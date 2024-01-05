"""Provides Operator for MR Image processing on GPU."""

import warnings
import numpy as np
from ..base import FourierOperatorBase, proper_trajectory
from .utils import (
    CUPY_AVAILABLE,
    check_size,
    get_ptr,
    is_cuda_array,
    is_host_array,
    nvtx_mark,
    pin_memory,
    sizeof_fmt,
)

CUFINUFFT_AVAILABLE = CUPY_AVAILABLE
try:
    import cupy as cp
    from cufinufft._plan import Plan
except ImportError:
    CUFINUFFT_AVAILABLE = False


OPTS_FIELD_DECODE = {
    "gpu_method": {1: "nonuniform pts driven", 2: "shared memory"},
    "gpu_sort": {0: "no sort (GM)", 1: "sort (GM-sort)"},
    "kerevalmeth": {0: "direct eval exp(sqrt())", 1: "Horner ppval"},
    "gpu_spreadinterponly": {
        0: "NUFFT",
        1: "spread or interpolate only",
    },
}

DTYPE_R2C = {"float32": "complex64", "float64": "complex128"}


def _error_check(ier, msg):
    if ier != 0:
        raise RuntimeError(msg)


class RawCufinufftPlan:
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

    @property
    def dtype(self):
        """Return the dtype (precision) of the transform."""
        try:
            return self.plans[1].dtype
        except AttributeError:
            return DTYPE_R2C[str(self.samples.dtype)]

    def _make_plan(self, typ, **kwargs):
        self.plans[typ] = Plan(
            typ,
            self.shape,
            self.n_trans,
            self.eps,
            dtype=DTYPE_R2C[str(self.samples.dtype)],
            **kwargs,
        )

    def _set_pts(self, typ):
        x, y, z = None, None, None
        x = cp.array(self.samples[:, 0])
        y = cp.array(self.samples[:, 1])
        self.plans[typ]._references = [x, y]

        fpts_axes = [get_ptr(y), get_ptr(x), None]
        if self.ndim == 3:
            z = cp.array(self.samples[:, 2])
            self.plans[typ]._references.append(z)
            fpts_axes.insert(0, get_ptr(z))
        M = x.size
        self.plans[typ]._setpts(
            self.plans[typ]._plan, M, *fpts_axes[:3], 0, None, None, None
        )

    def _destroy_plan(self, typ):
        if self.plans[typ] is not None:
            p = self.plans[typ]
            del p
            self.plans[typ] = None

    def type1(self, coeff_data_ptr, grid_data_ptr):
        """Type 1 transform. Non Uniform to Uniform."""
        ier = self.plans[1]._exec_plan(
            self.plans[1]._plan, coeff_data_ptr, grid_data_ptr
        )
        _error_check(ier, "Error in type 1 transform")

    def type2(self, coeff_data_ptr, grid_data_ptr):
        """Type 2 transform. Uniform to non-uniform."""
        ier = self.plans[2]._exec_plan(
            self.plans[2]._plan, coeff_data_ptr, grid_data_ptr
        )
        _error_check(ier, "Error in type 2 transform")


class MRICufiNUFFT(FourierOperatorBase):
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
    n_batchs: int
        Size of the batch dimension.
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
    squeeze_dims: bool, default False
        If True, will try to remove the singleton dimension for batch and coils.
    n_trans: int, default 1
        Number of transform to perform in parallel by cufinufft.
    kwargs :
        Extra kwargs for the raw cufinufft operator


    Notes
    -----
    Cufinufft is able to run multiple transform in parallel, this is controlled
    by the n_trans parameter. The data provided should be of shape, (n_batch,
    n_coils, img_shape) for op (type2) and (n_batch, n_coils, n_samples) for
    adjoint (type1). and in contiguous memory order.

    For now only single precision (float32 and complex64) is supported

    See Also
    --------
    cufinufft.raw_operator.RawCufinufft
    """

    backend = "cufinufft"
    available = CUFINUFFT_AVAILABLE and CUPY_AVAILABLE

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        n_batchs=1,
        smaps=None,
        smaps_cached=False,
        verbose=False,
        squeeze_dims=False,
        n_trans=1,
        **kwargs,
    ):
        # run the availaility check here to get detailled output.
        if not CUPY_AVAILABLE:
            raise RuntimeError("cupy is not installed")
        if not CUFINUFFT_AVAILABLE:
            raise RuntimeError("Failed to found cufinufft binary.")

        super().__init__()
        if (n_batchs * n_coils) % n_trans != 0:
            raise ValueError("n_batchs * n_coils should be a multiple of n_transf")

        self.shape = shape
        self.n_batchs = n_batchs
        self.n_trans = n_trans
        self.squeeze_dims = squeeze_dims
        self.n_coils = n_coils
        # For now only single precision is supported
        self.samples = np.asfortranarray(
            proper_trajectory(samples, normalize="pi").astype(np.float32)
        )
        self.dtype = self.samples.dtype

        # density compensation support
        if density is True:
            self.density = self.pipe(self.samples, shape)
        elif is_host_array(density) or is_cuda_array(density):
            self.density = cp.array(density)
        else:
            self.density = None

        # Smaps support
        self.smaps = smaps
        self.smaps_cached = False
        if smaps is not None:
            if not (is_host_array(smaps) or is_cuda_array(smaps)):
                raise ValueError(
                    "Smaps should be either a C-ordered ndarray, " "or a GPUArray."
                )
            self.smaps_cached = False
            if smaps_cached:
                warnings.warn(
                    f"{sizeof_fmt(smaps.size * np.dtype(self.cpx_dtype).itemsize)}"
                    "used on gpu for smaps."
                )
                self.smaps = cp.array(
                    smaps, order="C", copy=False, dtype=self.cpx_dtype
                )
                self.smaps_cached = True
            else:
                self.smaps = pin_memory(smaps.astype(self.cpx_dtype))
                self._smap_d = cp.empty(self.shape, dtype=self.cpx_dtype)

        self.raw_op = RawCufinufftPlan(
            self.samples,
            tuple(shape),
            n_trans=n_trans,
            **kwargs,
        )
        # Support for concurrent stream and computations.

    @nvtx_mark()
    def op(self, data, ksp_d=None):
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
        if self.uses_sense:
            check_size(data, (self.n_batchs, *self.shape))
        else:
            check_size(data, (self.n_batchs, self.n_coils, *self.shape))
        data = data.astype(self.cpx_dtype)

        # Dispatch to special case.
        if self.uses_sense and is_cuda_array(data):
            op_func = self._op_sense_device
        elif self.uses_sense:
            op_func = self._op_sense_host
        elif is_cuda_array(data):
            op_func = self._op_calibless_device
        else:
            op_func = self._op_calibless_host
        ret = op_func(data, ksp_d)

        ret /= self.norm_factor
        return self._safe_squeeze(ret)

    def _op_sense_device(self, data, ksp_d=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        image_dataf = cp.reshape(data, (B, *XYZ))
        ksp_d = ksp_d or cp.empty((B * C, K), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        for i in range(B * C // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            data_batched = image_dataf[idx_batch].reshape((T, *XYZ))
            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils].reshape((T, *XYZ)))
            else:
                smaps_batched = self.smaps[idx_coils].reshape((T, *XYZ))
            data_batched *= smaps_batched
            self.__op(get_ptr(data_batched), get_ptr(ksp_d[i * T : (i + 1) * T]))

        return ksp_d.reshape((B, C, K))

    def _op_sense_host(self, data, ksp=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        dataf = data.reshape((B, *XYZ))
        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        if ksp is None:
            ksp = np.empty((B, C, K), dtype=self.cpx_dtype)
        ksp = ksp.reshape((B * C, K))
        ksp_batched = cp.empty((T, K), dtype=self.cpx_dtype)

        for i in range(B * C // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            data_batched.set(dataf[idx_batch].reshape((T, *XYZ)))
            if not self.smaps_cached:
                coil_img_d.set(self.smaps[idx_coils].reshape((T, *XYZ)))
            else:
                cp.copyto(coil_img_d, self.smaps[idx_coils])
            coil_img_d *= data_batched
            self.__op(get_ptr(coil_img_d), get_ptr(ksp_batched))

            ksp[i * T : (i + 1) * T] = ksp_batched.get()
        ksp = ksp.reshape((B, C, K))
        return ksp

    def _op_calibless_device(self, data, ksp_d=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K = self.n_samples
        bsize_samples2gpu = T * self.ksp_size
        bsize_img2gpu = T * self.img_size
        if ksp_d is None:
            ksp_d = cp.empty((B, C, K), dtype=self.cpx_dtype)
        for i in range((B * C) // T):
            self.__op(
                get_ptr(data) + i * bsize_img2gpu,
                get_ptr(ksp_d) + i * bsize_samples2gpu,
            )
        return ksp_d

    def _op_calibless_host(self, data, ksp=None):
        # calibrationless, data on host
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        coil_img_d = cp.empty(np.prod(XYZ) * T, dtype=self.cpx_dtype)
        ksp_d = cp.empty((T, K), dtype=self.cpx_dtype)
        if ksp is None:
            ksp = np.zeros((B * C, K), dtype=self.cpx_dtype)
        ksp = ksp.reshape((B * C, K))
        # TODO: Add concurrency compute batch n while copying batch n+1 to device
        # and batch n-1 to host
        dataf = data.flatten()
        size_batch = T * np.prod(XYZ)
        for i in range((B * C) // T):
            coil_img_d.set(dataf[i * size_batch : (i + 1) * size_batch])
            self.__op(get_ptr(coil_img_d), get_ptr(ksp_d))
            ksp[i * T : (i + 1) * T] = ksp_d.get()
        ksp = ksp.reshape((B, C, K))
        return ksp

    @nvtx_mark()
    def __op(self, image_d, coeffs_d):
        # ensure everything is pointers before going to raw level.
        if is_cuda_array(image_d) and is_cuda_array(coeffs_d):
            return self.raw_op.type2(get_ptr(coeffs_d), get_ptr(image_d))
        return self.raw_op.type2(coeffs_d, image_d)

    @nvtx_mark()
    def adj_op(self, coeffs, img_d=None):
        """Non Cartesian MRI adjoint operator.

        Parameters
        ----------
        coeffs: np.array or GPUArray

        Returns
        -------
        Array in the same memory space of coeffs. (ie on cpu or gpu Memory).
        """
        check_size(coeffs, (self.n_batchs, self.n_coils, self.n_samples))
        # Dispatch to special case.
        if self.uses_sense and is_cuda_array(coeffs):
            adj_op_func = self._adj_op_sense_device
        elif self.uses_sense:
            adj_op_func = self._adj_op_sense_host
        elif is_cuda_array(coeffs):
            adj_op_func = self._adj_op_calibless_device
        else:
            adj_op_func = self._adj_op_calibless_host

        ret = adj_op_func(coeffs, img_d)
        ret /= self.norm_factor
        return self._safe_squeeze(ret)

    def _adj_op_sense_device(self, coeffs, img_d=None):
        """Perform sense reconstruction when data is on device."""
        # Define short name
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        # Allocate memory
        if img_d is None:
            img_d = cp.zeros((B, *XYZ), dtype=self.cpx_dtype)
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        if self.uses_density:
            ksp_new = cp.empty((T, K), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        for i in range(B * C // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils])
            else:
                smaps_batched = self.smaps[idx_coils]
            if self.uses_density:
                cp.copyto(ksp_new, coeffs[i * T : (i + 1) * T])
                ksp_new *= self.density
            else:
                ksp_new = coeffs[i * T : (i + 1) * T]
            self.__adj_op(get_ptr(ksp_new), get_ptr(coil_img_d))
            for t, b in enumerate(idx_batch):
                img_d[b, :] += coil_img_d[t] * smaps_batched[t].conj()
        img_d = img_d.reshape((B, 1, *XYZ))
        return img_d

    def _adj_op_sense_host(self, coeffs, img_d=None):
        """Perform sense reconstruction when data is on host.

        On device the following array are involved:
        - coil_img(S, T, 1, X,Y,Z)
        - ksp_batch(B, 1, X,Y,Z)
        - smaps_batched(S, T, X,Y,Z)
        - density_batched(T, K)

        """
        # Define short name
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        coeffs_f = coeffs.flatten()
        # Allocate memory
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        if img_d is None:
            img_d = cp.zeros((B, *XYZ), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        ksp_batched = cp.empty((T, K), dtype=self.cpx_dtype)
        if self.uses_density:
            density_batched = cp.repeat(self.density[None, :], T, axis=0)
        for i in range(B * C // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils])
            else:
                smaps_batched = self.smaps[idx_coils]
            ksp_batched.set(coeffs_f[i * T * K : (i + 1) * T * K].reshape(T, K))
            if self.uses_density:
                ksp_batched *= density_batched
            self.__adj_op(get_ptr(ksp_batched), get_ptr(coil_img_d))

            for t, b in enumerate(idx_batch):
                img_d[b, :] += coil_img_d[t] * smaps_batched[t].conj()
        img = img_d.get()
        img = img.reshape((B, 1, *XYZ))
        return img

    def _adj_op_calibless_device(self, coeffs, img_d=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        coeffs_f = coeffs.flatten()
        ksp_batched = cp.empty(T * K, dtype=self.cpx_dtype)
        if self.uses_density:
            density_batched = cp.repeat(self.density[None, :], T, axis=0).flatten()
        img_d = img_d or cp.empty((B, C, *XYZ), dtype=self.cpx_dtype)
        for i in range((B * C) // T):
            if self.uses_density:
                cp.copyto(ksp_batched, coeffs_f[i * T * K : (i + 1) * T * K])
                ksp_batched *= density_batched
                self.__adj_op(get_ptr(ksp_batched), get_ptr(img_d) + i * self.bsize_img)
            else:
                self.__adj_op(
                    get_ptr(coeffs_f) + i * self.bsize_ksp,
                    get_ptr(img_d) + i * self.bsize_img,
                )
        return img_d

    def _adj_op_calibless_host(self, coeffs, img_batched=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        coeffs_f = coeffs.flatten()
        ksp_batched = cp.empty(T * K, dtype=self.cpx_dtype)
        if self.uses_density:
            density_batched = cp.repeat(self.density[None, :], T, axis=0).flatten()

        img = np.zeros((B * C, *XYZ), dtype=self.cpx_dtype)
        if img_batched is None:
            img_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        # TODO: Add concurrency compute batch n while copying batch n+1 to device
        # and batch n-1 to host
        for i in range((B * C) // T):
            ksp_batched.set(coeffs_f[i * T * K : (i + 1) * T * K])
            if self.uses_density:
                ksp_batched *= density_batched
            self.__adj_op(get_ptr(ksp_batched), get_ptr(img_batched))
            img[i * T : (i + 1) * T] = img_batched.get()
        img = img.reshape((B, C, *XYZ))
        return img

    @nvtx_mark()
    def __adj_op(self, coeffs_d, image_d):
        if not isinstance(coeffs_d, int):
            ret = self.raw_op.type1(get_ptr(coeffs_d), get_ptr(image_d))
        else:
            ret = self.raw_op.type1(coeffs_d, image_d)
        return ret

    def get_grad(self, image_data, obs_data):
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
        B, C = self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        check_size(obs_data, (B, C, K))
        if self.uses_sense:
            check_size(image_data, (B, *XYZ))
        else:
            check_size(image_data, (B, C, *XYZ))

        if self.uses_sense and is_host_array(image_data):
            grad_func = self._grad_sense_host
        elif self.uses_sense and is_cuda_array(image_data):
            grad_func = self._grad_sense_device
        elif not self.uses_sense and is_host_array(image_data):
            grad_func = self._grad_calibless_host
        elif not self.uses_sense and is_cuda_array(image_data):
            grad_func = self._grad_calibless_device
        else:
            raise ValueError("No suitable gradient function found.")
        ret = grad_func(image_data, obs_data)
        return self._safe_squeeze(ret)

    def _grad_sense_host(self, image_data, obs_data):
        """Gradient computation when all data is on host."""
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        image_dataf = np.reshape(image_data, (B, *XYZ))
        obs_dataf = np.reshape(obs_data, (B * C, K))

        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)

        ksp_batched = cp.empty((T, K), dtype=self.cpx_dtype)
        obs_batched = cp.empty((T, K), dtype=self.cpx_dtype)

        grad_d = cp.zeros((B, *XYZ), dtype=self.cpx_dtype)
        grad = np.empty((B, *XYZ), dtype=self.cpx_dtype)
        for i in range(B * C // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            data_batched.set(image_dataf[idx_batch].reshape((T, *XYZ)))
            obs_batched.set(obs_dataf[i * T : (i + 1) * T])

            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils].reshape((T, *XYZ)))
            else:
                smaps_batched = self.smaps[idx_coils].reshape((T, *XYZ))
            data_batched *= smaps_batched
            self.__op(get_ptr(data_batched), get_ptr(ksp_batched))

            ksp_batched /= self.norm_factor
            ksp_batched -= obs_batched

            if self.uses_density:
                ksp_batched *= self.density
            self.__adj_op(get_ptr(ksp_batched), get_ptr(data_batched))

            for t, b in enumerate(idx_batch):
                grad_d[b, :] += data_batched[t] * smaps_batched[t].conj()
        grad_d /= self.norm_factor
        grad = grad_d.get()
        grad = grad.reshape((B, 1, *XYZ))
        return grad

    def _grad_sense_device(self, image_data, obs_data):
        """Gradient computation when all data is on device."""
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        image_dataf = cp.reshape(image_data, (B, *XYZ))
        obs_dataf = cp.reshape(obs_data, (B * C, K))
        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        ksp_batched = cp.empty((T, K), dtype=self.cpx_dtype)
        grad = cp.zeros((B, *XYZ), dtype=self.cpx_dtype)

        for i in range(B * C // T):
            idx_coils = np.arange(i * T, (i + 1) * T) % C
            idx_batch = np.arange(i * T, (i + 1) * T) // C
            data_batched.set(image_dataf[i * T : (i + 1) * T])
            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils].reshape((T, *XYZ)))
            else:
                smaps_batched = self.smaps[idx_coils].reshape((T, *XYZ))
            data_batched *= smaps_batched
            self.__op(get_ptr(data_batched), get_ptr(ksp_batched))
            ksp_batched /= self.norm_factor
            ksp_batched -= obs_dataf[i * T : (i + 1) * T]

            if self.uses_density:
                ksp_batched *= self.density
            self.__adj_op(get_ptr(ksp_batched), get_ptr(data_batched))

            for t, b in enumerate(idx_batch):
                # TODO write a kernel for that.
                grad[b] += data_batched[t] * smaps_batched[t].conj()
        grad = grad.reshape((B, 1, *XYZ))
        grad /= self.norm_factor
        return grad

    def _grad_calibless_host(self, image_data, obs_data):
        """Calibrationless Gradient computation when all data is on host."""
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        image_dataf = np.reshape(image_data, (B * C, *XYZ))
        obs_dataf = np.reshape(obs_data, (B * C, K))

        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)

        ksp_batched = cp.empty((T, K), dtype=self.cpx_dtype)
        obs_batched = cp.empty((T, K), dtype=self.cpx_dtype)

        grad = np.empty((B * C, *XYZ), dtype=self.cpx_dtype)

        for i in range(B * C // T):
            data_batched.set(image_dataf[i * T : (i + 1) * T])
            obs_batched.set(obs_dataf[i * T : (i + 1) * T])
            self.__op(get_ptr(data_batched), get_ptr(ksp_batched))
            ksp_batched /= self.norm_factor
            ksp_batched -= obs_batched
            if self.uses_density:
                ksp_batched *= self.density
            self.__adj_op(get_ptr(ksp_batched), get_ptr(data_batched))
            data_batched /= self.norm_factor
            grad[i * T : (i + 1) * T] = data_batched.get()
        grad = grad.reshape((B, C, *XYZ))
        return grad

    def _grad_calibless_device(self, image_data, obs_data):
        """Calibrationless Gradient computation when all data is on device."""
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        ksp_batched = cp.empty((T, K), dtype=self.cpx_dtype)

        grad = cp.empty((B * C, *XYZ), dtype=self.cpx_dtype)

        for i in range(B * C // T):
            data_batched.set(image_data[i * T : (i + 1) * T])
            self.__op(get_ptr(data_batched), get_ptr(ksp_batched))
            ksp_batched /= self.norm_factor
            ksp_batched -= obs_data[i * T : (i + 1) * T]
            if self.uses_density:
                ksp_batched *= self.density
            self.__adj_op(get_ptr(ksp_batched), get_ptr(data_batched))
            grad[i * T : (i + 1) * T] = data_batched
        grad = grad.reshape((B, C, *XYZ))
        grad /= self.norm_factor
        return grad

    def _safe_squeeze(self, arr):
        """Squeeze the first two dimensions of shape of the operator."""
        if self.squeeze_dims:
            try:
                arr = arr.squeeze(axis=1)
            except ValueError:
                pass
            try:
                arr = arr.squeeze(axis=0)
            except ValueError:
                pass
        return arr

    @property
    def eps(self):
        """Return the underlying precision parameter."""
        return self.raw_op.eps

    @property
    def bsize_ksp(self):
        """Size in Bytes of the compute batch of samples."""
        return self.n_trans * self.ksp_size

    @property
    def bsize_img(self):
        """Size in Bytes of the compute batch of images."""
        return self.n_trans * self.img_size

    @property
    def img_size(self):
        """Image size in bytes."""
        return int(np.prod(self.shape) * np.dtype(self.cpx_dtype).itemsize)

    @property
    def ksp_size(self):
        """k-space size in bytes."""
        return int(self.n_samples * np.dtype(self.cpx_dtype).itemsize)

    @property
    def norm_factor(self):
        """Norm factor of the operator."""
        return np.sqrt(np.prod(self.shape) * 2 ** len(self.shape))

    def __repr__(self):
        """Return info about the MRICufiNUFFT Object."""
        return (
            "MRICufiNUFFT(\n"
            f"  shape: {self.shape}\n"
            f"  n_coils: {self.n_coils}\n"
            f"  n_samples: {self.n_samples}\n"
            f"  uses_density: {self.uses_density}\n"
            f"  uses_sense: {self.uses_sense}\n"
            f"  smaps_cached: {self.smaps_cached}\n"
            f"  eps:{self.raw_op.eps:.0e}\n"
            ")"
        )
