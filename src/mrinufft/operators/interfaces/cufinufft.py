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
from ._cupy_kernels import sense_adj_mono, update_density

CUFINUFFT_AVAILABLE = CUPY_AVAILABLE
try:
    import cupy as cp
    from cufinufft._plan import Plan
    from cufinufft._cufinufft import _spread_interpf, NufftOpts, _default_opts
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


def repr_opts(self):
    """Get the value of the struct, like a dict."""
    ret = "Struct(\n"
    for fieldname, _ in self._fields_:
        ret += f"{fieldname}: {getattr(self, fieldname)},\n"
    ret += ")"
    return ret


def str_opts(self):
    """Get the value of the struct, with their meaning."""
    ret = "Struct(\n"
    for fieldname, _ in self._fields_:
        ret += f"{fieldname}: {getattr(self, fieldname)}"
        decode = OPTS_FIELD_DECODE.get(fieldname)
        if decode:
            ret += f" [{decode[getattr(self, fieldname)]}]"
        ret += "\n"
    ret += ")"
    return ret


if CUFINUFFT_AVAILABLE:
    NufftOpts.__repr__ = lambda self: repr_opts(self)
    NufftOpts.__str__ = lambda self: str_opts(self)


def get_default_opts(nufft_type, dim):
    """
    Generate a cufinufft opt struct of the dtype coresponding to plan.

    Parameters
    ----------
    finufft_type: int
        Finufft Type (`1` or `2`)
    dim: int
        Number of dimension (1, 2, 3)

    Returns
    -------
    nufft_opts structure.
    """
    nufft_opts = NufftOpts()

    ier = _default_opts(nufft_type, dim, nufft_opts)
    _error_check(ier, "Configuration not yet implemented.")

    return nufft_opts


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
        persist_plan=True,
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
            self.density = pipe(self.samples, shape)
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

        # Initialise NUFFT plans
        self.persist_plan = persist_plan

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
        if not self.persist_plan or self.raw_op.plans[2] is None:
            self.raw_op._make_plan(2)
            self.raw_op._set_pts(2)

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
        if not self.persist_plan:
            self.raw_op._destroy_plan(2)

        ret /= self.norm_factor
        return self._safe_squeeze(ret)

    def _op_sense_device(self, data, ksp_d=None):
        # FIXME: add batch support.
        ksp_d = ksp_d or cp.empty((self.n_coils, self.n_samples), dtype=self.cpx_dtype)
        img_d = cp.asarray(data, dtype=self.cpx_dtype)
        coil_img_d = cp.empty(self.shape, dtype=self.cpx_dtype)
        for i in range(self.n_coils):
            if self.smaps_cached:
                coil_img_d = img_d * self.smaps[i]  # sense forward
            else:
                self._smap_d.set(self.smaps[i])
                coil_img_d = img_d * self._smap_d[i]  # sense forward
            self.__op(get_ptr(coil_img_d), get_ptr(ksp_d) + i * self.ksp_size)
        return ksp_d

    def _op_sense_host(self, data, ksp=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        dataf = data.reshape((B, *XYZ))
        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        ksp = ksp or np.empty((B, C, K), dtype=self.cpx_dtype)
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
        if is_cuda_array(data):
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

        ksp = np.zeros((B * C, K), dtype=self.cpx_dtype)
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
        if not self.persist_plan or self.raw_op.plans[1] is None:
            self.raw_op._make_plan(1)
            self.raw_op._set_pts(1)

        # Dispatch to special case.
        if self.uses_sense and is_cuda_array(coeffs):
            adj_op_func = self._adj_op_sense_device
        elif self.uses_sense:
            adj_op_func = self._adj_op_sense_host
        else:
            adj_op_func = self._adj_op_calibless

        ret = adj_op_func(coeffs, img_d)
        ret /= self.norm_factor
        if not self.persist_plan:
            self.raw_op._destroy_plan(1)
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
        - coil_img(S, T, 1, *XYZ)
        - ksp_batch(B, 1, *XYZ)
        - smaps_batched(S, T, *XYZ)
        - density_batched(T, K)

        """
        # Define short name
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        # Allocate memory
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        if img_d is None:
            img_d = cp.zeros((B, *XYZ), dtype=self.cpx_dtype)

        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        coeffs_f = coeffs.flatten()
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

    def _adj_op_calibless(self, coeffs, img_d=None):
        coeffs_f = coeffs.flatten()
        n_trans_samples = self.n_trans * self.n_samples
        ksp_batched = cp.empty(n_trans_samples, dtype=self.cpx_dtype)
        if self.uses_density:
            density_batched = cp.repeat(
                self.density[None, :], self.n_trans, axis=0
            ).flatten()

        if is_cuda_array(coeffs_f):
            img_d = img_d or cp.empty(
                (self.n_batchs, self.n_coils, *self.shape), dtype=self.cpx_dtype
            )
            for i in range((self.n_coils * self.n_batchs) // self.n_trans):
                if self.uses_density:
                    cp.copyto(
                        ksp_batched,
                        coeffs_f[i * n_trans_samples : (i + 1) * n_trans_samples],
                    )
                    ksp_batched *= density_batched
                    self.__adj_op(
                        get_ptr(ksp_batched), get_ptr(img_d) + i * self.bsize_img
                    )
                else:
                    self.__adj_op(
                        get_ptr(coeffs_f) + i * self.bsize_ksp,
                        get_ptr(img_d) + i * self.bsize_img,
                    )
            return img_d
        # calibrationless, data on host
        img = np.zeros(
            (self.n_batchs * self.n_coils, *self.shape), dtype=self.cpx_dtype
        )
        img_batched = cp.empty((self.n_trans, *self.shape), dtype=self.cpx_dtype)
        # TODO: Add concurrency compute batch n while copying batch n+1 to device
        # and batch n-1 to host
        for i in range((self.n_batchs * self.n_coils) // self.n_trans):
            ksp_batched.set(coeffs_f[i * n_trans_samples : (i + 1) * n_trans_samples])
            if self.uses_density:
                ksp_batched *= density_batched
            self.__adj_op(get_ptr(ksp_batched), get_ptr(img_batched))
            img[i * self.n_trans : (i + 1) * self.n_trans] = img_batched.get()
        img = img.reshape((self.n_batchs, self.n_coils, *self.shape))
        return img

    @nvtx_mark()
    def __adj_op(self, coeffs_d, image_d):
        if not isinstance(coeffs_d, int):
            ret = self.raw_op.type1(get_ptr(coeffs_d), get_ptr(image_d))
        else:
            ret = self.raw_op.type1(coeffs_d, image_d)
        return ret

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
        check_size(obs_data, (self.n_batchs, self.n_coils, self.n_samples))
        if self.uses_sense:
            check_size(image_data, (self.n_batchs, *self.shape))
        else:
            check_size(image_data, (self.n_batchs, self.n_coils, *self.shape))

        if not self.persist_plan or self.raw_op.plans[1] is None:
            self.raw_op._make_plan(1)
            self.raw_op._set_pts(1)

        if self.uses_sense:
            dc_func = self._data_consistency_sense
        else:
            dc_func = self._data_consistency_calibless
        ret = dc_func(image_data, obs_data)
        if not self.persist_plan:
            self.raw_op._destroy_plan(1)
        return self._safe_squeeze(ret)

    def _data_consistency_sense(self, image_data, obs_data):
        img_d = cp.array(image_data, copy=True)
        coil_img_d = cp.empty(self.shape, dtype=self.cpx_dtype)
        coil_ksp_d = cp.empty(self.n_samples, dtype=self.cpx_dtype)
        if is_host_array(obs_data):
            coil_obs_data = cp.empty(self.n_samples, dtype=self.cpx_dtype)
            obs_data_pinned = pin_memory(obs_data)
            for i in range(self.n_coils):
                cp.copyto(coil_img_d, img_d)
                if self.smaps_cached:
                    coil_img_d *= self.smaps[i]
                else:
                    self._smap_d.set(self._smaps[i])
                    coil_img_d *= self._smap_d
                self.__op(get_ptr(coil_img_d), get_ptr(coil_ksp_d))
                coil_obs_data = cp.asarray(obs_data_pinned[i])
                coil_ksp_d -= coil_obs_data
                if self.uses_density:
                    coil_ksp_d *= self.density_d
                self.__adj_op(get_ptr(coil_ksp_d), get_ptr(coil_img_d))
                if self.smaps_cached:
                    sense_adj_mono(img_d, coil_img_d, self.smaps[i])
                else:
                    sense_adj_mono(img_d, coil_img_d, self._smap_d)
            del obs_data_pinned
            return img_d.get()

        for i in range(self.n_coils):
            cp.copyto(coil_img_d, img_d)
            if self.smaps_cached:
                coil_img_d *= self.smaps[i]
            else:
                self._smap_d.set(self._smaps[i])
                coil_img_d *= self._smap_d
            self.__op(get_ptr(coil_img_d), get_ptr(coil_ksp_d))
            coil_ksp_d -= obs_data[i]
            if self.uses_density:
                coil_ksp_d *= self.density_d
            self.__adj_op(get_ptr(coil_ksp_d), get_ptr(coil_img_d))
            if self.smaps_cached:
                sense_adj_mono(img_d, coil_img_d, self.smaps[i])
            else:
                sense_adj_mono(img_d, coil_img_d, self._smap_d)
        return img_d

    def _data_consistency_calibless(self, image_data, obs_data):
        if is_cuda_array(image_data):
            img_d = cp.empty((self.n_coils, *self.shape), dtype=self.cpx_dtype)
            ksp_d = cp.empty(self.n_samples, dtype=self.cpx_dtype)
            for i in range(self.n_coils):
                self.__op(get_ptr(image_data) + i * self.img_size, get_ptr(ksp_d))
                ksp_d /= self.norm_factor
                ksp_d -= obs_data[i]
                if self.uses_density:
                    ksp_d *= self.density_d
                self.__adj_op(get_ptr(ksp_d), get_ptr(img_d) + i * self.img_size)
            return img_d / self.norm_factor

        img_d = cp.empty(self.shape, dtype=self.cpx_dtype)
        img = np.zeros((self.n_coils, *self.shape), dtype=self.cpx_dtype)
        ksp_d = cp.empty(self.n_samples, dtype=self.cpx_dtype)
        obs_d = cp.empty(self.n_samples, dtype=self.cpx_dtype)
        for i in range(self.n_coils):
            img_d.set(image_data[i])
            obs_d.set(obs_data[i])
            self.__op(get_ptr(img_d), get_ptr(ksp_d))
            ksp_d /= self.norm_factor
            ksp_d -= obs_d
            if self.uses_density:
                ksp_d *= self.density_d
            self.__adj_op(get_ptr(ksp_d), get_ptr(img_d))
            cp.asnumpy(img_d, out=img[i])
        return img / self.norm_factor

    def _safe_squeeze(self, arr):
        """Squeeze the first two dimensions of shape of the operator."""
        if self.squeeze_dim:
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


def _get_samples_ptr(samples):
    x, y, z = None, None, None
    x = cp.ascontiguousarray(samples[:, 0])
    y = cp.ascontiguousarray(samples[:, 1])
    fpts_axes = [get_ptr(y), get_ptr(x), None]
    if samples.shape[1] == 3:
        z = cp.ascontiguousarray(samples[:, 2])
        fpts_axes.insert(0, get_ptr(z))
    return fpts_axes[:3], x.size


def _convert_shape_to_3D(shape, dim):
    return shape[::-1] + (1,) * (3 - dim)


def _do_spread_interp(samples, c, f, tol=1e-4, type=1):
    fpts_axes, n_samples = _get_samples_ptr(samples)
    shape = _convert_shape_to_3D(f.shape, samples.shape[-1])
    opts = get_default_opts(type, samples.shape[-1])
    _spread_interpf(
        type,
        samples.shape[-1],
        *shape,
        n_samples,
        *fpts_axes,
        get_ptr(c),
        get_ptr(f),
        opts,
        tol,
    )


def pipe(kspace, grid_shape, num_iter=10, tol=2e-7):
    """Estimate density compensation weight using the Pipe method.

    Parameters
    ----------
    kspace: array_like
        array of shape (M, 2) or (M, 3) containing the coordinates of the points.
    grid_shape: tuple
        shape of the image grid.
    num_iter: int, optional
        number of iterations.
    tol: float, optional
        tolerance of the density estimation.

    Returns
    -------
    density: array_like
        array of shape (M,) containing the density compensation weights.
    """
    if not CUFINUFFT_AVAILABLE:
        raise ImportError(
            "cuFINUFFT library not available to do Pipe density estimation"
        )
    import cupy as cp

    kspace = proper_trajectory(kspace, normalize="pi").astype(np.float32)
    if is_host_array(kspace):
        kspace = cp.array(kspace, order="F")
    image = cp.empty(grid_shape, dtype=np.complex64)
    density = cp.ones(kspace.shape[0], dtype=np.complex64)
    update = cp.empty_like(density)
    for _ in range(num_iter):
        _do_spread_interp(kspace, density, image, type=1, tol=tol)
        _do_spread_interp(kspace, update, image, type=2, tol=tol)
        update_density(density, update)
    return density.real
