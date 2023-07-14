"""Provides Operator for MR Image processing on GPU."""

import warnings

import numpy as np

from .base import FourierOperatorBase, proper_trajectory
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
from ._cupy_kernels import sense_adj_mono

CUFINUFFT_AVAILABLE = CUPY_AVAILABLE
try:
    import cupy as cp
    from cufinufft._plan import Plan
    from cufinufft._cufinufft import _spread_interpf
except ImportError:
    CUFINUFFT_AVAILABLE = False


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
        x = cp.ascontiguousarray(self.samples[:, 0])
        y = cp.ascontiguousarray(self.samples[:, 1])
        fpts_axes = [get_ptr(y), get_ptr(x), None]
        if self.ndim == 3:
            z = cp.ascontiguousarray(self.samples[:, 2])
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
        squeeze_dim=False,
        n_trans=1,
        n_streams=1,
        **kwargs,
    ):
        if not CUPY_AVAILABLE:
            raise RuntimeError("cupy is not installed")
        if not CUFINUFFT_AVAILABLE:
            raise RuntimeError("Failed to found cufinufft binary.")

        if (n_batchs * n_coils) % n_trans != 0:
            raise ValueError("n_batchs * n_coils should be a multiple of n_transf")

        self.shape = shape
        self.n_batchs = n_batchs
        self.n_trans = n_trans
        self.squeeze_dim = squeeze_dim
        self.samples = proper_trajectory(samples, normalize="pi").astype(np.float32)
        # For now only single precision is supported
        self.dtype = self.samples.dtype
        self.n_streams = n_streams
        if is_host_array(self.samples):
            samples_d = cp.asarray(self.samples, order="F")
        elif is_cuda_array(self.samples):
            samples_d = self.samples
        else:
            raise ValueError(
                "Samples should be either a C-ordered ndarray, " "or a GPUArray."
            )

        # density compensation support
        if density is True:
            # TODO estimate the density using pipe method.
            self.density = pipe(samples_d, shape)
            self.uses_density = False
        elif is_host_array(density) or is_cuda_array(density):
            self.density = density
            if len(density) != len(samples):
                raise ValueError(
                    "Density array and samples array should " "have the same length."
                )
            self.uses_density = True
            self.density_d = cp.asarray(density)
        else:
            self.density_d = None
            self.uses_density = False
        self.smaps_cached = False
        # Smaps support
        if n_coils < 1:
            raise ValueError("n_coils should be ≥ 1")
        self.n_coils = n_coils
        self.smaps = smaps
        if smaps is not None:
            if not (is_host_array(smaps) or is_cuda_array(smaps)):
                raise ValueError(
                    "Smaps should be either a C-ordered ndarray, " "or a GPUArray."
                )
            if smaps_cached:
                if verbose:
                    warnings.warn(f"{sizeof_fmt(smaps.nbytes)} will be used on gpu.")
                self._smaps_d = cp.array(smaps, order="C", copy=False)
                self.smaps_cached = True
            else:
                # allocate device memory
                self._smap_d = cp.empty((self.n_streams, *shape), dtype=self.cpx_dtype)
                self._smaps_pinned = pin_memory(smaps)
                self.smaps = self._smaps_pinned
        # Initialise NUFFT plans
        self.persist_plan = persist_plan
        self.raw_op = RawCufinufftPlan(
            samples_d,
            tuple(shape),
            n_trans=n_trans,
            **kwargs,
        )

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

        if self.uses_sense:
            ret = self._op_sense(data, ksp_d)
        else:  # calibrationless or monocoil cases
            ret = self._op_calibless(data, ksp_d)

        if not self.persist_plan:
            self.raw_op._destroy_plan(2)

        ret /= self.norm_factor
        return self._safe_squeeze(ret)

    def _op_sense(self, data, ksp_d=None):
        img_d = cp.asarray(data, dtype=self.cpx_dtype)
        coil_img_d = cp.empty(self.shape, dtype=self.cpx_dtype)
        if self.n_streams == 1:
            streams = [cp.cuda.Stream(null=True)]
        else:
            streams = [cp.cuda.Stream(non_blocking=True) for _ in range(self.n_streams)]
        for cur_stream_id in range(self.n_streams - 1):
            self._smap_d[cur_stream_id].set(
                self._smaps[cur_stream_id], streams[cur_stream_id]
            )
        if self.n_streams == 1:
            cur_stream_id = 0
        if is_host_array(data):
            ksp_d = cp.empty((self.n_batchs, self.n_samples), dtype=self.cpx_dtype)
            ksp = np.zeros(
                (self.n_batchs, self.n_coils, self.n_samples), dtype=self.cpx_dtype
            )
            for i in range(self.n_coils):
                cp.copyto(coil_img_d, img_d[i])
                if self.smaps_cached:
                    coil_img_d *= self._smaps_d[i]  # sense forward
                else:
                    self._smap_d.set(self._smaps[i])
                    coil_img_d *= self._smap_d  # sense forward
                self.__op(coil_img_d, ksp_d)
                cp.asnumpy(ksp_d, out=ksp[:, i])
            return ksp
        # data is already on device
        ksp_d = ksp_d or cp.empty((self.n_coils, self.n_samples), dtype=self.cpx_dtype)
        for i in range(self.n_coils):
            next_stream_id = (cur_stream_id + 1) % self.n_streams
            if self.smaps_cached:
                coil_img_d = img_d * self._smaps_d[i]  # sense forward
            else:
                if i < self.n_coils - self.n_streams + 1:
                    self._smap_d[next_stream_id].set(
                        self._smaps[i + self.n_streams - 1], streams[next_stream_id]
                    )
                streams[cur_stream_id].synchronize()
                coil_img_d = img_d * self._smap_d[cur_stream_id]  # sense forward
            self.__op(get_ptr(coil_img_d), get_ptr(ksp_d) + i * self.ksp_size)
            cur_stream_id = next_stream_id
        return ksp_d

    def _op_calibless(self, data, ksp_d=None):
        bsize_samples2gpu = self.n_trans * self.ksp_size
        bsize_img2gpu = self.n_trans * self.img_size
        if is_cuda_array(data):
            if ksp_d is None:
                ksp_d = cp.empty(
                    (self.n_batchs, self.n_coils, self.n_samples), dtype=self.cpx_dtype
                )
            for i in range((self.n_batchs * self.n_coils) // self.n_trans):
                self.__op(
                    get_ptr(data) + i * bsize_img2gpu,
                    get_ptr(ksp_d) + i * bsize_samples2gpu,
                )
            return ksp_d
        # calibrationless, data on host
        coil_img_d = cp.empty(np.prod(self.shape) * self.n_trans, dtype=self.cpx_dtype)
        ksp_d = cp.empty(self.n_trans * self.n_samples, dtype=self.cpx_dtype)
        ksp = np.zeros(
            (self.n_batchs * self.n_coils, self.n_samples), dtype=self.cpx_dtype
        )
        # TODO: Add concurrency compute batch n while copying batch n+1 to device
        # and batch n-1 to host
        dataf = data.flatten()
        size_batch = self.n_trans * np.prod(self.shape)
        for i in range((self.n_batchs * self.n_coils) // self.n_trans):
            coil_img_d.set(dataf[i * size_batch : (i + 1) * size_batch])
            self.__op(get_ptr(coil_img_d), get_ptr(ksp_d))
            ksp[i * self.n_trans : (i + 1) * self.n_trans] = ksp_d.get()
            ksp = ksp.reshape((self.n_batchs, self.n_coils, self.n_samples))
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
        if self.uses_sense:
            ret = self._adj_op_sense(coeffs, img_d)
        # calibrationless
        else:
            ret = self._adj_op_calibless(coeffs, img_d)

        if not self.persist_plan:
            self.raw_op._destroy_plan(1)

        ret /= self.norm_factor
        return self._safe_squeeze(ret)

    def _adj_op_sense(self, coeffs, img_d=None):
        coil_img_d = cp.empty(self.shape, dtype=self.cpx_dtype)
        if self.n_streams == 1:
            streams = [cp.cuda.Stream(null=True)]
        else:
            streams = [cp.cuda.Stream(non_blocking=True) for _ in range(self.n_streams)]
        if img_d is None:
            img_d = cp.zeros(self.shape, dtype=self.cpx_dtype)
        if not is_host_array(coeffs) and self.uses_density:
            # If k-space is already on device, we can quickly do DC
            coeffs *= self.density_d[None, :]
        else:
            # TODO: FIX streams here too
            coil_ksp_d = cp.empty(
                (self.n_streams, self.n_samples), dtype=self.cpx_dtype
            )
            for cur_stream_id in range(self.n_streams - 1):
                self._smap_d[cur_stream_id].set(
                    self._smaps[cur_stream_id], streams[cur_stream_id]
                )
                coil_ksp_d[cur_stream_id].set(
                    coeffs[cur_stream_id], streams[cur_stream_id]
                )
            if self.n_streams == 1:
                cur_stream_id = 0
            for i in range(self.n_coils):
                next_stream_id = (cur_stream_id + 1) % self.n_streams
                with streams[cur_stream_id]:
                    if i < self.n_coils - self.n_streams + 1:
                        coil_ksp_d[next_stream_id].set(
                            coeffs[i + self.n_streams - 1], streams[next_stream_id]
                        )
                    streams[cur_stream_id].synchronize()
                    if self.uses_density:
                        coil_ksp_d[cur_stream_id] *= self.density_d
                    self.__adj_op(get_ptr(coil_ksp_d), get_ptr(coil_img_d))
                    if self.smaps_cached:
                        sense_adj_mono(img_d, coil_img_d, self._smaps_d[i])
                    else:
                        if i < self.n_coils - self.n_streams + 1:
                            self._smap_d[next_stream_id].set(
                                self._smaps[i + self.n_streams - 1],
                                streams[cur_stream_id],
                            )
                        sense_adj_mono(img_d, coil_img_d, self._smap_d)
                    cur_stream_id = next_stream_id
            return img_d.get()

        for cur_stream_id in range(self.n_streams - 1):
            self._smap_d[cur_stream_id].set(
                self._smaps[cur_stream_id], streams[cur_stream_id]
            )
        if self.n_streams == 1:
            cur_stream_id = 0
        for i in range(self.n_coils):
            next_stream_id = (cur_stream_id + 1) % self.n_streams
            with streams[cur_stream_id]:
                self.__adj_op(
                    get_ptr(coeffs) + i * self.ksp_size,
                    get_ptr(coil_img_d[cur_stream_id]),
                )
                if self.smaps_cached:
                    sense_adj_mono(img_d, coil_img_d, self._smaps_d[i])
                else:
                    if i < self.n_coils - self.n_streams + 1:
                        self._smap_d[next_stream_id].set(
                            self._smaps[i + self.n_streams - 1], streams[next_stream_id]
                        )
                    streams[cur_stream_id].synchronize()
                    sense_adj_mono(img_d, coil_img_d, self._smap_d)
                cur_stream_id = next_stream_id
        if is_cuda_array(coeffs):
            return img_d
        return img_d.get()

    def _adj_op_calibless(self, coeffs, img_d=None):
        coeffs_f = coeffs.flatten()
        n_trans_samples = self.n_trans * self.n_samples
        ksp_batched = cp.empty(n_trans_samples, dtype=self.cpx_dtype)
        if self.uses_density:
            density_batched = cp.repeat(
                self.density_d[None, :], self.n_trans, axis=0
            ).flatten()

        if is_cuda_array(coeffs_f):
            img_d = img_d or cp.empty(
                (self.n_batchs, self.n_coils, *self.shape), dtype=self.cpx_dtype
            )
            for i in range((self.n_coils * self.n_batchs) // self.n_trans):
                if self.uses_density:
                    cp.copyto(
                        ksp_batched,
                        coeffs_f[i * self.bsize_ksp : (i + 1) * self.bsize_ksp],
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
        if self.n_coils == 1:
            return self._data_consistency_mono(image_data, obs_data)
        if self.uses_sense:
            return self._data_consistency_sense(image_data, obs_data)
        return self._data_consistency_calibless(image_data, obs_data)

    def _data_consistency_mono(self, image_data, obs_data):
        img_d = cp.array(image_data, copy=True)
        obs_d = cp.asarray(obs_data)
        ksp_d = cp.empty(self.n_samples, dtype=self.cpx_dtype)
        self.__op(img_d, ksp_d)
        ksp_d -= obs_d
        self.__adj_op(ksp_d, img_d)
        if is_cuda_array(image_data):
            return img_d
        return img_d.get()

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
                    coil_img_d *= self._smaps_d[i]
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
            self.__op(get_ptr(coil_img_d), get_ptr(coil_ksp_d))
            coil_ksp_d -= obs_data[i]
            if self.uses_density:
                coil_ksp_d *= self.density_d
            self.__adj_op(get_ptr(coil_ksp_d), get_ptr(coil_img_d))
            if self.smaps_cached:
                sense_adj_mono(img_d, coil_img_d, self._smaps_d[i])
            else:
                sense_adj_mono(img_d, coil_img_d, self._smap_d)
        return img_d

    def _data_consistency_calibless(self, image_data, obs_data):
        if is_cuda_array(image_data):
            img_d = cp.empty((self.n_coils, *self.shape), dtype=self.cpx_dtype)
            ksp_d = cp.empty(self.n_samples, dtype=self.cpx_dtype)
            for i in range(self.n_coils):
                self.__op(get_ptr(image_data) + i * self.img_size, get_ptr(ksp_d))
                ksp_d -= obs_data[i]
                if self.uses_density:
                    ksp_d *= self.density_d
                self.__adj_op(get_ptr(ksp_d), get_ptr(img_d) + i * self.img_size)
            return img_d

        img_d = cp.empty(self.shape, dtype=self.cpx_dtype)
        img = np.zeros((self.n_coils, *self.shape), dtype=self.cpx_dtype)
        ksp_d = cp.empty(self.n_samples, dtype=self.cpx_dtype)
        obs_d = cp.empty(self.n_samples, dtype=self.cpx_dtype)
        for i in range(self.n_coils):
            img_d.set(image_data[i])
            obs_d.set(obs_data[i])
            self.__op(get_ptr(img_d), get_ptr(ksp_d))
            ksp_d -= obs_d
            if self.uses_density:
                ksp_d *= self.density_d
            self.__adj_op(get_ptr(ksp_d), get_ptr(img_d))
            cp.asnumpy(img_d, out=img[i])
        return img

    def _safe_squeeze(self, arr):
        """Squeeze the shape of the operator."""
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
            f"  uses_sense: {self._uses_sense}\n"
            f"  smaps_cached: {self.smaps_cached}\n"
            f"  eps:{self.raw_op.eps:.0e}\n"
            ")"
        )


def _get_samples_ptr(samples):
    x, y, z = None, None, None
    x = cp.ascontiguousarray(samples[:, 0])
    y = cp.ascontiguousarray(samples[:, 1])
    fpts_axes = [get_ptr(y), get_ptr(x), None]
    if self.ndim == 3:
        z = cp.ascontiguousarray(samples[:, 2])
        fpts_axes.insert(0, get_ptr(z))
    return fpts_axes[:3], x.size


def _convert_shape_to_3D(shape, dim):
    return shape[::-1] + (1,) * (3 - dim)


def _do_spread_interp(samples, c, f, tol=1e-4, type=1):
    fpts_axes, n_samples = get_kx_ky_kz_pointers(samples)
    shape = convert_shape_to_3D(f.shape, samples.shape[-1])
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


def pipe(kspace, grid_shape, num_iter=10):
    """Estimate density compensation weight using the Pipe method.

    Parameters
    ----------
    kspace: array_like
        array of shape (M, 2) or (M, 3) containing the coordinates of the points.
    grid_shape: tuple
        shape of the image grid.
    num_iter: int, optional
        number of iterations.

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

    kspace = proper_trajectory(kspace)
    if is_host_array(kspace):
        kspace = cp.array(kspace.copy(order="F"))
    image = cp.empty(grid_shape, dtype=np.complex64)
    update = cp.empty_like(density)
    for _ in range(num_iter):
        do_spread_interp(kspace, density, image, type=1)
        do_spread_interp(kspace, update, image, type=2)
        update_density(density, update)
    return density.real
