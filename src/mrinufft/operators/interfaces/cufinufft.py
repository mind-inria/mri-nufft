"""Provides Operator for MR Image processing on GPU."""

import weakref
import numpy as np
from numpy.typing import NDArray
from mrinufft.operators.base import (
    FourierOperatorBase,
    with_numpy_cupy,
    power_method,
    _ToggleGradPlanMixin,
)
from mrinufft._utils import (
    proper_trajectory,
    sizeof_fmt,
)

from mrinufft._array_compat import (
    CUPY_AVAILABLE,
    is_cuda_array,
    is_host_array,
    pin_memory,
    get_array_module,
    auto_cast,
)
from mrinufft.operators.interfaces.utils import nvtx_mark

CUFINUFFT_AVAILABLE = CUPY_AVAILABLE
try:
    import cupy as cp
    from cufinufft import Plan
    from cufinufft._compat import get_array_ptr

    _coil_combine_kernel = cp.ElementwiseKernel(
        "raw T data, raw T smaps, int64 b, int32 n_t, int64 vol",
        "raw T img",
        """
        long long off = b * vol + i;
        for (int t = 0; t < n_t; t++) {
            T d = data[t * vol + i];
            T s = smaps[t * vol + i];
            img[off] += d * T(s.real(), -s.imag());
        }
        """,
        "coil_combine_kernel",
    )

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


class RawCufinufftPlan:
    """Light wrapper around the guru interface of finufft."""

    def __init__(
        self,
        samples: NDArray,
        shape: tuple[int, ...],
        n_trans: int = 1,
        eps: float = 1e-6,
        **kwargs,
    ):
        self.shape = shape
        self.ndim = len(shape)
        self.eps = float(eps)
        self.n_trans = n_trans
        self._dtype = samples.dtype
        # the first element is dummy to index type 1 with 1
        # and type 2 with 2.
        self.plans: list[Plan | None] = [None, None, None]
        self.grad_plan = None
        for i in [1, 2]:
            self._make_plan(i, **kwargs)
            self._set_pts(i, samples)

    @property
    def dtype(self):
        """Return the dtype (precision) of the transform."""
        try:
            return self.plans[1].dtype
        except AttributeError:
            return DTYPE_R2C[str(self._dtype)]

    def _make_plan(self, typ, **kwargs):
        self.plans[typ] = Plan(
            typ,
            self.shape,
            self.n_trans,
            self.eps,
            dtype=DTYPE_R2C[str(self._dtype)],
            **kwargs,
        )
        self.plans[typ]._nk = 0  # no type3 points.

    def _set_pts(self, typ, samples: NDArray):
        plan = self.grad_plan if typ == "grad" else self.plans[typ]
        if plan is None:
            raise ValueError(f"Plan of type {typ} has not been initialized.")
        M, d = samples.shape
        if d != self.ndim:
            raise ValueError(
                f"Samples should have shape (N_samples, {self.ndim}), "
                f"got {samples.shape}."
            )
        # Samples should be F-ordered (column-major) !!
        ptr_samples = [None, None, None]
        ptr_samples[0] = get_array_ptr(samples[:, -1])
        ptr_samples[1] = get_array_ptr(samples[:, -2])
        if self.ndim == 3:
            ptr_samples[2] = get_array_ptr(samples[:, -3])

        plan._references = ptr_samples
        plan._nj = M
        plan._setpts(plan._plan, M, *ptr_samples, 0, None, None, None)

    def _destroy_plan(self, typ):
        if self.plans[typ] is not None:
            p = self.plans[typ]
            del p
            self.plans[typ] = None

    def _destroy_plan_grad(self):
        if self.grad_plan is not None:
            p = self.grad_plan
            del p
            self.grad_plan = None

    def type1(self, coeff_data, grid_data):
        """Type 1 transform. Non Uniform to Uniform."""
        return self.plans[1].execute(coeff_data, grid_data)

    def type2(self, grid_data, coeff_data):
        """Type 2 transform. Uniform to non-uniform."""
        return self.plans[2].execute(grid_data, coeff_data)

    def toggle_grad_traj(self):
        """Toggle between the gradient trajectory and the plan for type 1 transform."""
        self.plans[2], self.grad_plan = self.grad_plan, self.plans[2]


class MRICufiNUFFT(FourierOperatorBase, _ToggleGradPlanMixin):
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
    async_transfer: bool, default False
        If True, pipeline host<->device transfers with compute (double
        buffering) in the host-input code paths, using dedicated
        non-blocking CUDA streams. This overlaps H2D/D2H copies with
        cufinufft compute across batches, at the cost of extra pinned
        host buffers and device buffers (roughly 2x the per-batch memory).
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
    autograd_available = True

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
        async_transfer=False,
        **kwargs,
    ):
        # run the availaility check here to get detailled output.
        if not CUPY_AVAILABLE:
            raise RuntimeError("cupy is not installed")
        if not CUFINUFFT_AVAILABLE:
            raise RuntimeError("Failed to found cufinufft binary.")
        # set CUDA device
        gpu_device_id = kwargs.get("gpu_device_id", 0)
        try:
            cp.cuda.Device(gpu_device_id).use()
        except Exception as e:
            self.log.warning("Failed to set CUDA device %s: %s", gpu_device_id, e)
        super().__init__()
        self.shape = shape
        self.n_batchs = n_batchs
        self.n_trans = n_trans
        self.async_transfer = async_transfer
        self._host_registry = {}
        self.squeeze_dims = squeeze_dims
        self.n_coils = n_coils
        self.autograd_available = True
        self._samples = cp.array(
            proper_trajectory(samples, normalize="pi"), order="F", copy=None
        )
        self.dtype = self.samples.dtype
        # density compensation support
        if is_cuda_array(density):
            self.density = density
        else:
            self.compute_density(density)
            if is_host_array(self.density):
                self.density = cp.array(self.density)

        self.smaps_cached = smaps_cached
        self.compute_smaps(smaps)
        # Smaps support
        if self.smaps is not None and (
            not (is_host_array(self.smaps) or is_cuda_array(self.smaps))
        ):
            raise ValueError(
                "Smaps should be either a C-ordered np.ndarray, or a GPUArray."
            )

        # n_trans must tiles the number of coils and batchs
        if (n_batchs * n_coils) % n_trans != 0:
            raise ValueError(
                f"n_trans={n_trans} must divide n_batchs*n_coils={n_batchs * n_coils}"
            )
        # for optimal sense, n_trans must divide n_coils.
        if self.uses_sense:
            if self.n_coils % n_trans != 0:
                raise ValueError(
                    f"n_trans={n_trans} must divide n_coils={self.n_coils}"
                )
        self.raw_op = RawCufinufftPlan(
            self._samples,
            tuple(shape),
            n_trans=n_trans,
            **kwargs,
        )

    @FourierOperatorBase.smaps.setter
    def smaps(self, new_smaps):
        """Update smaps.

        If the number of coils is different, it is updated.

        Parameters
        ----------
        new_smaps: C-ordered ndarray or a GPUArray.
        """
        if new_smaps is not None and hasattr(self, "smaps_cached"):
            C = new_smaps.shape[0]
            XYZ = new_smaps.shape[1:]
            if XYZ != self.shape:
                raise ValueError("Smaps shape does not match image shape.")
            if C != self.n_coils:
                self.log.warning("n_coils updated via smaps.")
                self.n_coils = C
            if self.smaps_cached or is_cuda_array(new_smaps):
                self.smaps_cached = True
                self.log.warning(
                    "%s used on gpu for smaps.",
                    sizeof_fmt(new_smaps.size * np.dtype(self.cpx_dtype).itemsize),
                )
                self._smaps = cp.array(
                    new_smaps, order="C", copy=None, dtype=self.cpx_dtype
                )
            else:
                if self._smaps is None:
                    self._smaps = pin_memory(
                        new_smaps.astype(self.cpx_dtype, copy=False)
                    )
                    self._smap_d = cp.empty(self.shape, dtype=self.cpx_dtype)
                else:
                    # copy the array to pinned memory
                    np.copyto(self._smaps, new_smaps.astype(self.cpx_dtype, copy=False))
        else:
            self._smaps = new_smaps

    @nvtx_mark()
    def update_samples(self, new_samples, *, unsafe=False):
        """Update the samples of the NUFFT operator.

        Parameters
        ----------
        new_samples: np.ndarray or GPUArray
            The new samples location of shape ``Nsamples x N_dimensions``.
        unsafe: bool, default False
            If True, the original array is used directly without any checks.
            This should be used with caution as it might lead to unexpected behavior.

        Notes
        -----
        If unsafe is True, the new_samples should be of shape (Nsamples, N_dimensions),
        F-ordered (column-major) and in the range [-pi, pi]. If not, this will lead to
        unexpected behavior. You have been warned.

        If unsafe is False, this is automatically handled.
        """
        if not unsafe:
            self._samples = cp.array(
                proper_trajectory(new_samples, normalize="pi"), copy=None
            ).astype(np.float32, order="F", copy=False)
        else:
            self._samples = new_samples
        for typ in [1, 2, "grad"]:
            if typ == "grad" and not self._grad_wrt_traj:
                continue
            self.raw_op._set_pts(typ, self._samples)
        self.compute_density(self._density_method)

    @FourierOperatorBase.density.setter
    def density(self, new_density):
        """Update the density compensation."""
        if new_density is None:
            self._density = None
            return
        self._density = cp.array(new_density, copy=None)

    @with_numpy_cupy
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
        self.check_shape(image=data, ksp=ksp_d)
        data = auto_cast(data, self.cpx_dtype)
        # Dispatch to special case. The *_device variants are also called
        # directly by other operators (e.g. MRIStackedNUFFTGPU), which
        # apply their own normalization on top -- they must keep returning
        # an unnormalized result, so `op` normalizes for them here. The
        # *_host variants are only used through this wrapper, so they
        # normalize their own result in-place, on-device, before any
        # device->host copy: multiplying by the reciprocal is much faster
        # than dividing (complex/real division is poorly vectorized,
        # ~5-6x slower for the same array), and doing it on the GPU avoids
        # a slow single-threaded host-side pass over what can be a
        # multi-GiB array.
        if self.uses_sense and is_cuda_array(data):
            op_func = self._op_sense_device
            needs_norm = True
        elif self.uses_sense:
            op_func = self._op_sense_host
            needs_norm = False
        elif is_cuda_array(data):
            op_func = self._op_calibless_device
            needs_norm = True
        else:
            op_func = self._op_calibless_host
            needs_norm = False

        ret = op_func(data, ksp_d)
        if needs_norm:
            ret *= 1.0 / self.norm_factor
        return self._safe_squeeze(ret)

    def _get_async_streams(self):
        """Return (and lazily create) the dedicated H2D/D2H transfer streams.

        Compute keeps running on the default stream (where cufinufft's
        Plans already run, since a Plan's CUDA stream cannot be changed
        after construction). These streams are created non-blocking so
        they do not implicitly synchronize with the default stream,
        allowing real overlap of transfers with compute.
        """
        if not hasattr(self, "_h2d_stream"):
            self._h2d_stream = cp.cuda.Stream(non_blocking=True)
            self._d2h_stream = cp.cuda.Stream(non_blocking=True)
        return self._h2d_stream, self._d2h_stream

    def _coil_slice(self, i):
        """Return the coil ``slice`` for chunk ``i`` of the ``(B*C)//T`` loop.

        ``n_trans`` (``T``) is required to divide ``n_coils`` (``C``, see
        `__init__`), so a chunk's ``T`` coils are always a contiguous,
        non-wrapping range within a single batch: a plain slice into
        ``self.smaps`` is a zero-copy view (unlike fancy/advanced indexing
        with an int array, which always allocates and copies).
        """
        start = (i * self.n_trans) % self.n_coils
        return slice(start, start + self.n_trans)

    def _host_register(self, arr, anchor=None):
        """Page-lock ARR's memory in place (no copy), for async H2D/D2H.

        ``cudaHostRegister``/``cudaHostUnregister`` are expensive (roughly
        size-proportional: ~230ms combined for a 4GiB array), so the
        registration is cached by memory address and reused across calls
        as long as the underlying buffer is still alive, instead of paying
        that cost on every ``op``/``adj_op`` call. It is released
        automatically via a :class:`weakref.finalize` callback once
        ``anchor`` (the array whose lifetime owns that memory -- ``arr``
        itself if not given explicitly) is garbage collected.

        Returns True on success. On failure (e.g. the memory cannot be
        pinned), returns False so the caller can fall back to the
        synchronous path rather than silently doing unsafe transfers.
        """
        anchor = arr if anchor is None else anchor
        ptr = arr.ctypes.data
        key = (ptr, arr.nbytes)
        fin = self._host_registry.get(key)
        if fin is not None and fin.alive:
            return True
        try:
            cp.cuda.runtime.hostRegister(ptr, arr.nbytes, 0)
        except Exception:
            return False

        registry = self._host_registry

        def _cleanup(ptr=ptr, key=key):
            registry.pop(key, None)
            try:
                cp.cuda.runtime.hostUnregister(ptr)
            except Exception:
                pass

        self._host_registry[key] = weakref.finalize(anchor, _cleanup)
        return True

    def _register_contiguous(
        self, arr: NDArray, shape: tuple[int, ...] | None = None
    ) -> NDArray | None:
        """Page-lock a contiguous view of ``arr``, reshaped if ``shape`` given.

        Fuses making ``arr`` contiguous with `_host_register`: registration
        is keyed off the contiguous buffer's own address, anchored on
        whichever array owns that memory -- ``arr`` itself if it was
        already contiguous (no copy made), or the freshly made contiguous
        copy otherwise. Returns None (caller falls back to the synchronous
        path) if the memory cannot be page-locked.
        """
        contig = np.ascontiguousarray(arr)
        anchor = arr if contig.ctypes.data == arr.ctypes.data else contig
        if not self._host_register(contig, anchor=anchor):
            return None
        return contig.reshape(shape) if shape is not None else contig

    def _accumulate_coil_combine(self, img_d, i, data_batched, smaps_batched):
        """``img_d[b] += sum_t data_batched[t] * conj(smaps_batched[t])``.

        A single custom kernel: conj, multiply and the reduction over the
        ``T`` coils all happen in one pass over memory, with no
        intermediate arrays and no extra kernel launches -- instead of
        the previous conj -> multiply -> sum -> add chain. It's just
        another kernel queued on the current stream right after
        `_op`/`_adj_op`, so it doesn't introduce any blocking sync of
        its own.

        Since ``n_trans`` is required to divide ``n_coils`` (see
        `__init__`), a chunk never straddles a batch boundary: the batch
        index for every one of the ``T`` coils in chunk ``i`` is the same
        single value ``(i*T)//n_coils``, computed on the host as a plain
        Python int (no device array/upload needed at all).
        """
        T = data_batched.shape[0]
        vol = data_batched[0].size
        b = (i * T) // self.n_coils
        _coil_combine_kernel(data_batched, smaps_batched, b, T, vol, img_d, size=vol)

    def _op_sense_device(self, data, ksp_d=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        data = cp.asarray(data)
        image_dataf = cp.reshape(data, (B, *XYZ))
        ksp_d = ksp_d or cp.empty((B * C, K), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        for i in range((B * C) // T):
            idx_coils = self._coil_slice(i)
            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils].reshape((T, *XYZ)))
            else:
                smaps_batched = self.smaps[idx_coils].reshape((T, *XYZ))
            # A chunk's T coils share a single batch (see `_coil_slice`):
            # a single fused broadcast-multiply kernel straight from the
            # caller's image view, instead of gathering (and copying) a
            # redundant T-way repeat of it via fancy indexing.
            cp.multiply(image_dataf[(i * T) // C], smaps_batched, out=data_batched)
            self._op(data_batched, ksp_d[i * T : (i + 1) * T])

        return ksp_d.reshape((B, C, K))

    def _op_sense_host(self, data, ksp=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        if ksp is None:
            ksp = np.zeros((B, C, K), dtype=self.cpx_dtype)
        n_call = (B * C) // T

        if self.async_transfer and n_call > 1:
            ret = self._op_sense_host_async(data, ksp)
            if ret is not None:
                return ret

        ksp_flat = ksp.reshape((B * C, K))
        dataf = data.reshape((B, *XYZ))
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        # A chunk's T coils share a single batch (see `_coil_slice`), so
        # only one copy of that image needs to cross PCIe; broadcasting it
        # against the T smaps directly with `cp.multiply(..., out=...)` (a
        # single fused kernel) is much cheaper than gathering T redundant
        # host-side copies of the same image and multiplying separately.
        data_single_d = cp.empty((1, *XYZ), dtype=self.cpx_dtype)
        ksp_batched = cp.zeros((T, K), dtype=self.cpx_dtype)
        inv_norm = 1.0 / self.norm_factor
        for i in range(n_call):
            idx_coils = self._coil_slice(i)
            data_single_d[0].set(dataf[(i * T) // C])
            if not self.smaps_cached:
                coil_img_d.set(self.smaps[idx_coils])
            else:
                cp.copyto(coil_img_d, self.smaps[idx_coils])
            coil_img_d *= data_single_d
            self._op(coil_img_d, ksp_batched)
            ksp_batched *= inv_norm
            ksp_flat[i * T : (i + 1) * T] = ksp_batched.get()
        return ksp_flat.reshape((B, C, K))

    def _op_sense_host_async(self, data, ksp):
        """Pipelined (double-buffered) forward SENSE op for host data.

        Registers ``data``/``ksp`` in place (no staging copy) and issues
        H2D/compute/D2H in an overlapped double-buffered pipeline. Returns
        None (caller falls back to the synchronous loop) if the arrays
        cannot be page-locked.
        """
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        n_call = (B * C) // T

        dataf = self._register_contiguous(data, shape=(B, *XYZ))
        if dataf is None:
            return None
        ksp_flat = ksp.reshape((B * C, K))
        if not self._host_register(ksp_flat, anchor=ksp):
            return None

        h2d_stream, d2h_stream = self._get_async_streams()
        # `n_trans` divides `n_coils` (enforced in `__init__`), so every
        # chunk's T coils belong to a single, constant batch -- only one
        # copy of that image is needed, broadcast against the T smaps
        # during the multiply, instead of T redundant copies of the same
        # image transferred and stored.
        data_batched_d = [cp.empty((1, *XYZ), dtype=self.cpx_dtype) for _ in range(2)]
        coil_img_d = [cp.empty((T, *XYZ), dtype=self.cpx_dtype) for _ in range(2)]
        ksp_batched_d = [cp.empty((T, K), dtype=self.cpx_dtype) for _ in range(2)]
        h2d_done = [cp.cuda.Event() for _ in range(2)]
        compute_done = [cp.cuda.Event() for _ in range(2)]
        inv_norm = 1.0 / self.norm_factor

        def prefetch(i):
            b = i % 2
            if i >= 2:
                # buffer slot b is still being read by the compute
                # issued two iterations ago; wait before overwriting
                # the device buffer. No host-side wait is needed:
                # dataf/self.smaps are read directly (in place,
                # host-registered), not staged into a shared buffer.
                h2d_stream.wait_event(compute_done[b])
            batch_idx = (i * T) // C
            idx_coils = self._coil_slice(i)
            with h2d_stream:
                data_batched_d[b][0].set(dataf[batch_idx])
                if not self.smaps_cached:
                    coil_img_d[b].set(self.smaps[idx_coils])
                else:
                    cp.copyto(coil_img_d[b], self.smaps[idx_coils])
                coil_img_d[b] *= data_batched_d[b]
            h2d_done[b].record(h2d_stream)

        prefetch(0)
        for i in range(n_call):
            b = i % 2
            cp.cuda.get_current_stream().wait_event(h2d_done[b])
            if i + 1 < n_call:
                prefetch(i + 1)
            self._op(coil_img_d[b], ksp_batched_d[b])
            # Scale on-device (cheap, GPU-parallel) rather than the
            # returned host array afterward (single-threaded, and much
            # slower for large arrays -- see `op`/`adj_op`).
            ksp_batched_d[b] *= inv_norm
            compute_done[b].record()
            d2h_stream.wait_event(compute_done[b])
            ksp_batched_d[b].get(
                out=ksp_flat[i * T : (i + 1) * T], stream=d2h_stream, blocking=False
            )
        d2h_stream.synchronize()
        return ksp_flat.reshape((B, C, K))

    def _op_calibless_device(self, data, ksp_d=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        data = cp.asarray(data).reshape(B * C, *XYZ)
        if ksp_d is None:
            ksp_d = cp.empty((B * C, K), dtype=self.cpx_dtype)
        for i in range((B * C) // T):
            self._op(
                data[i * T : (i + 1) * T],
                ksp_d[i * T : (i + 1) * T],
            )
        return ksp_d.reshape(B, C, K)

    def _op_calibless_host(self, data, ksp=None):
        # calibrationless, data on host
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        if ksp is None:
            ksp = np.zeros((B * C, K), dtype=self.cpx_dtype)
        n_call = (B * C) // T

        if self.async_transfer and n_call > 1:
            ret = self._op_calibless_host_async(data, ksp)
            if ret is not None:
                return ret
        ksp_flat = ksp.reshape((B * C, K))
        data_ = data.reshape(B * C, *XYZ)
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        ksp_d = cp.empty((T, K), dtype=self.cpx_dtype)
        inv_norm = 1.0 / self.norm_factor
        for i in range(n_call):
            coil_img_d.set(data_[i * T : (i + 1) * T])
            self._op(coil_img_d, ksp_d)
            ksp_d *= inv_norm
            ksp_flat[i * T : (i + 1) * T] = ksp_d.get()
        return ksp_flat.reshape((B, C, K))

    def _op_calibless_host_async(self, data, ksp):
        """Pipelined (double-buffered) forward calibrationless op for host data.

        Registers ``data``/``ksp`` in place (no staging copy) and issues
        H2D/compute/D2H in an overlapped double-buffered pipeline. Returns
        None (caller falls back to the synchronous loop) if the arrays
        cannot be page-locked.
        """
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        n_call = (B * C) // T

        data_ = self._register_contiguous(data, shape=(B * C, *XYZ))
        if data_ is None:
            return None
        ksp_flat = ksp.reshape((B * C, K))
        if not self._host_register(ksp_flat, anchor=ksp):
            return None

        h2d_stream, d2h_stream = self._get_async_streams()
        coil_img_d = [cp.empty((T, *XYZ), dtype=self.cpx_dtype) for _ in range(2)]
        ksp_d = [cp.empty((T, K), dtype=self.cpx_dtype) for _ in range(2)]
        h2d_done = [cp.cuda.Event() for _ in range(2)]
        compute_done = [cp.cuda.Event() for _ in range(2)]
        inv_norm = 1.0 / self.norm_factor

        def prefetch(i):
            b = i % 2
            if i >= 2:
                # buffer slot b is still being read by the compute
                # issued two iterations ago; wait before overwriting
                # it. No host-side wait is needed: data_ is read
                # directly (in place, host-registered).
                h2d_stream.wait_event(compute_done[b])
            coil_img_d[b].set(data_[i * T : (i + 1) * T], stream=h2d_stream)
            h2d_done[b].record(h2d_stream)

        prefetch(0)
        for i in range(n_call):
            b = i % 2
            cp.cuda.get_current_stream().wait_event(h2d_done[b])
            if i + 1 < n_call:
                prefetch(i + 1)
            self._op(coil_img_d[b], ksp_d[b])
            # Scale on-device (cheap, GPU-parallel) rather than the
            # returned host array afterward (single-threaded, and much
            # slower for large arrays -- see `op`/`adj_op`).
            ksp_d[b] *= inv_norm
            compute_done[b].record()
            d2h_stream.wait_event(compute_done[b])
            ksp_d[b].get(
                out=ksp_flat[i * T : (i + 1) * T], stream=d2h_stream, blocking=False
            )
        d2h_stream.synchronize()
        return ksp_flat.reshape((B, C, K))

    @nvtx_mark()
    def _op(self, image_d, coeffs_d):
        # ensure everything is pointers before going to raw level.
        return self.raw_op.type2(image_d, coeffs_d)

    @nvtx_mark()
    @with_numpy_cupy
    def adj_op(self, coeffs, img_d=None):
        """Non Cartesian MRI adjoint operator.

        Parameters
        ----------
        coeffs: np.array or GPUArray

        Returns
        -------
        Array in the same memory space of coeffs. (ie on cpu or gpu Memory).
        """
        self.check_shape(image=img_d, ksp=coeffs)
        coeffs = auto_cast(coeffs, self.cpx_dtype)
        # See the comment in `op` for why the device variants need
        # normalizing here while the host variants normalize themselves.
        if self.uses_sense and is_cuda_array(coeffs):
            adj_op_func = self._adj_op_sense_device
            needs_norm = True
        elif self.uses_sense:
            adj_op_func = self._adj_op_sense_host
            needs_norm = False
        elif is_cuda_array(coeffs):
            adj_op_func = self._adj_op_calibless_device
            needs_norm = True
        else:
            adj_op_func = self._adj_op_calibless_host
            needs_norm = False

        ret = adj_op_func(coeffs, img_d)
        if needs_norm:
            ret *= 1.0 / self.norm_factor
        return self._safe_squeeze(ret)

    def _adj_op_sense_device(self, coeffs, img_d=None):
        """Perform sense reconstruction when data is on device."""
        # Define short name
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        coeffs = cp.asarray(coeffs).reshape(B * C, K)
        # Allocate memory
        if img_d is None:
            img_d = cp.zeros((B, *XYZ), dtype=self.cpx_dtype)
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        if self.uses_density:
            ksp_new = cp.empty((T, K), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        for i in range((B * C) // T):
            idx_coils = self._coil_slice(i)
            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils])
            else:
                smaps_batched = self.smaps[idx_coils]
            if self.uses_density:
                cp.copyto(ksp_new, coeffs[i * T : (i + 1) * T])
                ksp_new *= self.density
            else:
                ksp_new = coeffs[i * T : (i + 1) * T]
            self._adj_op(ksp_new, coil_img_d)
            self._accumulate_coil_combine(img_d, i, coil_img_d, smaps_batched)
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

        # Allocate memory
        if img_d is None:
            img_d = cp.zeros((B, *XYZ), dtype=self.cpx_dtype)
        n_call = (B * C) // T

        if self.async_transfer and n_call > 1:
            ret = self._adj_op_sense_host_async(coeffs, img_d)
            if ret is not None:
                return ret

        if self.uses_density:
            density_batched = cp.repeat(self.density[None, :], T, axis=0)
        coeffs_f = coeffs.flatten()
        coil_img_d = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        ksp_batched = cp.empty((T, K), dtype=self.cpx_dtype)
        for i in range(n_call):
            idx_coils = self._coil_slice(i)
            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils])
            else:
                smaps_batched = self.smaps[idx_coils]
            ksp_batched.set(coeffs_f[i * T * K : (i + 1) * T * K].reshape(T, K))
            if self.uses_density:
                ksp_batched *= density_batched
            self._adj_op(ksp_batched, coil_img_d)

            self._accumulate_coil_combine(img_d, i, coil_img_d, smaps_batched)
        img_d *= 1.0 / self.norm_factor
        img = img_d.get()
        img = img.reshape((B, 1, *XYZ))
        return img

    def _adj_op_sense_host_async(self, coeffs, img_d):
        """Pipelined (double-buffered) adjoint SENSE op for host data.

        Only the input (ksp + smaps) side is pipelined: the output
        accumulates in-place into the shared ``img_d`` across every coil, so
        a single readback after the loop is used instead of a per-batch D2H
        copy. Registers ``coeffs`` in place (no staging copy). Returns None
        (caller falls back to the synchronous loop) if it cannot be
        page-locked.
        """
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        n_call = (B * C) // T
        if self.uses_density:
            density_batched = cp.repeat(self.density[None, :], T, axis=0)

        coeffs_reg = self._register_contiguous(coeffs, shape=(B * C, K))
        if coeffs_reg is None:
            return None

        h2d_stream, _ = self._get_async_streams()
        ksp_batched_d = [cp.empty((T, K), dtype=self.cpx_dtype) for _ in range(2)]
        smaps_batched_d = [cp.empty((T, *XYZ), dtype=self.cpx_dtype) for _ in range(2)]
        coil_img_d = [cp.empty((T, *XYZ), dtype=self.cpx_dtype) for _ in range(2)]
        h2d_done = [cp.cuda.Event() for _ in range(2)]
        compute_done = [cp.cuda.Event() for _ in range(2)]

        def prefetch(i):
            b = i % 2
            if i >= 2:
                # buffer slot b is still being read by the compute
                # issued two iterations ago; wait before overwriting
                # the device buffer. No host-side wait is needed:
                # coeffs_reg/self.smaps are read directly (in place,
                # host-registered), not staged into a shared buffer.
                h2d_stream.wait_event(compute_done[b])
            idx_coils = self._coil_slice(i)
            with h2d_stream:
                # One copy per buffer instead of one per coil: T separate
                # `.set()`/`copyto()` calls each carry their own Python +
                # CUDA-API dispatch overhead, which adds up fast for small
                # per-coil chunks and was serializing the "prefetch" step
                # far more than the actual transferred byte count justifies.
                ksp_batched_d[b].set(coeffs_reg[i * T : (i + 1) * T])
                if not self.smaps_cached:
                    smaps_batched_d[b].set(self.smaps[idx_coils])
                else:
                    cp.copyto(smaps_batched_d[b], self.smaps[idx_coils])
                if self.uses_density:
                    ksp_batched_d[b] *= density_batched
            h2d_done[b].record(h2d_stream)

        prefetch(0)
        for i in range(n_call):
            b = i % 2
            cp.cuda.get_current_stream().wait_event(h2d_done[b])
            if i + 1 < n_call:
                prefetch(i + 1)
            self._adj_op(ksp_batched_d[b], coil_img_d[b])
            self._accumulate_coil_combine(img_d, i, coil_img_d[b], smaps_batched_d[b])
            compute_done[b].record()
        # Scale on-device (cheap, GPU-parallel) rather than the returned
        # host array afterward (single-threaded, and much slower for
        # large arrays -- see `op`/`adj_op`).
        img_d *= 1.0 / self.norm_factor
        img = img_d.get()
        img = img.reshape((B, 1, *XYZ))
        return img

    def _adj_op_calibless_device(self, coeffs, img_d=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        coeffs = cp.asarray(coeffs)
        coeffs_f = coeffs.reshape(B * C, K)
        ksp_batched = cp.empty((T, K), dtype=self.cpx_dtype)
        if self.uses_density:
            density_batched = cp.repeat(self.density[None, :], T, axis=0)
        img_d = img_d or cp.empty((B, C, *XYZ), dtype=self.cpx_dtype)
        img_d = img_d.reshape(B * C, *XYZ)
        for i in range((B * C) // T):
            if self.uses_density:
                cp.copyto(ksp_batched, coeffs_f[i * T : (i + 1) * T])
                ksp_batched *= density_batched
                self._adj_op(ksp_batched, img_d[i * T : (i + 1) * T])
            else:
                self._adj_op(
                    coeffs_f[i * T : (i + 1) * T],
                    img_d[i * T : (i + 1) * T],
                )
        return img_d.reshape(B, C, *XYZ)

    def _adj_op_calibless_host(self, coeffs, img_batched=None):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        img = np.zeros((B * C, *XYZ), dtype=self.cpx_dtype)
        n_call = (B * C) // T

        if self.async_transfer and n_call > 1:
            ret = self._adj_op_calibless_host_async(coeffs, img)
            if ret is not None:
                return ret

        coeffs_ = coeffs.reshape(B * C, K)
        if self.uses_density:
            density_batched = cp.repeat(self.density[None, :], T, axis=0)
        ksp_batched = cp.empty((T, K), dtype=self.cpx_dtype)
        if img_batched is None:
            img_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        inv_norm = 1.0 / self.norm_factor
        for i in range(n_call):
            ksp_batched.set(coeffs_[i * T : (i + 1) * T])
            if self.uses_density:
                ksp_batched *= density_batched
            self._adj_op(ksp_batched, img_batched)
            img_batched *= inv_norm
            img[i * T : (i + 1) * T] = img_batched.get()
        return img.reshape((B, C, *XYZ))

    def _adj_op_calibless_host_async(self, coeffs, img):
        """Pipelined (double-buffered) adjoint calibrationless op for host data.

        Registers ``coeffs``/``img`` in place (no staging copy) and issues
        H2D/compute/D2H in an overlapped double-buffered pipeline. Returns
        None (caller falls back to the synchronous loop) if the arrays
        cannot be page-locked.
        """
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape
        n_call = (B * C) // T
        if self.uses_density:
            density_batched = cp.repeat(self.density[None, :], T, axis=0)

        coeffs_ = self._register_contiguous(coeffs, shape=(B * C, K))
        if coeffs_ is None:
            return None
        if not self._host_register(img):
            return None

        h2d_stream, d2h_stream = self._get_async_streams()
        ksp_batched_d = [cp.empty((T, K), dtype=self.cpx_dtype) for _ in range(2)]
        img_batched_d = [cp.empty((T, *XYZ), dtype=self.cpx_dtype) for _ in range(2)]
        h2d_done = [cp.cuda.Event() for _ in range(2)]
        compute_done = [cp.cuda.Event() for _ in range(2)]
        inv_norm = 1.0 / self.norm_factor

        def prefetch(i):
            b = i % 2
            if i >= 2:
                # buffer slot b is still being read by the compute
                # issued two iterations ago; wait before overwriting
                # it. No host-side wait is needed: coeffs_ is read
                # directly (in place, host-registered).
                h2d_stream.wait_event(compute_done[b])
            with h2d_stream:
                ksp_batched_d[b].set(coeffs_[i * T : (i + 1) * T])
                if self.uses_density:
                    ksp_batched_d[b] *= density_batched
            h2d_done[b].record(h2d_stream)

        prefetch(0)
        for i in range(n_call):
            b = i % 2
            cp.cuda.get_current_stream().wait_event(h2d_done[b])
            if i + 1 < n_call:
                prefetch(i + 1)
            self._adj_op(ksp_batched_d[b], img_batched_d[b])
            # Scale on-device (cheap, GPU-parallel) rather than the
            # returned host array afterward (single-threaded, and much
            # slower for large arrays -- see `op`/`adj_op`).
            img_batched_d[b] *= inv_norm
            compute_done[b].record()
            d2h_stream.wait_event(compute_done[b])
            img_batched_d[b].get(
                out=img[i * T : (i + 1) * T], stream=d2h_stream, blocking=False
            )
        d2h_stream.synchronize()
        return img.reshape((B, C, *XYZ))

    @nvtx_mark()
    def _adj_op(self, coeffs_d, image_d):
        return self.raw_op.type1(coeffs_d, image_d)

    @with_numpy_cupy
    def gram_op(self, data, img_d=None, toeplitz=True):
        """Compute the Gram operator of the NUFFT.

        Parameters
        ----------
        data: array
            Input data array.
        img_d: array, optional
            Preallocated output array.
        toeplitz: bool, default True
            If True, use the Toeplitz method to compute the Gram operator.
            If False, use the direct method.

        Returns
        -------
        NDArray
            Array with the Gram operator applied.
        """
        self.check_shape(image=data)
        if not toeplitz:
            return self.adj_op(self.op(data))
        if self._toeplitz_kernel is None:
            self.compute_toeplitz_kernel()
        if self.uses_sense and is_cuda_array(data):
            gram_func = self._gram_op_sense_device
        elif self.uses_sense:
            gram_func = self._gram_op_sense_host
        elif is_cuda_array(data):
            gram_func = self._gram_op_calibless_device
        else:
            gram_func = self._gram_op_calibless_host
        ret = gram_func(data, img_d)
        return self._safe_squeeze(ret)

    def _gram_op_sense_host(self, data, img_d):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        XYZ = self.shape
        image_dataf = np.reshape(data, (B, *XYZ))

        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        # A chunk's T coils share a single batch (see `_coil_slice`), so
        # only one copy of that image needs to cross PCIe; broadcasting it
        # against the T smaps directly with `cp.multiply(..., out=...)` (a
        # single fused kernel) is much cheaper than gathering T redundant
        # host-side copies of the same image and multiplying separately.
        data_single_d = cp.empty((1, *XYZ), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        padded_array = cp.empty((T, *(s * 2 for s in XYZ)), dtype=self.cpx_dtype)

        img_d = cp.zeros((B, *XYZ), dtype=self.cpx_dtype)
        for i in range(B * C // T):
            idx_coils = self._coil_slice(i)
            data_single_d[0].set(image_dataf[(i * T) // C])

            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils].reshape((T, *XYZ)))
            else:
                smaps_batched = self.smaps[idx_coils].reshape((T, *XYZ))
            cp.multiply(data_single_d, smaps_batched, out=data_batched)

            self._gram_op_raw_device(data_batched, data_batched, padded_array)
            self._accumulate_coil_combine(img_d, i, data_batched, smaps_batched)
        img = img_d.get()
        img = img_d.reshape((B, 1, *XYZ))
        return img

    def _gram_op_sense_device(self, data, img_d):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        XYZ = self.shape

        image_data = cp.asarray(data)
        image_dataf = cp.reshape(image_data, (B, *XYZ))
        img_d = cp.zeros((B, *XYZ), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        padded_array = cp.empty((T, *(s * 2 for s in XYZ)), dtype=self.cpx_dtype)
        for i in range(B * C // T):
            idx_coils = self._coil_slice(i)
            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils].reshape((T, *XYZ)))
            else:
                smaps_batched = self.smaps[idx_coils].reshape((T, *XYZ))
            # A chunk's T coils share a single batch (see `_coil_slice`):
            # a single fused broadcast-multiply kernel straight from the
            # caller's image view, instead of gathering (and copying) a
            # redundant T-way repeat of it via fancy indexing.
            cp.multiply(image_dataf[(i * T) // C], smaps_batched, out=data_batched)
            self._gram_op_raw_device(data_batched, data_batched, padded_array)

            self._accumulate_coil_combine(img_d, i, data_batched, smaps_batched)
        img_d = img_d.reshape((B, 1, *XYZ))
        return img_d

    def _gram_op_calibless_host(self, data, img_d):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        XYZ = self.shape
        image_dataf = np.reshape(data, (B * C, *XYZ))
        if img_d is None:
            img_d = np.empty((B * C, *XYZ), dtype=self.cpx_dtype)
        else:
            img_d = img_d.reshape((B * C, *XYZ))
        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        padded_array = cp.empty((T, *(s * 2 for s in XYZ)), dtype=self.cpx_dtype)
        for i in range(B * C // T):
            data_batched.set(image_dataf[i * T : (i + 1) * T])
            self._gram_op_raw_device(data_batched, data_batched, padded_array)
            img_d[i * T : (i + 1) * T] = data_batched.get()
        img_d = img_d.reshape((B, C, *XYZ))
        return img_d

    def _gram_op_calibless_device(self, data, img_d):
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        XYZ = self.shape

        image_data = cp.asarray(data).reshape(B * C, *XYZ)
        padded_array = cp.empty((T, *(s * 2 for s in XYZ)), dtype=self.cpx_dtype)

        if img_d is None:
            img_d = cp.empty((B * C, *XYZ), dtype=self.cpx_dtype)
        else:
            img_d = img_d.reshape((B * C, *XYZ))

        for i in range(B * C // T):
            self._gram_op_raw_device(
                image_data[i * T : (i + 1) * T],
                img_d[i * T : (i + 1) * T],
                padded_array,
            )
        img_d = img_d.reshape((B, C, *XYZ))
        return img_d

    def _gram_op_raw_device(self, in_d, out_d, padded_array=None):
        """Apply the toeplitz Gram operator on device on a single image."""
        # TODO Add support for batching with n_trans.
        from mrinufft.operators.toeplitz import apply_toeplitz_kernel

        cp.copyto(
            out_d, apply_toeplitz_kernel(in_d, self._toeplitz_kernel, padded_array)
        )
        return out_d

    def data_consistency(self, image_data, obs_data):
        """Compute the data consistency estimation directly on gpu.

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
        xp = get_array_module(image_data)
        if xp.__name__ == "torch" and image_data.is_cpu:
            image_data = image_data.numpy()
        xp = get_array_module(obs_data)
        if xp.__name__ == "torch" and obs_data.is_cpu:
            obs_data = obs_data.numpy()
        obs_data = auto_cast(obs_data, self.cpx_dtype)
        image_data = auto_cast(image_data, self.cpx_dtype)

        self.check_shape(image=image_data, ksp=obs_data)

        if self.uses_sense and is_host_array(image_data):
            grad_func = self._dc_sense_host
        elif self.uses_sense and is_cuda_array(image_data):
            grad_func = self._dc_sense_device
        elif not self.uses_sense and is_host_array(image_data):
            grad_func = self._dc_calibless_host
        elif not self.uses_sense and is_cuda_array(image_data):
            grad_func = self._dc_calibless_device
        else:
            raise ValueError("No suitable gradient function found.")
        ret = grad_func(image_data, obs_data)

        ret = self._safe_squeeze(ret)
        if xp.__name__ == "torch" and is_cuda_array(ret):
            ret = xp.as_tensor(ret, device=image_data.device)
        elif xp.__name__ == "torch":
            ret = xp.from_numpy(ret)
        return ret

    def _dc_sense_host(self, image_data, obs_data):
        """Gradient computation when all data is on host."""
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        image_dataf = np.reshape(image_data, (B, *XYZ))
        obs_dataf = np.reshape(obs_data, (B * C, K))

        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        # A chunk's T coils share a single batch (see `_coil_slice`), so
        # only one copy of that image needs to cross PCIe; broadcasting it
        # against the T smaps directly with `cp.multiply(..., out=...)` (a
        # single fused kernel) is much cheaper than gathering T redundant
        # host-side copies of the same image and multiplying separately.
        data_single_d = cp.empty((1, *XYZ), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)

        ksp_batched = cp.empty((T, K), dtype=self.cpx_dtype)
        obs_batched = cp.empty((T, K), dtype=self.cpx_dtype)

        grad_d = cp.zeros((B, *XYZ), dtype=self.cpx_dtype)
        grad = np.empty((B, *XYZ), dtype=self.cpx_dtype)
        inv_norm = 1.0 / self.norm_factor
        for i in range(B * C // T):
            idx_coils = self._coil_slice(i)
            data_single_d[0].set(image_dataf[(i * T) // C])
            obs_batched.set(obs_dataf[i * T : (i + 1) * T])

            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils].reshape((T, *XYZ)))
            else:
                smaps_batched = self.smaps[idx_coils].reshape((T, *XYZ))
            cp.multiply(data_single_d, smaps_batched, out=data_batched)
            self._op(data_batched, ksp_batched)

            ksp_batched *= inv_norm
            ksp_batched -= obs_batched

            if self.uses_density:
                ksp_batched *= self.density
            self._adj_op(ksp_batched, data_batched)

            self._accumulate_coil_combine(grad_d, i, data_batched, smaps_batched)
        grad_d *= inv_norm
        grad = grad_d.get()
        grad = grad.reshape((B, 1, *XYZ))
        return grad

    def _dc_sense_device(self, image_data, obs_data):
        """Gradient computation when all data is on device."""
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        image_data = cp.asarray(image_data)
        obs_data = cp.asarray(obs_data)
        image_dataf = cp.reshape(image_data, (B, *XYZ))
        obs_dataf = cp.reshape(obs_data, (B * C, K))
        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        smaps_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        ksp_batched = cp.empty((T, K), dtype=self.cpx_dtype)
        grad = cp.zeros((B, *XYZ), dtype=self.cpx_dtype)
        inv_norm = 1.0 / self.norm_factor

        for i in range(B * C // T):
            idx_coils = self._coil_slice(i)
            if not self.smaps_cached:
                smaps_batched.set(self.smaps[idx_coils].reshape((T, *XYZ)))
            else:
                smaps_batched = self.smaps[idx_coils].reshape((T, *XYZ))
            # A chunk's T coils share a single batch (see `_coil_slice`):
            # a single fused broadcast-multiply kernel straight from the
            # caller's image view, instead of gathering (and copying) a
            # redundant T-way repeat of it via fancy indexing.
            cp.multiply(image_dataf[(i * T) // C], smaps_batched, out=data_batched)
            self._op(data_batched, ksp_batched)
            ksp_batched *= inv_norm
            ksp_batched -= obs_dataf[i * T : (i + 1) * T]

            if self.uses_density:
                ksp_batched *= self.density
            self._adj_op(ksp_batched, data_batched)

            self._accumulate_coil_combine(grad, i, data_batched, smaps_batched)
        grad = grad.reshape((B, 1, *XYZ))
        grad *= inv_norm
        return grad

    def _dc_calibless_host(self, image_data, obs_data):
        """Calibrationless Gradient computation when all data is on host."""
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        image_dataf = np.reshape(image_data, (B * C, *XYZ))
        obs_dataf = np.reshape(obs_data, (B * C, K))

        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)

        ksp_batched = cp.empty((T, K), dtype=self.cpx_dtype)
        obs_batched = cp.empty((T, K), dtype=self.cpx_dtype)

        grad = np.empty((B * C, *XYZ), dtype=self.cpx_dtype)
        inv_norm = 1.0 / self.norm_factor

        for i in range(B * C // T):
            data_batched.set(image_dataf[i * T : (i + 1) * T])
            obs_batched.set(obs_dataf[i * T : (i + 1) * T])
            self._op(data_batched, ksp_batched)
            ksp_batched *= inv_norm
            ksp_batched -= obs_batched
            if self.uses_density:
                ksp_batched *= self.density
            self._adj_op(ksp_batched, data_batched)
            data_batched *= inv_norm
            grad[i * T : (i + 1) * T] = data_batched.get()
        grad = grad.reshape((B, C, *XYZ))
        return grad

    def _dc_calibless_device(self, image_data, obs_data):
        """Calibrationless Gradient computation when all data is on device."""
        T, B, C = self.n_trans, self.n_batchs, self.n_coils
        K, XYZ = self.n_samples, self.shape

        image_data = cp.asarray(image_data).reshape(B * C, *XYZ)
        obs_data = cp.asarray(obs_data).reshape(B * C, K)

        data_batched = cp.empty((T, *XYZ), dtype=self.cpx_dtype)
        ksp_batched = cp.empty((T, K), dtype=self.cpx_dtype)

        grad = cp.empty((B * C, *XYZ), dtype=self.cpx_dtype)
        inv_norm = 1.0 / self.norm_factor

        for i in range(B * C // T):
            cp.copyto(data_batched, image_data[i * T : (i + 1) * T])
            self._op(data_batched, ksp_batched)
            ksp_batched *= inv_norm
            ksp_batched -= obs_data[i * T : (i + 1) * T]
            if self.uses_density:
                ksp_batched *= self.density
            self._adj_op(ksp_batched, data_batched)
            grad[i * T : (i + 1) * T] = data_batched
        grad = grad.reshape((B, C, *XYZ))
        grad *= inv_norm
        return grad

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
            f"  n_trans: {self.n_trans}\n"
            f"  n_batchs: {self.n_batchs}\n"
            f"  uses_density: {self.uses_density}\n"
            f"  uses_sense: {self.uses_sense}\n"
            f"  smaps_cached: {self.smaps_cached}\n"
            f"  async_transfer: {self.async_transfer}\n"
            f"  eps:{self.raw_op.eps:.0e}\n"
            ")"
        )

    def _make_plan_grad(self, **kwargs):
        self.raw_op.grad_plan = Plan(
            2,
            self.shape,
            self.n_trans,
            self.raw_op.eps,
            dtype=DTYPE_R2C[str(self.samples.dtype)],
            isign=1,
            **kwargs,
        )
        self.raw_op._set_pts("grad", self._samples)

    def get_lipschitz_cst(self, max_iter=10, **kwargs):
        """Return the Lipschitz constant of the operator.

        Parameters
        ----------
        max_iter: int
            Number of iteration to perform to estimate the Lipschitz constant.
        kwargs:
            Extra kwargs for the cufinufft operator.

        Returns
        -------
        float
            Lipschitz constant of the operator.
        """
        # Disable coil dimension for faster computation
        n_coils = self.n_coils
        n_batchs = self.n_batchs
        smaps = self.smaps
        squeeze_dims = self.squeeze_dims

        self.smaps = None
        self.n_coils = 1
        self.n_batchs = 1
        self.squeeze_dims = True

        x = 1j * np.random.random(self.shape).astype(self.cpx_dtype, copy=False)
        x += np.random.random(self.shape).astype(self.cpx_dtype, copy=False)

        x = cp.asarray(x)
        lipschitz_cst, _ = power_method(
            max_iter, self, norm_func=lambda x: cp.linalg.norm(x.flatten()), x=x
        )

        # restore coil setup
        self.n_coils = n_coils
        self.n_batchs = n_batchs
        self.smaps = smaps
        self.squeeze_dims = squeeze_dims

        return lipschitz_cst

    @classmethod
    def pipe(
        cls,
        kspace_loc,
        volume_shape,
        max_iter=10,
        osf=2,
        normalize=True,
        **kwargs,
    ):
        """Compute the density compensation weights for a given set of kspace locations.

        Parameters
        ----------
        kspace_loc: np.ndarray
            the kspace locations
        volume_shape: np.ndarray
            the volume shape
        max_iter: int default 10
            the number of iterations for density estimation
        osf: float or int
            The oversampling factor the volume shape
        normalize: bool
            Whether to normalize the density compensation.
            We normalize such that the energy of PSF = 1
        """
        if CUFINUFFT_AVAILABLE is False:
            raise ValueError(
                "cufinufft is not available, cannot estimate the density compensation"
            )
        grid_op = cls(
            samples=kspace_loc,
            shape=volume_shape,
            upsampfac=osf,
            gpu_spreadinterponly=1,
            gpu_kerevalmeth=0,
            **kwargs,
        )
        density_comp = cp.ones(kspace_loc.shape[0], dtype=grid_op.cpx_dtype)
        for _ in range(max_iter):
            density_comp /= cp.abs(
                grid_op.op(
                    grid_op.adj_op(density_comp.astype(grid_op.cpx_dtype))
                ).squeeze()
            )
        if normalize:
            test_op = cls(samples=kspace_loc, shape=volume_shape, **kwargs)
            test_im = cp.ones(volume_shape, dtype=test_op.cpx_dtype)
            test_im_recon = test_op.adj_op(density_comp * test_op.op(test_im))
            density_comp /= cp.mean(cp.abs(test_im_recon))
        return abs(density_comp.squeeze())
