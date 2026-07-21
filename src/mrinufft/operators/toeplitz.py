"""Toeplitz approximation of the Auto-adjoint operator for NUFFT."""

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from mrinufft._array_compat import (
    with_numpy_cupy,
    get_array_module,
    is_cuda_array,
    is_host_array,
    CUPY_AVAILABLE,
)
from mrinufft._utils import proper_trajectory

from mrinufft.operators.base import FourierOperatorBase
from mrinufft.operators.gpu_utils import _coil_combine_kernel

if CUPY_AVAILABLE:
    import cupy as cp


def _get_fftn(xp):
    if xp.__name__ == "numpy":
        from scipy.fft import fftn, ifftn
    elif xp.__name__ == "cupy":
        from cupyx.scipy.fft import fftn, ifftn
    else:  # fallback for torch and others
        fftn = xp.fft.fftn
        ifftn = xp.fft.ifftn
    return fftn, ifftn


def compute_toeplitz_kernel(
    nufft: FourierOperatorBase, weights: NDArray | None = None
) -> NDArray:
    """
    Compute the Toeplitz kernel for the NUFFT operator.

    Parameters
    ----------
    nufft : FourierOperatorBase
        The NUFFT operator instance.
    weights : NDArray | None, optional
        Weights to apply during the adjoint operation. If None, uses the
        density compensation from the NUFFT operator if available, otherwise
        uses uniform weights. Default is None.

    Returns
    -------
    NDArray
        The computed Toeplitz kernel in the frequency domain.

    See Also
    --------
    apply_toeplitz_kernel: Apply the Toeplitz kernel to an image.
    """
    xp = get_array_module(nufft.samples)
    fftn, _ = _get_fftn(xp)
    backup_density = None
    if nufft.density is not None:
        backup_density = nufft.density.copy()
        nufft.density = None
        backup_density_method = nufft._density_method
        nufft._density_method = None
    if weights is None and backup_density is not None:
        weights = xp.astype(backup_density, dtype=nufft.cpx_dtype)
    elif weights is None:
        weights = xp.ones(nufft.n_samples, dtype=nufft.cpx_dtype)

    weights = weights.astype(dtype=nufft.cpx_dtype, copy=False)

    if nufft.ndim == 2:
        kernel = _compute_toep_2d(nufft, weights)
    elif nufft.ndim == 3:
        kernel = _compute_toep_3d(nufft, weights)
    else:
        raise ValueError(
            f"Toeplitz kernel calculation not implemented for ndim={nufft.ndim}"
        )

    # set back density
    if backup_density is not None:
        nufft.density = backup_density
        nufft._density_method = backup_density_method
    scale = 1 / nufft.norm_factor
    # T = A^H W A is Hermitian for real weights, so its circulant spectrum (the
    # frequency-domain kernel) is real. Keeping only the real part halves the
    # kernel memory (dominant in 3D, where it is 2**d the image size) and is
    # equivalent to enforcing exact Hermitian symmetry of the image-domain kernel.
    return fftn(kernel * scale, overwrite_x=True, norm="ortho").real


def _modulated_weights(
    weights: NDArray, omega: NDArray, shifts: tuple[int, ...], cpx_dtype
) -> NDArray:
    """Fourier-shift the adjoint's lag window by ``shifts`` pixels.

    Modulating the weights by ``exp(i * omega . shifts)`` slides the ``N``-wide
    window of lags produced by a single ``_adj_op`` onto the ``[N/2, N)`` outer
    lags, so the full ``2N-1`` support of the Toeplitz kernel is recovered at
    ``N``-resolution (no extra memory versus a plain adjoint).
    """
    xp = get_array_module(weights)
    shift_vec = xp.asarray(shifts, dtype=omega.dtype)
    phase = xp.exp(1j * (omega @ shift_vec))
    return (weights * phase).astype(cpx_dtype, copy=False)


def _check_even_shape(shape: tuple[int, ...]) -> None:
    if any(s % 2 for s in shape):
        raise ValueError(
            f"Toeplitz kernel computation only supports even grid sizes, got {shape}."
        )


def _compute_toep_2d(nufft: FourierOperatorBase, weights: NDArray) -> NDArray:
    xp = get_array_module(nufft.samples)
    _check_even_shape(nufft.shape)
    N0, N1 = nufft.shape
    h0, h1 = N0 // 2, N1 // 2
    omega = proper_trajectory(nufft.samples, normalize="pi")

    kernel = xp.zeros((2 * N0, 2 * N1), dtype=nufft.cpx_dtype)
    tmp = xp.empty(nufft.shape, dtype=nufft.cpx_dtype)

    # Two adjoints cover the positive-x half of the kernel: the y-window is shifted
    # by +h1 (inner lags) then -h1 (outer lags). Column/row index N is the circulant
    # "don't-care" slot (never touched by the zero-padded apply), so we skip it.
    nufft._adj_op(_modulated_weights(weights, omega, (h0, h1), nufft.cpx_dtype), tmp)
    kernel[:N0, :N1] = tmp
    nufft._adj_op(_modulated_weights(weights, omega, (h0, -h1), nufft.cpx_dtype), tmp)
    kernel[:N0, N1 + 1 :] = tmp[:, 1:]

    # negative-x half by Hermitian symmetry (c[-d] = conj(c[d]) for real weights)
    ksym = xp.roll(xp.roll(kernel[::-1, ::-1], 1, axis=0), 1, axis=1).conj()
    kernel[N0 + 1 :, :] = ksym[N0 + 1 :, :]
    return kernel


def _compute_toep_3d(nufft: FourierOperatorBase, weights: NDArray) -> NDArray:
    xp = get_array_module(nufft.samples)
    _check_even_shape(nufft.shape)
    N0, N1, N2 = nufft.shape
    h0, h1, h2 = N0 // 2, N1 // 2, N2 // 2
    omega = proper_trajectory(nufft.samples, normalize="pi")

    kernel = xp.zeros((2 * N0, 2 * N1, 2 * N2), dtype=nufft.cpx_dtype)
    tmp = xp.empty(nufft.shape, dtype=nufft.cpx_dtype)

    # Four adjoints cover the positive-x half; the four sign combinations of the
    # (y, z) shift recover the inner/outer lags along each axis.
    def adj(shifts):
        nufft._adj_op(_modulated_weights(weights, omega, shifts, nufft.cpx_dtype), tmp)
        return tmp

    kernel[:N0, :N1, :N2] = adj((h0, h1, h2))
    kernel[:N0, :N1, N2 + 1 :] = adj((h0, h1, -h2))[:, :, 1:]
    kernel[:N0, N1 + 1 :, :N2] = adj((h0, -h1, h2))[:, 1:, :]
    kernel[:N0, N1 + 1 :, N2 + 1 :] = adj((h0, -h1, -h2))[:, 1:, 1:]

    # negative-x half by Hermitian symmetry (c[-d] = conj(c[d]) for real weights)
    ksym = xp.roll(
        xp.roll(xp.roll(kernel[::-1, ::-1, ::-1], 1, axis=0), 1, axis=1), 1, axis=2
    ).conj()
    kernel[N0 + 1 :] = ksym[N0 + 1 :]
    return kernel


def apply_toeplitz_kernel(
    image: NDArray,
    toeplitz_kernel: NDArray,
    padded_array: NDArray | None = None,
    paired_batch: bool = False,
) -> NDArray:
    """Apply the 2D or 3D Toeplitz kernel to an image using FFT.

    Parameters
    ----------
    image : NDArray
        The input 2D or 3D image to which the Toeplitz kernel will be applied.
    toeplitz_kernel : NDArray
        The 2D or 3D Toeplitz kernel in the frequency domain.
    padded_array : NDArray | None, optional
        An optional pre-allocated array for padding the image. If None,
        a new array will be created. Default is None.
    paired_batch : bool, optional
        If True, pairs the batch dimension of the image with the toeplitz_kernel.
        Default is False, ie the same kernel is applied to all images in the batch.

    Returns
    -------
    NDArray
        The result of applying the Toeplitz kernel to the image.

    See Also
    --------
    compute_toeplitz_kernel : Compute Toeplitz kernel to be used with this function.
    """
    xp = get_array_module(image)
    fftn, ifftn = _get_fftn(xp)
    img_shape = image.shape
    if image.ndim == toeplitz_kernel.ndim:
        # add extra batch dimension
        image = image[None, ...]
        batch_size = 1
    elif image.ndim - 1 == toeplitz_kernel.ndim:
        batch_size = image.shape[0]
        toeplitz_kernel = toeplitz_kernel[None, ...]
    else:
        raise ValueError("Image and toeplitz_kernel must have compatible dimensions.")

    if padded_array is None:
        padded_array = xp.zeros((batch_size, *toeplitz_kernel.shape), dtype=image.dtype)
    else:
        if batch_size == 1 and padded_array.ndim != image.ndim:
            # expand padded_array to have batch dimension
            padded_array = padded_array[None, :]
        elif padded_array.shape != toeplitz_kernel.shape:
            raise ValueError("padded_array shape must match toeplitz_kernel shape.")
        # padded_array is caller-provided scratch memory (e.g. cp.empty, reused
        # across calls): the padding region is not guaranteed to be zero, so it
        # must be cleared here or the "zero-padding" invariant the FFT
        # convolution relies on is silently broken.
        padded_array[...] = 0

    tl_corner = tuple(slice(0, s) for s in image.shape)

    # FIXME cannot unpack slice directly in python 3.10
    padded_array[tl_corner] = image
    axis = tuple(range(1, padded_array.ndim))
    tmp = fftn(padded_array, axes=axis, overwrite_x=True)
    tmp *= toeplitz_kernel
    result = ifftn(tmp, overwrite_x=True, axes=axis)

    return result[tl_corner].reshape(img_shape)


class _GramOpGpuMixin(FourierOperatorBase):
    """Batched (``n_trans``-chunked) Toeplitz Gram operator for GPU backends.

    Chunks the ``(batch, coil)`` loop by ``n_trans`` (``T``) instead of
    processing one image/coil at a time, so the smaps transfer and the Toeplitz
    FFT apply both run as fewer, larger batched calls instead of many small
    ones. ``n_trans`` must divide ``n_coils`` (validated in the host class'
    ``__init__``), so a chunk's ``T`` coils never straddle a batch boundary.

    Host classes must provide: ``n_trans``, ``n_batchs``, ``n_coils``,
    ``shape``, ``cpx_dtype``, ``uses_sense``, ``smaps``, ``smaps_cached``,
    ``_toeplitz_kernel``/``compute_toeplitz_kernel``, ``check_shape``,
    ``_safe_squeeze``.
    """

    def _coil_slice(self, i):
        """Return the coil ``slice`` for chunk ``i`` of the ``(B*C)//T`` loop.

        ``n_trans`` (``T``) is required to divide ``n_coils`` (``C``), so a
        chunk's ``T`` coils are always a contiguous, non-wrapping range
        within a single batch: a plain slice into ``self.smaps`` is a
        zero-copy view (unlike fancy/advanced indexing with an int array,
        which always allocates and copies).
        """
        start = (i * self.n_trans) % self.n_coils
        return slice(start, start + self.n_trans)

    def _accumulate_coil_combine(
        self,
        img_d: NDArray,
        i: int,
        data_batched: NDArray,
        smaps_batched: NDArray,
    ):
        """``img_d[b] += sum_t data_batched[t] * conj(smaps_batched[t])``.

        A single custom kernel: conj, multiply and the reduction over the
        ``T`` coils all happen in one pass over memory, with no
        intermediate arrays and no extra kernel launches.

        Since ``n_trans`` is required to divide ``n_coils``, a chunk never
        straddles a batch boundary: the batch index for every one of the
        ``T`` coils in chunk ``i`` is the same single value
        ``(i*T)//n_coils``, computed on the host as a plain Python int (no
        device array/upload needed at all).
        """
        T = data_batched.shape[0]
        vol = data_batched[0].size
        b = (i * T) // self.n_coils
        _coil_combine_kernel(data_batched, smaps_batched, b, T, vol, img_d, size=vol)

    def _gram_op_raw_device(self, in_d, out_d, padded_array=None):
        """Apply the Toeplitz Gram operator on device to a (batched) image."""
        from mrinufft.operators.toeplitz import apply_toeplitz_kernel

        cp.copyto(
            out_d,
            apply_toeplitz_kernel(in_d, self._toeplitz_kernel, padded_array),
        )
        return out_d

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
            if is_host_array(self._toeplitz_kernel):
                self._toeplitz_kernel = cp.asarray(self._toeplitz_kernel)
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
        img = img_d.get().reshape((B, 1, *XYZ))
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
