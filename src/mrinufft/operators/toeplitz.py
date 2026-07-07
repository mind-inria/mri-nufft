"""Toeplitz approximation of the Auto-adjoint operator for NUFFT."""

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from mrinufft._array_compat import with_numpy_cupy, get_array_module
from mrinufft._utils import proper_trajectory

from mrinufft.operators.base import FourierOperatorBase


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
