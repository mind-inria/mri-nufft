"""Toeplitz approximation of the Auto-adjoint operator for NUFFT."""

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from mrinufft._array_compat import with_numpy_cupy, get_array_module

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
    fftn, ifftn = _get_fftn(xp)
    backup_density = None
    if nufft.uses_density:
        backup_density = nufft.density.copy()
        nufft.density = None
        backup_density_method = nufft._density_method
        nufft._density_method = None
    if weights is None and backup_density is not None:
        weights = xp.astype(backup_density, dtype=nufft.cpx_dtype)
    elif weights is None:
        weights = xp.ones(nufft.n_samples, dtype=nufft.cpx_dtype)

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
    return fftn(kernel * scale, overwrite_x=True, norm="ortho")


def _compute_toep_2d(nufft: FourierOperatorBase, weights: NDArray) -> NDArray:
    xp = get_array_module(nufft.samples)

    # Trajectory Flipping (Y-axis)
    samples_orig = nufft.samples.copy()
    samples_flip_y = samples_orig.copy()
    samples_flip_y[:, 1] *= -1

    kernel = xp.zeros(tuple(s * 2 for s in nufft.shape), dtype=nufft.cpx_dtype)
    tmp = xp.empty(nufft.shape, dtype=nufft.cpx_dtype)

    # compute first two quadrants
    nufft._adj_op(weights, tmp)
    kernel[: nufft.shape[0], : nufft.shape[1]] = tmp
    kernel[nufft.shape[0] + 1 :, : nufft.shape[1]] = tmp[:0:-1, :].conj()

    # flip samples and compute other two quadrants
    nufft.update_samples(samples_flip_y, unsafe=True)
    nufft._adj_op(weights, tmp)
    kernel[: nufft.shape[0], nufft.shape[1] + 1 :] = tmp[:, :0:-1]
    kernel[nufft.shape[0] + 1 :, nufft.shape[1] + 1 :] = tmp[:0:-1, :0:-1].conj()

    # Restore original samples for nufft object consistency
    nufft.update_samples(samples_orig, unsafe=True)

    # Enforce strict Hermitian symmetry
    kernel = (kernel + kernel[::-1, ::-1].conj()) / 2

    # Move the PSF peak from center to [0, 0] to align with corner-padded image
    # This is more robust than trying to mess with fftshifts and off-by-one errors
    #

    pos = xp.argmax(xp.abs(kernel))
    if xp.__name__ == "cupy":
        pos = pos.get()
    peak_idx = xp.unravel_index(pos, kernel.shape)
    kernel = xp.roll(kernel, shift=[-p for p in peak_idx], axis=(0, 1))

    return kernel


def _compute_toep_3d(nufft: FourierOperatorBase, weights: NDArray) -> NDArray:
    xp = get_array_module(nufft.samples)

    # Initialize the operator (density=False to get the raw Gram kernel)

    # Prepare Trajectories for 4 octants
    # We flip Y, Z, and YZ. X is handled by Hermitian symmetry later.
    samples_orig = nufft.samples.copy()

    samples_z = samples_orig.copy()
    samples_z[:, 2] *= -1

    samples_y = samples_orig.copy()
    samples_y[:, 1] *= -1

    samples_yz = samples_orig.copy()
    samples_yz[:, 1] *= -1
    samples_yz[:, 2] *= -1

    NX, NY, NZ = nufft.shape
    kernel = xp.zeros(tuple(s * 2 for s in nufft.shape), dtype=nufft.cpx_dtype)
    tmp = xp.empty(nufft.shape, dtype=nufft.cpx_dtype)

    # 2. Compute 4 Adjoints
    nufft._adj_op(weights, tmp)  # Original
    kernel[:NX, :NY, :NZ] = tmp

    nufft.update_samples(samples_z, unsafe=True)
    nufft._adj_op(weights, tmp)  # Z-flipped
    kernel[:NX, :NY, NZ + 1 :] = tmp[:, :, :0:-1]

    nufft.update_samples(samples_y, unsafe=True)
    nufft._adj_op(weights, tmp)  # Y-flipped
    kernel[:NX, NY + 1 :, :NZ] = tmp[:, :0:-1, :]

    nufft.update_samples(samples_yz, unsafe=True)
    nufft._adj_op(weights, tmp)  # YZ-flipped
    kernel[:NX, NY + 1 :, NZ + 1 :] = tmp[:, :0:-1, :0:-1]

    nufft.update_samples(samples_orig, unsafe=True)

    # fill in the other 4 octants by Hermitian symmetry
    kernel[NX + 1 :, :, :] = kernel[NX - 1 : 0 : -1, :, :].conj()
    # Hermitify to ensure symmetry.
    kernel = (kernel + kernel[::-1, ::-1, ::-1].conj()) / 2

    # Move the PSF peak from center to [0, 0] to align with corner-padded image
    # This is more robust than trying to mess with fftshifts and off-by-one errors
    pos = xp.argmax(xp.abs(kernel))
    if xp.__name__ == "cupy":
        pos = pos.get()
    peak_idx = xp.unravel_index(pos, kernel.shape)
    kernel = xp.roll(kernel, shift=[-p for p in peak_idx], axis=(0, 1, 2))
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
    elif batch_size == 1 and padded_array.ndim != toeplitz_kernel.ndim:
        # expand padded_array to have batch dimension
        padded_array = padded_array[None, :]

    elif padded_array.shape != toeplitz_kernel.shape:
        raise ValueError("padded_array shape must match toeplitz_kernel shape.")

    tl_corner = tuple(slice(0, s) for s in image.shape)

    padded_array[tl_corner] = image
    axis = tuple(range(1, padded_array.ndim))
    tmp = fftn(padded_array, axes=axis)
    tmp *= toeplitz_kernel
    result = ifftn(tmp, overwrite_x=True, axes=axis)

    return result[tl_corner].reshape(img_shape)
