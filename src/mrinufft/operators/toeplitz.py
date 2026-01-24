"""Toeplitz approximation of the Auto-adjoint operator for NUFFT."""

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from mrinufft._array_compat import with_numpy_cupy, get_array_module

from mrinufft.operators.base import FourierOperatorBase


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
    backup_density = None
    if nufft.uses_density:
        backup_density = nufft.density.copy()
        nufft.density = None
        backup_density_method = nufft._density_method
        nufft._density_method = None
    if weights is None and backup_density is not None:
        weights = backup_density
    elif weights is None:
        xp = get_array_module(nufft.samples)
        weights = xp.ones(nufft.n_samples, dtype=nufft.dtype)

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

    return kernel


def _compute_toep_2d(nufft: FourierOperatorBase, weights: NDArray) -> NDArray:
    xp = get_array_module(nufft.samples)

    # Trajectory Flipping (Y-axis)
    samples_orig = nufft.samples.copy()
    samples_flip_y = samples_orig.copy()
    samples_flip_y[:, 1] *= -1

    k1 = nufft.adj_op(weights)
    nufft.update_samples(samples_flip_y, unsafe=True)
    k2 = nufft.adj_op(weights)

    # Restore original samples for nufft object consistency
    nufft.update_samples(samples_orig, unsafe=True)

    # Assemble Spatial Kernel
    zero_y = xp.zeros((k1.shape[0], 1), dtype=k1.dtype)
    ky = xp.concatenate([k1, zero_y, k2[:, :0:-1]], axis=1)

    zero_x = xp.zeros((1, ky.shape[1]), dtype=ky.dtype)
    kernel = xp.concatenate([ky, zero_x, ky[:0:-1, :].conj()], axis=0)

    # Enforce strict Hermitian symmetry
    kernel = (kernel + kernel[::-1, ::-1].conj()) / 2

    # Move the PSF peak from center to [0, 0] to align with corner-padded image
    # This is more robust than trying to mess with fftshifts and off-by-one errors
    #
    peak_idx = xp.unravel_index(xp.argmax(xp.abs(kernel)), kernel.shape)
    kernel = xp.roll(kernel, shift=[-p for p in peak_idx], axis=(0, 1))

    # osf is fixed to 2.0
    # TODO: Generalize to other osf values by cropping / padding ?
    # what is the relation with the NUFFT osf grid size ?
    grid_size = xp.prod(xp.array(nufft.shape) * 2)
    scale = xp.prod(xp.array(kernel.shape)) / grid_size
    return xp.fft.fftn(kernel * scale)


def _compute_toep_3d(nufft: FourierOperatorBase, weights: NDArray) -> NDArray:
    xp = get_array_module(nufft.samples)

    # Initialize the operator (density=False to get the raw Gram kernel)

    # 1. Prepare Trajectories for 4 octants
    # We flip Y, Z, and YZ. X is handled by Hermitian symmetry later.
    samples_orig = nufft.samples.copy()

    samples_z = samples_orig.copy()
    samples_z[:, 2] *= -1

    samples_y = samples_orig.copy()
    samples_y[:, 1] *= -1

    samples_yz = samples_orig.copy()
    samples_yz[:, 1] *= -1
    samples_yz[:, 2] *= -1

    # 2. Compute 4 Adjoints
    k00 = nufft.adj_op(weights)  # Original

    nufft.update_samples(samples_z, unsafe=True)
    k01 = nufft.adj_op(weights)  # Z-flipped

    nufft.update_samples(samples_y, unsafe=True)
    k10 = nufft.adj_op(weights)  # Y-flipped

    nufft.update_samples(samples_yz, unsafe=True)
    k11 = nufft.adj_op(weights)  # YZ-flipped

    nufft.update_samples(samples_orig, unsafe=True)

    # 3. Assemble Z-axis (axis 2)
    zero_z = xp.zeros((*k00.shape[:2], 1), dtype=k00.dtype)
    kz0 = xp.concatenate([k00, zero_z, k01[:, :, :0:-1]], axis=2)
    kz1 = xp.concatenate([k10, zero_z, k11[:, :, :0:-1]], axis=2)

    # 4. Assemble Y-axis (axis 1)
    # flip and conjugate the second half of the Y-assembly relative to Z
    # but since these are all forward lags, the kyz assembly follows the 2D logic
    zero_y = xp.zeros((kz0.shape[0], 1, kz0.shape[2]), dtype=kz0.dtype)
    kyz = xp.concatenate([kz0, zero_y, kz1[:, :0:-1, :]], axis=1)

    # 5. Assemble X-axis using Hermitian symmetry (axis 0)
    zero_x = xp.zeros((1, *kyz.shape[1:]), dtype=kyz.dtype)
    # Reflect and conjugate across all dimensions to fill the negative X lags
    kernel = xp.concatenate([kyz, zero_x, kyz[:0:-1, :, :].conj()], axis=0)

    # 6. Hermitify to ensure strict symmetry
    kernel = (kernel + kernel[::-1, ::-1, ::-1].conj()) / 2

    # Move the PSF peak from center to [0, 0] to align with corner-padded image
    # This is more robust than trying to mess with fftshifts and off-by-one errors
    pos = xp.argmax(xp.abs(kernel))
    if xp.__name__ == "cupy":
        pos = pos.get()
    peak_idx = xp.unravel_index(pos, kernel.shape)
    kernel = xp.roll(kernel, shift=[-p for p in peak_idx], axis=(0, 1, 2))

    return xp.fft.fftn(kernel)


def apply_toeplitz_kernel(
    image: NDArray,
    toeplitz_kernel: NDArray,
    padded_array: NDArray | None = None,
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

    Returns
    -------
    NDArray
        The result of applying the Toeplitz kernel to the image.

    See Also
    --------
    compute_toeplitz_kernel : Compute Toeplitz kernel to be used with this function.
    """
    xp = get_array_module(image)
    if xp.__name__ == "numpy":
        from scipy.fft import fftn, ifftn
    elif xp.__name__ == "cupy":
        from cupy.cupyx.scipy.fft import fftn, ifftn
    else:  # fallback for torch and others
        fftn = xp.fft.fftn
        ifftn = xp.fft.ifftn
    if padded_array is None:
        padded_array = xp.zeros(toeplitz_kernel.shape, dtype=image.dtype)
    elif padded_array.shape != toeplitz_kernel.shape:
        raise ValueError("padded_array shape must match toeplitz_kernel shape.")

    tl_corner = tuple(slice(0, s) for s in image.shape)

    padded_array[tl_corner] = image
    tmp = fftn(padded_array)
    tmp *= toeplitz_kernel
    result = ifftn(tmp, overwrite_x=True)

    return result[tl_corner]
