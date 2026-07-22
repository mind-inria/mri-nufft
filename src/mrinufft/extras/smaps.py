"""Smaps module for sensitivity maps estimation.

.. autoregistry:: smaps

"""

from __future__ import annotations

import gc
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from mrinufft._array_compat import get_array_module, with_numpy_cupy
from mrinufft._utils import MethodRegister, _fill_doc
from mrinufft.density.utils import flat_traj

from .cartesian import fft, ifft

########################################################
# Estimation of Off-Resonance-Correction Interpolators #
########################################################

_smap_docs = dict(
    base_params="""
Parameters
----------
traj : numpy.ndarray
    The trajectory of the samples.
shape : tuple
    The shape of the image.
kspace_data : numpy.ndarray
    The k-space data.
threshold : float, or tuple of float, optional
    The threshold used for extracting the k-space center.
    By default it is 0.1
backend : str
    The backend used for the operator.
density : numpy.ndarray, optional
    The density compensation weights.
max_iter : int, optional
    The max iterations for internal `pinv` computations
""",
    returns="""
Returns
-------
Smaps : numpy.ndarray
    The sensitivity maps.
""",
    espirit_pisco_params="""calib_width : int or tuple of int, default 24.
    The calibration region width. 
decim : int, default 1.
    The decimation factor for the calculation of sensitivity maps.
    This can be used to speed up the computation and
    significantly reduce memory usage. The final result is upsampled back
    to the original size through periodic sinc interpolation.
kernel_size : int or tuple of int, default 7.
    Side length of the k-space calibration kernel along each dimension
    (for an ellipsoidal kernel, the ellipsoid inscribed in this box is
    used). 
kernel_shape: "ellipsoid" or "rect", default "ellipsoid".
    The shape of the k-space calibration kernel. "ellipsoid" drops the corners of the
    rectangular support for reduced cost at negligible accuracy cost.
thresh : float, default 0.05
    Relative threshold (w.r.t. the largest singular value) on the
    calibration matrix's singular values used to select its subspace:
    for ESPIRiT, singular vectors *above* `thresh` are kept (signal
    subspace); for PISCO, singular vectors *below* `thresh` are kept
    (nullspace). Raising `thresh` therefore widens the retained subspace
    for PISCO but narrows it for ESPIRiT.
crop : float, default 0.08
    Threshold on the gap between the local coil-covariance matrix's
    extreme (normalized) eigenvalue at each voxel x and its ideal value
    within the image support -- the largest eigenvalue and its ideal
    value of 1 for ESPIRiT, the smallest eigenvalue and its ideal value
    of 0 for PISCO: x is considered outside the support (and masked out)
    when this gap exceeds `crop`.
power_iter : int, default 30
    Number of (batched) power iterations used to extract the min or max 
    eigenvector of the local coil-covariance matrices.
""",
    espirit_pisco_ref="""
References
----------
    Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS,
    Lustig M. ESPIRiT--an eigenvalue approach to autocalibrating parallel
    MRI: where SENSE meets GRAPPA. Magn Reson Med.
    2014 Mar;71(3):990-1001. doi: 10.1002/mrm.24751.
    PMID: 23649942; PMCID: PMC4142121.
    
    R. A. Lobos, C.-C. Chan, J. P. Haldar. New Theory and Faster
    Computations for Subspace-Based Sensitivity Map Estimation in
    Multichannel MRI. IEEE Transactions on Medical Imaging
    43:286-296, 2024. doi: 10.1109/TMI.2023.3299599.

    MATLAB implementation https://github.com/ralobos/PISCO
""",
)


def _cosine_window(distance_ratio, kind, xp):
    r"""Hann/Hamming raised-cosine window.

    .. math::
        w = a_0 + (1-a_0) \cos(\pi \cdot \text{distance\_ratio})

    with :math:`a_0=0.5` for "hann"/"hanning", and the equiripple-optimal
    :math:`a_0=0.53836` for "hamming".

    Parameters
    ----------
    distance_ratio : NDArray
        Distance to the window center, normalized so that it is 0 at the
        center and 1 at the edge (e.g. a k-space sample's radius divided by
        the window radius, or a grid index's distance from the center bin
        divided by the half-width).
    kind : {"hann", "hanning", "hamming"}
        Which window to compute.
    xp : module
        Array module (``numpy`` or ``cupy``) to compute with.

    Returns
    -------
    NDArray
        The window weights, same shape as `distance_ratio`.
    """
    a_0 = 0.5 if kind in ("hann", "hanning") else 0.53836
    return a_0 + (1 - a_0) * xp.cos(xp.pi * distance_ratio)


def _extract_kspace_center(
    kspace_data: NDArray,
    kspace_loc: NDArray,
    threshold: float | NDArray | None = None,
    density: NDArray | None = None,
    window_fun: str | Callable[[NDArray], NDArray] = "ellipse",
) -> tuple[NDArray, NDArray, NDArray | None]:
    r"""Extract k-space center and corresponding sampling locations.

    The extracted center of the k-space, i.e. both the kspace locations and
    kspace values. If the density compensators are passed, the corresponding
    compensators for the center of k-space data will also be returned. The
    return dtypes for density compensation and kspace data is same as input

    Parameters
    ----------
    kspace_data: numpy.ndarray
        The value of the samples
    kspace_loc: numpy.ndarray
        The samples location in the k-space domain (between [-0.5, 0.5[)
    threshold: tuple or float
        The threshold used to extract the k_space center (between (0, 1])
    window_fun: "Hann", "Hanning", "Hamming", or a callable, default None.
        The window function to apply to the selected data. It is computed with
        the center locations selected. Only works with circular mask.
        If window_fun is a callable, it takes as input the array (n_samples x n_dims)
        of sample positions and returns an array of n_samples weights to be
        applied to the selected k-space values, before the smaps estimation.

    Returns
    -------
    data_thresholded: ndarray
        The k-space values in the center region.
    center_loc: ndarray
        The locations in the center region.
    density_comp: ndarray, optional
        The density compensation weights (if requested)

    Notes
    -----
    The Hann (or Hanning) and Hamming windows  of width :math:`2\theta` are defined as:
    .. math::

        w(x,y) = a_0 + (1-a_0) * \cos(\pi * \sqrt{x^2+y^2}/\theta),
        \sqrt{x^2+y^2} \le \theta

    In the case of Hann window :math:`a_0=0.5`.
    For Hamming window we consider the optimal value in the equiripple sense:
    :math:`a_0=0.53836`.
    """
    xp = get_array_module(kspace_data)
    if isinstance(threshold, float):
        threshold = (threshold,) * kspace_loc.shape[1]

    threshold = xp.asarray(threshold, dtype=kspace_loc.dtype)

    if window_fun == "rect":
        condition = (
            xp.sum((xp.abs(kspace_loc) - threshold) <= 0, axis=1) == kspace_loc.shape[1]
        )
        center_locations = kspace_loc[condition, :]
        data_thresholded = kspace_data[..., condition]
        if density is not None:
            dc = density[condition]
        else:
            dc = None
        return xp.ascontiguousarray(data_thresholded), center_locations, dc
    else:
        if callable(window_fun):
            window = window_fun(kspace_loc)
        else:
            if window_fun in ["hann", "hanning", "hamming"]:
                radius = xp.linalg.norm(kspace_loc, axis=1)
                window = _cosine_window(radius / threshold[0], window_fun, xp)
            elif window_fun == "ellipse":
                window = xp.sum(kspace_loc**2 / xp.asarray(threshold) ** 2, axis=1) <= 1
            else:
                raise ValueError("Unsupported window function.")
            if xp != np:
                window = xp.asarray(window)
        data_thresholded = window * kspace_data
        # Return k-space locations & density just for consistency
        return data_thresholded, kspace_loc, density


register_smaps = MethodRegister("smaps", docstring_subs=_smap_docs)

get_smaps: Callable[[str], Callable[..., NDArray]] = register_smaps.make_getter()


def _crop_or_pad(arr, target_shape, mode="constant", constant_values=0):
    """
    Crop or pad a NumPy/CuPy array to the target shape (centered).

    Parameters
    ----------
    arr : np.ndarray or cupy.ndarray
        Input array.
    target_shape : tuple of int
        Desired output shape.
    mode : str, optional
        Padding mode (same as np.pad / cupy.pad). Default is 'constant'.
    constant_values : scalar, optional
        Used if mode='constant'.

    Returns
    -------
    out : np.ndarray or cupy.ndarray
        Cropped/padded array.
    """
    xp = get_array_module(arr)
    in_shape = arr.shape
    pad_width = []
    slices = []

    for _, (s, t) in enumerate(zip(in_shape, target_shape)):
        diff = t - s
        pad_before = max(diff, 0) // 2
        pad_after = max(diff, 0) - pad_before
        crop_before = max(-diff, 0) // 2
        crop_after = crop_before + min(s, t)

        pad_width.append((pad_before, pad_after))
        slices.append(slice(crop_before, crop_after))

    arr = arr[tuple(slices)]
    if any(pw != (0, 0) for pw in pad_width):
        arr = xp.pad(arr, pad_width, mode=mode, constant_values=constant_values)
    return arr


@register_smaps
@with_numpy_cupy
@flat_traj
def low_frequency(
    traj: NDArray,
    shape: tuple[int, ...],
    kspace_data: NDArray,
    backend: str,
    threshold: float | tuple[float, ...] = 0.1,
    density: NDArray | None = None,
    max_iter: int = 10,
    window_fun: str | Callable[[NDArray], NDArray] = "ellipse",
    blurr_factor: int | float | tuple[float, ...] = 0.0,
    mask: bool | NDArray = False,
) -> NDArray:
    """
    Calculate low-frequency sensitivity maps.

    ${base_params}
    window_fun: "Hann", "Hanning", "Hamming", or a callable, default None.
        The window function to apply to the selected data. It is computed with
        the center locations selected. Only works with circular mask.
        If window_fun is a callable, it takes as input the array ``(n_samples, n_dims)``
        of sample positions and returns an array of n_samples weights to be
        applied to the selected k-space values, before the smaps estimation.
    blurr_factor : float or list, optional
        The blurring factor for smoothing the sensitivity maps.
        Applies a gaussian filter on the Smap images to get smoother Sensitivty maps.
        By default it is 0.0, i.e. no smoothing is done
    mask: bool, optional default `False`
        Whether the Sensitivity maps must be masked

    ${returns}

    References
    ----------
    Loubna El Gueddari, C. Lazarus, H Carrié, A. Vignaud, Philippe Ciuciu.
    Self-calibrating nonlinear reconstruction algorithms for variable density
    sampling and parallel reception MRI. 10th IEEE Sensor Array and Multichannel
    Signal Processing workshop, Jul 2018, Sheffield, United Kingdom. ⟨hal-01782428v1⟩
    """
    # defer import to later to prevent circular import
    from mrinufft import get_operator

    try:
        from skimage.filters import gaussian, threshold_otsu
        from skimage.morphology import convex_hull_image
    except ImportError as err:
        raise ImportError(
            "The scikit-image module is not available. Please install "
            "it along with the [extra] dependencies "
            "or using `pip install scikit-image`."
        ) from err

    k_space, samples, dc = _extract_kspace_center(
        kspace_data=kspace_data,
        kspace_loc=traj,
        threshold=threshold,
        density=density,
        window_fun=window_fun,
    )
    Smaps = get_operator(backend)(
        samples,
        shape,
        n_coils=k_space.shape[-2],
        squeeze_dims=True,
    ).pinv_solver(k_space, max_iter=max_iter)
    SOS = np.linalg.norm(Smaps, axis=0)
    if isinstance(mask, np.ndarray):
        Smaps = Smaps * mask
    elif isinstance(mask, bool) and mask:
        thresh = threshold_otsu(SOS)
        # Create convex hull from mask
        convex_hull = convex_hull_image(SOS > thresh)
        Smaps = Smaps * convex_hull
    # Smooth out the sensitivity maps
    if np.sum(blurr_factor) > 0:
        if isinstance(blurr_factor, float | int):
            blurr_factor = (blurr_factor,) * SOS.ndim
        Smaps = gaussian(np.abs(Smaps), sigma=(0,) + blurr_factor) * np.exp(
            1j * np.angle(Smaps)
        )
    # Re-normalize the sensitivity maps
    if np.any(mask) or np.sum(blurr_factor) > 0:
        # ReCalculate SOS with a minor eps to ensure divide by 0 is ok
        SOS = np.linalg.norm(Smaps, axis=0) + 1e-10
    Smaps = Smaps / SOS
    return Smaps


def _nufft_calibration_kspace(
    traj: NDArray,
    shape: tuple[int, ...],
    kspace_data: NDArray,
    backend: str,
    density: NDArray | None,
    max_iter: int,
    calib_width: int | tuple[int, ...],
) -> NDArray:
    """Reconstruct the calibration region and bring it back to Cartesian k-space.

    Shared by all calibration-based (ESPIRiT-like) non-Cartesian smaps methods:
    extract the central k-space samples, reconstruct a low-resolution image
    with them, and re-grid it to Cartesian k-space via FFT.

    Parameters
    ----------
    traj : NDArray
        The trajectory of the samples.
    shape : tuple of int
        The shape of the image.
    kspace_data : NDArray
        The k-space data.
    backend : str
        The backend used for the NUFFT operator.
    density : NDArray, optional
        The density compensation weights.
    max_iter : int
        The max iterations for the internal `pinv` computation.
    calib_width : int or tuple of int
        The calibration region width.

    Returns
    -------
    NDArray
        The calibration region, on a Cartesian k-space grid, shape
        ``(n_coils, *shape)``.
    """
    # defer import to later to prevent circular import
    from mrinufft import get_operator

    k_space, samples, dc = _extract_kspace_center(
        kspace_data=kspace_data,
        kspace_loc=traj,
        threshold=tuple(float(sh) for sh in calib_width / np.asarray(shape)),
        density=density,
        window_fun="rect",
    )
    central_kspace_img = get_operator(backend)(
        samples,
        shape,
        n_coils=k_space.shape[-2],
        squeeze_dims=True,
    ).pinv_solver(k_space, max_iter=max_iter)

    return fft(central_kspace_img, dims=len(shape))


def _kernel_mask(kernel_width, kernel_shape, xp):
    """Boolean mask selecting the FIR filter support Lambda within a kernel block.

    Parameters
    ----------
    kernel_width : tuple of int
        Side length of the kernel block along each spatial dimension.
    kernel_shape : {"rect", "ellipsoid"}
        The shape of the kernel support.
    xp : module
        Array module (``numpy`` or ``cupy``) to compute with.

    Notes
    -----
    ``kernel_shape="ellipsoid"`` implements the ellipsoidal support of Lobos
    et al. (Sec. IV-B), cutting corners compares to `"rect"`.

    Returns
    -------
    NDArray or None
        Boolean mask of shape `kernel_width` for ``"ellipsoid"``, or
        ``None`` for ``"rect"`` (the full block, no masking needed).
    """
    if kernel_shape == "rect":
        return None
    if kernel_shape != "ellipsoid":
        raise ValueError(f"Unknown kernel_shape: {kernel_shape!r}")
    grid = xp.meshgrid(
        *[xp.arange(k) - (k - 1) / 2 for k in kernel_width], indexing="ij"
    )
    # Scale each axis by its own half-width so the ellipsoid is inscribed in
    # the (possibly anisotropic) box, rather than a sphere of radius
    # min(kernel_width)/2 clipped to it.
    return sum((g / (k / 2)) ** 2 for g, k in zip(grid, kernel_width)) <= 1


def _chc_via_fft(calib, kernel_width, mask):
    """Compute C^H C (Eq. 6) without ever forming C, via Lobos et al. Sec. IV-A.

    Precomputes zero-padded FFTs of the calibration data for each coil (Q
    FFTs) and forms every pairwise cross-correlation r_pq[n] via Q^2 inverse
    FFTs (Eq. 13); entries of C^H C are then read off by gathering, for
    every pair of kernel-support positions, the correlation value at their
    spatial difference. This neglects the "extraction of P valid rows"
    masking step of Eq. 13 (following Ongie & Jacob [28]), which is a minor
    approximation in the regime P >> |Lambda| that PISCO targets; the
    O(P (Q|Lambda|)^2) direct computation of C^H C via matrix
    multiplication is replaced here by O(Q^2 N log N + Q^2 |Lambda|^2).

    Parameters
    ----------
    calib : NDArray
        Calibration k-space block, shape ``(n_coils, *calib_width)``.
    kernel_width : tuple of int
        Side length of the kernel support Lambda along each dimension.
    mask : NDArray of bool, or None
        Boolean mask of shape `kernel_width` selecting a non-rectangular
        (e.g. ellipsoidal) support, as returned by `_kernel_mask`. ``None``
        keeps the full rectangular support.

    Returns
    -------
    ChC : NDArray
        The Hermitian matrix C^H C, shape
        ``(n_coils * patch_size, n_coils * patch_size)``.
    patch_size : int
        Number of kernel entries kept per coil (``prod(kernel_width)`` if
        `mask` is None, ``mask.sum()`` otherwise).
    """
    xp = get_array_module(calib)
    n_coils = calib.shape[0]
    calib_width = calib.shape[1:]
    ndim = len(calib_width)

    pad_shape = tuple(cw + kw - 1 for cw, kw in zip(calib_width, kernel_width))
    S = xp.fft.fftn(calib, s=pad_shape, axes=range(1, ndim + 1))
    corr = xp.fft.ifftn(
        xp.conj(S)[:, None, ...] * S[None, :, ...], axes=range(2, ndim + 2)
    )  # corr[p, q, n] = sum_k conj(calib_p[k]) calib_q[k + n]

    grid = xp.meshgrid(*[xp.arange(k) for k in kernel_width], indexing="ij")
    idx = xp.stack(grid, axis=-1)
    idx = idx[mask] if mask is not None else idx.reshape(-1, ndim)
    patch_size = idx.shape[0]

    diff = (idx[None, :, :] - idx[:, None, :]) % xp.asarray(pad_shape)
    gather = tuple(diff[..., d] for d in range(ndim))
    block = corr[(slice(None), slice(None)) + gather]  # (p, q, s, t)
    ChC = block.transpose(0, 2, 1, 3).reshape(
        n_coils * patch_size, n_coils * patch_size
    )
    return ChC, patch_size


def _scatter_add(xp, W, pos, spatial):
    """Fold `W` into a full-resolution ``(n_coils, n_coils, *spatial)`` array.

    Different `(m, n)` kernel-tap pairs can share the same `pos[m, n]`
    (whenever `idx[m] - idx[n]` collides for two pairs), so a plain indexed
    write would silently keep only one of them; this instead sums the
    duplicates, via a single vectorized `bincount` over a combined
    (coil-pair, spatial) index -- real and imaginary parts separately,
    since neither numpy's nor cupy's `bincount` accepts complex weights.
    This is also markedly faster than `numpy.add.at`/`cupyx.scatter_add` at
    the array sizes `_gram_via_projector` uses it at.

    Parameters
    ----------
    xp : module
        Array module (``numpy`` or ``cupy``) `W` and `pos` belong to.
    W : NDArray
        Values to accumulate, shape ``(n_coils, n_coils, *pos.shape[:-1])``.
    pos : NDArray
        Target spatial position for each entry of `W`'s trailing axes,
        shape ``(*idx_shape, len(spatial))``, with values in
        ``[0, spatial[d])``.
    spatial : tuple of int
        Spatial shape to scatter into.

    Returns
    -------
    NDArray
        The accumulated array, shape ``(n_coils, n_coils, *spatial)``.
    """
    n_coils = W.shape[0]
    batch_size = n_coils * n_coils
    n_spatial = int(np.prod(spatial))

    flat_spatial = xp.ravel_multi_index(
        tuple(pos[..., d].reshape(-1) for d in range(len(spatial))), spatial
    )
    batch_offset = xp.arange(batch_size) * n_spatial
    combined = (batch_offset[:, None] + flat_spatial[None, :]).reshape(-1)
    flat_W = W.reshape(batch_size, -1)
    total_bins = batch_size * n_spatial

    real = xp.bincount(
        combined, weights=xp.real(flat_W).reshape(-1), minlength=total_bins
    )
    if xp.iscomplexobj(W):
        imag = xp.bincount(
            combined, weights=xp.imag(flat_W).reshape(-1), minlength=total_bins
        )
        result = real + 1j * imag
    else:
        result = real
    return result.reshape(n_coils, n_coils, *spatial).astype(W.dtype)


def _gram_via_projector(U, kernel_width, mask, target_shape, dtype):
    """Compute the local Gram matrices G(x) via PISCO's W-matrix FFT trick.

    Implements Lobos et al. Eq. (14)-(16): given an orthonormal basis of a
    Q|Lambda|-dimensional subspace (the approximate nullspace for PISCO, or
    the signal subspace for ESPIRiT), evaluates G(x) = H(x)^H H(x) (or
    A(x)^H A(x) for ESPIRiT) at every voxel x with a single set of Q^2
    FFTs, independent of the subspace dimension R.

    Parameters
    ----------
    U : NDArray
        Orthonormal basis (as columns) of the subspace, shape
        ``(n_coils * patch_size, R)``, laid out coil-major (row index =
        ``coil * patch_size + tap``).
    kernel_width : tuple of int
        Side length of the kernel support Lambda along each dimension.
    mask : NDArray of bool, or None
        Boolean mask of shape `kernel_width` selecting the kernel support,
        as returned by `_kernel_mask` (must match the mask `U` was derived
        with). ``None`` for the full rectangular support.
    target_shape : tuple of int
        Shape ``(n_coils, *spatial_shape)`` of the voxel grid to evaluate
        G(x) on.
    dtype : numpy.dtype
        Output dtype.

    Returns
    -------
    NDArray
        The local Gram matrices, shape
        ``(*spatial_shape[::-1], n_coils, n_coils)``.
    """
    xp = get_array_module(U)
    n_coils = target_shape[0]
    patch_size = U.shape[0] // n_coils
    ndim = len(kernel_width)
    spatial = target_shape[1:]

    grid = xp.meshgrid(*[xp.arange(k) for k in kernel_width], indexing="ij")
    idx = xp.stack(grid, axis=-1)
    idx = idx[mask] if mask is not None else idx.reshape(-1, ndim)

    # Position, in a full spatial-shaped array, whose ifft (following this
    # module's ifftshift/ifftn/fftshift convention) carries exactly
    # frequency `idx[m] - idx[n]` -- independent of any kernel/patch-size
    # dependent centering, since that cancels between the conjugated (m)
    # and non-conjugated (n) factors of W.
    diff = idx[:, None, :] - idx[None, :, :]
    spatial_arr = xp.asarray(spatial)
    pos = (diff + spatial_arr // 2) % spatial_arr

    W = xp.conj(U @ xp.conj(U).T).reshape(n_coils, patch_size, n_coils, patch_size)
    W = W.transpose(0, 2, 1, 3)  # (q, p, m, n)

    full = _scatter_add(xp, W, pos, spatial)

    img = ifft(full, dims=ndim) / xp.sqrt(float(np.prod(spatial)))
    # G[..., q, p] = ifft(full)[q, p, ...]: move spatial axes first (in
    # reversed order, matching this module's other `.T`-based conventions)
    # then (q, p).
    axes_order = tuple(range(ndim + 1, 1, -1)) + (0, 1)
    return img.transpose(*axes_order).astype(dtype)


def _hamming_window(n, xp):
    """1D Hamming (raised-cosine) apodization window of length `n`.

    Uses the same `_cosine_window` formula as the k-space windowing in
    `_extract_kspace_center`, here evaluated over the (normalized) distance
    of each grid index to the array's center bin rather than a k-space radius.

    Parameters
    ----------
    n : int
        Length of the window.
    xp : module
        Array module (``numpy`` or ``cupy``) to compute with.

    Returns
    -------
    NDArray
        The window weights, shape ``(n,)``.
    """
    if n == 1:
        return xp.ones(1)
    distance_ratio = xp.abs(2 * xp.arange(n) / (n - 1) - 1)
    return _cosine_window(distance_ratio, "hamming", xp)


def _gaussian_window(n, xp, std_frac=0.4):
    """1D Gaussian apodization window of length `n`.

    Parameters
    ----------
    n : int
        Length of the window.
    xp : module
        Array module (``numpy`` or ``cupy``) to compute with.
    std_frac : float, optional
        Standard deviation of the Gaussian, as a fraction of the window's
        half-width. By default 0.4.

    Returns
    -------
    NDArray
        The window weights, shape ``(n,)``.
    """
    if n == 1:
        return xp.ones(1)
    x = xp.arange(n) - (n - 1) / 2
    std = std_frac * (n - 1) / 2
    return xp.exp(-0.5 * (x / std) ** 2)


def _outer_window(window_1d, shape, xp):
    """N-D separable apodization window of the given `shape` from `window_1d`.

    Parameters
    ----------
    window_1d : Callable[[int, module], NDArray]
        1D window function, e.g. `_hamming_window` or `_gaussian_window`.
    shape : tuple of int
        Shape of the N-D window to build, one length per axis.
    xp : module
        Array module (``numpy`` or ``cupy``) to compute with.

    Returns
    -------
    NDArray
        The N-D window weights, of shape `shape`, built as the outer
        product of the per-axis 1D windows.
    """
    win = window_1d(shape[0], xp)
    for n in shape[1:]:
        win = xp.multiply.outer(win, window_1d(n, xp))
    return win


def _fft_interpolate(arr, out_shape, xp, window=False, complex_output=False):
    """Interpolate the last axes of an array via periodic sinc interpolation.

    Implements Lobos et al. Sec. IV-D with zero-padded FFTs: crop/zero-pad
    the (centered) spectrum along the last ``len(out_shape)`` axes, then
    invert.

    Parameters
    ----------
    arr : NDArray
        Array to interpolate, its last ``len(out_shape)`` axes matching
        the low-resolution grid.
    out_shape : tuple of int
        Target (higher-resolution) shape for the last ``len(out_shape)``
        axes.
    xp : module
        Array module (``numpy`` or ``cupy``) to compute with.
    window : bool, optional
        If True, apply a Hamming apodization (`_hamming_window`) to the
        low-res spectrum before zero-padding trading a little
        resolution for less Gibbs ringing. By default False.
    complex_output : bool, optional
        If True, keep the interpolated array complex (e.g. for a
        phase-referenced Smaps array); otherwise only the real part is
        returned (appropriate for real-valued inputs such as magnitude or
        eigenvalue maps). By default False.

    Returns
    -------
    NDArray
        The interpolated array, with its last ``len(out_shape)`` axes
        replaced by `out_shape`.
    """
    ndim = len(out_shape)
    axes = tuple(range(-ndim, 0))
    in_shape = arr.shape[-ndim:]
    # Promote to the matching complex dtype (e.g. complex64 for float32 or
    # complex64 input) rather than `complex`, which numpy/cupy always
    # resolve to complex128 and would silently double memory/compute.
    complex_dtype = xp.result_type(arr.dtype, xp.complex64)
    k = xp.fft.fftshift(
        xp.fft.fftn(arr.astype(complex_dtype), axes=axes, norm="ortho"), axes=axes
    )
    if window:
        k = k * _outer_window(_hamming_window, in_shape, xp)
    k = _crop_or_pad(k, arr.shape[:-ndim] + out_shape)
    out = xp.fft.ifftn(xp.fft.ifftshift(k, axes=axes), axes=axes, norm="ortho")
    scale = float(np.sqrt(np.prod(out_shape) / np.prod(in_shape)))
    out = out * scale
    return out if complex_output else xp.real(out)


def _low_res_calib_image(calib, target_shape, xp):
    """Gaussian-apodized IFFT reconstruction of the calibration region.

    Used only to obtain a smooth, physically-meaningful phase reference for
    FFT interpolation (see `_decim_and_interpolate`); the apodization
    reduces truncation (Gibbs) artifacts from the small calibration block,
    matching the reference PISCO MATLAB implementation.

    Parameters
    ----------
    calib : NDArray
        Calibration k-space block, shape ``(n_coils, *calib_width)``.
    target_shape : tuple of int
        Spatial shape to zero-pad the (apodized) calibration block to
        before the IFFT.
    xp : module
        Array module (``numpy`` or ``cupy``) to compute with.

    Returns
    -------
    NDArray
        The low-resolution per-coil image, shape
        ``(n_coils, *target_shape)``.
    """
    ndim = len(target_shape)
    windowed = calib * _outer_window(_gaussian_window, calib.shape[-ndim:], xp)
    padded = _crop_or_pad(windowed, calib.shape[:-ndim] + target_shape)
    return ifft(padded, dims=ndim)


def _cleanup_gpu(xp):
    """Force garbage collection and free the GPU memory pool, if applicable.

    Parameters
    ----------
    xp : module
        Array module in use (``numpy`` or ``cupy``); the memory pool is
        only freed when this is ``cupy``.
    """
    gc.collect()
    if xp is not np:
        xp.get_default_memory_pool().free_all_blocks()


def _batched_power_iteration(
    G: NDArray,
    n_iter: int = 30,
    mode: str = "max",
    seed: int = 0,
) -> tuple[NDArray, NDArray]:
    """Batched power iteration for an extreme eigenpair of Hermitian PSD matrices.

    At every batch location, extracts the eigenvector associated with the
    largest (``mode="max"``) or smallest (``mode="min"``, via a shifted
    deflation ``shift * I - G``) eigenvalue of a Hermitian positive
    semi-definite matrix, without ever forming a full eigendecomposition.

    Parameters
    ----------
    G : NDArray
        Stack of Hermitian PSD matrices, e.g. local coil-covariance
        matrices, shape ``(..., n_coils, n_coils)``.
    n_iter : int, optional
        Number of power iterations. By default 30.
    mode : {"max", "min"}, optional
        Which eigenpair to extract. By default "max".
    seed : int, optional
        Seed for the (shared, across batch locations) random initial vector.

    Returns
    -------
    eigval : NDArray
        The extracted eigenvalue at each batch location, shape ``(...,)``.
    eigvec : NDArray
        The associated unit-norm eigenvector, shape ``(..., n_coils)``.
    """
    xp = get_array_module(G)
    n_coils = G.shape[-1]
    batch_shape = G.shape[:-2]

    if mode == "min":
        # Shift by a power-iteration estimate of the largest eigenvalue, so
        # that power iteration on `shift * I - G` converges to G's smallest
        # eigenpair. A loose bound (e.g. trace(G)) would work in theory, but
        # collapses the eigenvalue gap of the shifted operator and can stall
        # convergence; using the actual (approximate) largest eigenvalue
        # keeps that gap as large as possible.
        max_eigval, _ = _batched_power_iteration(
            G, n_iter=n_iter, mode="max", seed=seed
        )
        shift = max_eigval[..., None, None]
        op = shift * xp.eye(n_coils, dtype=G.dtype) - G
    elif mode == "max":
        op = G
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    rng = np.random.default_rng(seed)
    v0 = rng.standard_normal(n_coils) + 1j * rng.standard_normal(n_coils)
    v0 = (v0 / np.linalg.norm(v0)).astype(G.dtype)
    v = xp.broadcast_to(xp.asarray(v0), (*batch_shape, n_coils)).copy()[..., None]

    for _ in range(n_iter):
        v = op @ v
        norm = xp.linalg.norm(v, axis=-2, keepdims=True)
        v = v / xp.where(norm == 0, 1, norm)

    eigval = xp.real(xp.conj(v.swapaxes(-1, -2)) @ (op @ v))[..., 0, 0]
    if mode == "min":  # remove the shift introduced.
        eigval = max_eigval - eigval
    return eigval, v[..., 0]


def _subspace_gram_eig(
    kspace,
    target_shape,
    n_coils,
    calib_width,
    kernel_size,
    mask,
    thresh,
    power_iter,
    mode,
):
    """Calibration matrix -> Gram matrix -> extreme eigenpair pipeline.

    Shared by ESPIRiT (``mode="max"``, signal subspace) and PISCO
    (``mode="min"``, nullspace). Builds the calibration matrix's Gram
    matrix C^H C (Eq. 6, via the FFT-based approximation of Sec. IV-A),
    selects its `mode`-appropriate subspace at the (normalized) singular
    value threshold `thresh` (kept above `thresh` for "max"/signal, below
    for "min"/nullspace), evaluates the local coil-covariance matrix
    G(x)/A(x)^H A(x) at every voxel via PISCO's W-matrix FFT trick
    (Eq. 14-16), and extracts its extreme eigenpair via power iteration
    (Sec. IV-E).

    Parameters
    ----------
    kspace : NDArray
        K-space data containing the calibration region, shape
        ``(n_coils, *anything)``.
    target_shape : tuple of int
        Spatial shape to evaluate the Gram matrix on.
    n_coils : int
        Number of coils.
    calib_width : tuple of int
        Calibration region width along each dimension.
    kernel_size : tuple of int
        FIR filter kernel support size along each dimension.
    mask : NDArray of bool, or None
        Kernel support mask, as returned by `_kernel_mask`.
    thresh : float
        Relative singular-value threshold selecting the subspace.
    power_iter : int
        Number of power iterations.
    mode : {"max", "min"}
        Which eigenpair/subspace to extract: "max" for ESPIRiT's signal
        subspace, "min" for PISCO's nullspace.

    Returns
    -------
    Smaps : NDArray
        Sensitivity maps, phase-referenced to coil 0, shape
        ``(n_coils, *target_shape)``.
    eig : NDArray
        The extracted extreme eigenvalue map, shape `target_shape`.
    calib : NDArray
        The calibration k-space block used, shape
        ``(n_coils, *calib_width)``.
    """
    xp = get_array_module(kspace)
    calib = _crop_or_pad(kspace, (n_coils, *calib_width))

    ChC, patch_size = _chc_via_fft(calib, kernel_size, mask)
    eigval, eigvec = xp.linalg.eigh(ChC)
    eigval = xp.sqrt(xp.clip(eigval, 0, None))
    eigval /= eigval[-1]
    U = eigvec[:, eigval > thresh] if mode == "max" else eigvec[:, eigval < thresh]

    G = _gram_via_projector(
        U, kernel_size, mask, (n_coils, *target_shape), kspace.dtype
    )
    G *= np.prod(target_shape) / patch_size

    eig, Smaps = _batched_power_iteration(G, n_iter=power_iter, mode=mode)
    Smaps = Smaps.T
    eig = eig.T
    Smaps = Smaps * xp.conj(Smaps[0] / xp.abs(Smaps[0]))
    return Smaps, eig, calib


def _decim_and_interpolate(xp, shape, decim, calib_width, compute_fn, kspace, n_coils):
    """Shared low-resolution + interpolation driver for PISCO/ESPIRiT (Sec. IV-D).

    Since sensitivity maps are spatially smooth, G(x)/A(x)^H A(x) and its
    extreme eigenvector are estimated on a grid decimated by `decim` (at
    least as large as the calibration region) and interpolated back to
    `shape` using periodic sinc interpolation (zero-padded FFTs).
    `compute_fn` receives the (possibly decimated) kspace and its spatial
    shape, and must return `(Smaps, eig, calib)` with `Smaps`
    phase-referenced to coil 0 and `calib` the calibration k-space block
    (shape ``(n_coils, *calib_width)``) it used.

    Parameters
    ----------
    xp : module
        Array module (``numpy`` or ``cupy``) to compute with.
    shape : tuple of int
        Target (full-resolution) spatial shape.
    decim : int
        Decimation factor. ``1`` disables decimation/interpolation
        entirely (`compute_fn` is then called directly on `shape`).
    calib_width : tuple of int
        Calibration region width along each dimension; the decimated grid
        is never made smaller than this.
    compute_fn : Callable[[NDArray, tuple[int, ...]], tuple[NDArray, NDArray, NDArray]]
        Function estimating `(Smaps, eig, calib)` on a given kspace and its
        spatial shape, as described above.
    kspace : NDArray
        Full-resolution k-space data, shape ``(n_coils, *shape)``.
    n_coils : int
        Number of coils.

    Returns
    -------
    Smaps : NDArray
        The (possibly interpolated) sensitivity maps, shape
        ``(n_coils, *shape)``.
    eig : NDArray
        The (possibly interpolated) extreme eigenvalue map, shape `shape`.
    """
    decim_shape = shape
    if decim > 1:
        decim_shape = tuple(max(sh // decim, cw) for sh, cw in zip(shape, calib_width))
        kspace = _crop_or_pad(kspace, (n_coils,) + decim_shape)

    Smaps, eig, calib = compute_fn(kspace, decim_shape)

    if decim > 1:
        low_res_img = _low_res_calib_image(calib, decim_shape, xp)
        num = xp.sum(xp.conj(Smaps) * low_res_img, axis=0)
        den = xp.sum(xp.abs(Smaps) ** 2, axis=0)
        cim = num / xp.where(den == 0, 1, den)
        Smaps = Smaps * xp.exp(1j * xp.angle(cim))

        Smaps = _fft_interpolate(Smaps, shape, xp, window=True, complex_output=True)
        eig = _fft_interpolate(eig, shape, xp, window=True)
    return Smaps, eig


@register_smaps
@with_numpy_cupy
@flat_traj
def espirit(
    traj: NDArray,
    shape: tuple[int, ...],
    kspace_data: NDArray,
    backend: str,
    density: NDArray | None = None,
    max_iter: int = 10,
    calib_width: int | tuple[int, ...] = 24,
    kernel_size: int | tuple[int, ...] = 7,
    kernel_shape: str = "ellipsoid",
    thresh: float = 0.05,
    crop: float = 0.08,
    decim: int = 1,
    power_iter: int = 30,
) -> NDArray:
    """ESPIRIT algorithm on non-Cartesian data.

    ${base_params}
    ${espirit_pisco_params}

    ${returns}

    ${espirit_pisco_ref}
    """
    central_kspace = _nufft_calibration_kspace(
        traj, shape, kspace_data, backend, density, max_iter, calib_width
    )
    return cartesian_espirit(
        central_kspace,
        shape,
        calib_width,
        kernel_size,
        kernel_shape,
        thresh,
        crop,
        decim,
        power_iter,
    )


@_fill_doc(_smap_docs)
@with_numpy_cupy
def cartesian_espirit(
    kspace: NDArray,
    shape: tuple[int, ...],
    calib_width: int | tuple[int, ...] = 24,
    kernel_size: int | tuple[int, ...] = 7,
    kernel_shape: str = "ellipsoid",
    thresh: float = 0.05,
    crop: float = 0.08,
    decim: int = 1,
    power_iter: int = 30,
) -> NDArray:
    """ESPIRIT algorithm on Cartesian data.

    Parameters
    ----------
    kspace: NDArray
        The k-space data in Cartesian grid. Shape ``(n_coils, *kspace_shape)``
    shape : tuple
        The shape of the image.
    ${espirit_pisco_params}

    ${returns}

    ${espirit_pisco_ref}
    """
    if isinstance(calib_width, int):
        calib_width = (calib_width,) * (kspace.ndim - 1)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * (kspace.ndim - 1)

    xp = get_array_module(kspace)
    n_coils = kspace.shape[0]
    mask = _kernel_mask(kernel_size, kernel_shape, xp)

    # Signal subspace of C (Eq. 6): eigenvectors of C^H C kept *above*
    # `thresh`. This is the complement of PISCO's nullspace filters (Sec.
    # III): ESPIRiT's eigenvalue formulation builds A(x)^H A(x) directly
    # from these signal-space kernels and takes its *largest* eigenvector,
    # shown in Eq. (12) to be algebraically equivalent to the
    # nullspace-based approach.
    def compute(kspace, target_shape):
        return _subspace_gram_eig(
            kspace,
            target_shape,
            n_coils,
            calib_width,
            kernel_size,
            mask,
            thresh,
            power_iter,
            mode="max",
        )

    Smaps, max_eig = _decim_and_interpolate(
        xp, shape, decim, calib_width, compute, kspace, n_coils
    )
    # `crop` measures the gap to the *ideal* extreme eigenvalue
    # so that the two functions share the same default:
    # x is outside the image support when that gap exceeds `crop`.
    Smaps = Smaps * (1 - max_eig < crop)
    _cleanup_gpu(xp)
    return Smaps


@register_smaps
@with_numpy_cupy
@flat_traj
def pisco(
    traj: NDArray,
    shape: tuple[int, ...],
    kspace_data: NDArray,
    backend: str,
    density: NDArray | None = None,
    max_iter: int = 10,
    calib_width: int | tuple[int, ...] = 24,
    kernel_size: int | tuple[int, ...] = 7,
    kernel_shape: str = "ellipsoid",
    thresh: float = 0.05,
    crop: float = 0.08,
    decim: int = 1,
    power_iter: int = 30,
) -> NDArray:
    """PISCO algorithm on non-Cartesian data.

    ${base_params}
    ${espirit_pisco_params}

    ${returns}

    ${espirit_pisco_ref}
    """
    central_kspace = _nufft_calibration_kspace(
        traj, shape, kspace_data, backend, density, max_iter, calib_width
    )
    return cartesian_pisco(
        central_kspace,
        shape,
        calib_width,
        kernel_size,
        kernel_shape,
        thresh,
        crop,
        decim,
        power_iter,
    )


@_fill_doc(_smap_docs)
@with_numpy_cupy
def cartesian_pisco(
    kspace: NDArray,
    shape: tuple[int, ...],
    calib_width: int | tuple[int, ...] = 24,
    kernel_size: int | tuple[int, ...] = 7,
    kernel_shape: str = "ellipsoid",
    thresh: float = 0.05,
    crop: float = 0.08,
    decim: int = 1,
    power_iter: int = 30,
) -> NDArray:
    """PISCO algorithm on Cartesian data.

    Nullspace-based estimation of sensitivity maps: nullspace vectors of a
    k-space calibration matrix are used to build, for every spatial
    location x, a small Gram matrix G(x) whose eigenvector of smallest
    eigenvalue is the coil sensitivity profile at x.

    Parameters
    ----------
    kspace: NDArray
        The k-space data in Cartesian grid. Shape ``(n_coils, *kspace_shape)``
    shape : tuple
        The shape of the image.
    ${espirit_pisco_params}

    ${returns}

    ${espirit_pisco_ref}
    """
    xp = get_array_module(kspace)
    n_coils = kspace.shape[0]
    ndim = kspace.ndim - 1

    if isinstance(calib_width, int):
        calib_width = (calib_width,) * ndim
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * ndim
    # Ellipsoidal FIR filter support Lambda = {n : ||n||_2 <= radius}
    # (Sec. IV-B, Ref. [30]): drops the corners of the rectangular support,
    # negligible effect on the estimated maps but reduces |Lambda| (and
    # hence the cost of Eq. 6/13/15-16).
    mask = _kernel_mask(kernel_size, kernel_shape, xp)

    # Approximate nullspace {h_r}_{r=1..R} of C (Eq. 5): eigenvectors of
    # C^H C kept *below* `thresh`, evaluated into the local Gram matrix
    # G(x) = H(x)^H H(x) (Eq. 14) via PISCO's W-matrix FFT trick
    # (Eq. 15-16). Sensitivity map c(x) at each voxel x = approximate
    # nullspace vector of H(x), i.e. eigenvector of the smallest eigenvalue
    # of G(x) (Eq. 9-11).
    def compute(kspace, target_shape):
        Smaps, min_eig, calib = _subspace_gram_eig(
            kspace,
            target_shape,
            n_coils,
            calib_width,
            kernel_size,
            mask,
            thresh,
            power_iter,
            mode="min",
        )
        den = xp.sqrt(xp.sum(xp.abs(Smaps) ** 2, axis=0))
        den = xp.where(den == 0, 1, den)
        Smaps = Smaps / den
        return Smaps, min_eig, calib

    Smaps, min_eig = _decim_and_interpolate(
        xp, shape, decim, calib_width, compute, kspace, n_coils
    )
    Smaps = Smaps * (min_eig < crop)
    _cleanup_gpu(xp)
    return Smaps


@with_numpy_cupy
def coil_compression(
    kspace_data: NDArray,
    K: int | float,
    traj: NDArray | None = None,
    krad_thresh: float | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Coil compression using principal component analysis on k-space data.

    Parameters
    ----------
    kspace_data : NDArray
        Multi-coil k-space data. Shape: (n_coils, n_samples).
    K : int or float
        Number of virtual coils to retain (if int), or energy threshold (if
        float between 0 and 1).
    traj : NDArray, optional
        Sampling trajectory. Shape: (n_samples, n_dims).
    krad_thresh : float, optional
        Relative k-space radius (as a fraction of maximum) to use for selecting
        the calibration region for principal component analysis. If None, use
        all k-space samples.

    Returns
    -------
    NDArray
        Coil-compressed data. Shape: (K, n_samples) if K is int, number of
        retained components otherwise.
    NDArray
        The compression matrix. Shape: (K, n_coils).
    """
    xp = get_array_module(kspace_data)

    if krad_thresh is not None and traj is not None:
        traj_rad = xp.sqrt(xp.sum(traj**2, axis=-1))
        center_data = kspace_data[:, traj_rad < krad_thresh * xp.max(traj)]
    elif krad_thresh is None:
        center_data = kspace_data
    else:
        raise ValueError("traj and krad_thresh must be specified.")

    # Compute the covar matrix of selected data
    cov = center_data @ center_data.T.conj()
    w, v = xp.linalg.eigh(cov)
    # sort eigenvalues largest to smallest
    si = xp.argsort(w)[::-1]
    w_sorted = w[si]
    v_sorted = v[si]
    if isinstance(K, float):
        # retain enough components to reach energy K
        w_cumsum = xp.cumsum(w_sorted)  # from largest to smallest
        total_energy = xp.sum(w_sorted)
        K = int(xp.searchsorted(w_cumsum / total_energy, K, side="left") + 1)
        K = min(K, w_sorted.size)
    V = v_sorted[:K]  # use top K component
    compress_data = V @ kspace_data
    return compress_data, V
