"""SMaps module for sensitivity maps estimation."""

from __future__ import annotations

from mrinufft.density.utils import flat_traj
from mrinufft._array_compat import with_numpy_cupy, get_array_module, with_torch
from mrinufft._utils import MethodRegister
import numpy as np
from mrinufft.extras.cartesian import fft, ifft
from numpy.typing import NDArray

from collections.abc import Callable


def _extract_kspace_center(
    kspace_data: NDArray,
    kspace_loc: NDArray,
    threshold: float | tuple[float, ...] = None,
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

    w(x,y) = a_0 - (1-a_0) * \cos(\pi * \sqrt{x^2+y^2}/\theta),
    \sqrt{x^2+y^2} \le \theta

    In the case of Hann window :math:`a_0=0.5`.
    For Hamming window we consider the optimal value in the equiripple sense:
    :math:`a_0=0.53836`.
    .. Wikipedia:: https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows

    """
    xp = get_array_module(kspace_data)
    if isinstance(threshold, float):
        threshold = (threshold,) * kspace_loc.shape[1]

    if window_fun == "rect":
        condition = np.logical_and.reduce(
            tuple(
                np.abs(kspace_loc[:, i]) <= threshold[i] for i in range(len(threshold))
            )
        )
        center_locations = kspace_loc[condition, :]
        data_thresholded = kspace_data[:, condition]
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
                radius = np.linalg.norm(kspace_loc, axis=1)
                a_0 = 0.5 if window_fun in ["hann", "hanning"] else 0.53836
                window = a_0 + (1 - a_0) * np.cos(np.pi * radius / threshold[0])
            elif window_fun == "ellipse":
                window = np.sum(kspace_loc**2 / np.asarray(threshold) ** 2, axis=1) <= 1
            else:
                raise ValueError("Unsupported window function.")
            if xp != np:
                window = xp.asarray(window)
        data_thresholded = window * kspace_data
        # Return k-space locations & density just for consistency
        return data_thresholded, kspace_loc, density


register_smaps = MethodRegister("smaps")
get_smaps = register_smaps.make_getter()


@with_numpy_cupy
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
    xp = np if isinstance(arr, np.ndarray) else __import__("cupy")
    in_shape = arr.shape
    pad_width = []
    slices = []

    for i, (s, t) in enumerate(zip(in_shape, target_shape)):
        diff = t - s
        if diff > 0:
            # need to pad
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((pad_before, pad_after))
            slices.append(slice(None))
        else:
            # need to crop
            crop_before = (-diff) // 2
            crop_after = crop_before + t
            pad_width.append((0, 0))
            slices.append(slice(crop_before, crop_after))

    arr = arr[tuple(slices)]
    if any(pw != (0, 0) for pw in pad_width):
        arr = xp.pad(arr, pad_width, mode=mode, constant_values=constant_values)
    return arr


@register_smaps
@flat_traj
def low_frequency(
    traj: NDArray,
    shape: tuple[int, ...],
    kspace_data: NDArray,
    backend: str,
    threshold: float | tuple[float, ...] = 0.1,
    density: NDArray | None = None,
    window_fun: str | Callable[[NDArray], NDArray] = "ellipse",
    blurr_factor: int | float | tuple[float, ...] = 0.0,
    mask: bool | NDArray = False,
) -> NDArray:
    """
    Calculate low-frequency sensitivity maps.

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
    window_fun: "Hann", "Hanning", "Hamming", or a callable, default None.
        The window function to apply to the selected data. It is computed with
        the center locations selected. Only works with circular mask.
        If window_fun is a callable, it takes as input the array (n_samples x n_dims)
        of sample positions and returns an array of n_samples weights to be
        applied to the selected k-space values, before the smaps estimation.
    blurr_factor : float or list, optional
        The blurring factor for smoothing the sensitivity maps.
        Applies a gaussian filter on the Smap images to get smoother Sensitivty maps.
        By default it is 0.0, i.e. no smoothing is done
    mask: bool, optional default `False`
        Whether the Sensitivity maps must be masked

    Returns
    -------
    Smaps : numpy.ndarray
        The low-frequency sensitivity maps.
    """
    # defer import to later to prevent circular import
    from mrinufft import get_operator

    try:
        from skimage.filters import threshold_otsu, gaussian
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
    smaps_adj_op = get_operator(backend)(
        samples,
        shape,
        density=dc,
        n_coils=k_space.shape[-2],
        squeeze_dims=True,
    )
    Smaps = smaps_adj_op.cg(k_space)
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
        if isinstance(blurr_factor, (float, int)):
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


@with_torch
def _unfold_blocks(calib, calib_width):
    for i, width in enumerate(calib_width):
        calib = calib.unfold(dimension=i + 1, size=width, step=1)
    return calib


@register_smaps
@flat_traj
def espirit(
    traj: NDArray,
    shape: tuple[int, ...],
    kspace_data: NDArray,
    backend: str,
    density: NDArray | None = None,
    calib_width: int | tuple[int, ...] = 24,
    kernel_width: int | tuple[int, ...] = 6,
    thresh: float = 0.02,
    crop: float = 0.95,
    decim: int = 1,
) -> NDArray:
    """
    Calculate low-frequency sensitivity maps.

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
        The backend used for the operator for `pinv` computation
    density : numpy.ndarray, optional
        The density compensation weights.
    calib_width : int or tuple of int, optional
        The calibration region width. By default it is 24.
    kernel_width : int or tuple of int, optional
        The kernel width. By default it is 6.
    thresh : float, optional
        The threshold for the singular values. By default it is 0.02.
    crop : float, optional
        The cropping threshold for the sensitivity maps. By default it is 0.95.
    decim : int, optional
        The decimation factor for the caluclation of sensitivity maps. By default it is 1.
        This can be used to speed up the computation and significantly reduce memory usage.
        The final result is upsampled back to the original size through linear interpolation.

    Returns
    -------
    Smaps : numpy.ndarray
        The sensitivity maps.
    """
    # defer import to later to prevent circular import
    from mrinufft import get_operator
    from mrinufft.operators.base import power_method

    try:
        from skimage.restoration import unwrap_phase
    except ImportError as err:
        raise ImportError(
            "The scikit-image module is not available. Please install "
            "it along with the [extra] dependencies "
            "or using `pip install scikit-image`."
        ) from err
    if isinstance(calib_width, int):
        calib_width = (calib_width,) * traj.shape[-1]
    if isinstance(kernel_width, int):
        kernel_width = (kernel_width,) * traj.shape[-1]
    xp = get_array_module(kspace_data)
    k_space, samples, dc = _extract_kspace_center(
        kspace_data=kspace_data,
        kspace_loc=traj,
        threshold=tuple(float(sh) for sh in calib_width / np.asarray(shape)),
        density=density,
        window_fun="rect",
    )
    n_coils = k_space.shape[0]
    central_kspace_img = get_operator(backend)(
        samples,
        shape,
        density=dc,
        n_coils=k_space.shape[-2],
        squeeze_dims=True,
    ).cg(k_space)
    central_kspace = fft(central_kspace_img)
    if decim > 1:
        central_kspace = _crop_or_pad(
            central_kspace,
            tuple(
                sh // decim if i != 0 else sh
                for i, sh in enumerate(central_kspace.shape)
            ),
        )
    calib_shape = (n_coils, *calib_width)
    calib = _crop_or_pad(central_kspace, calib_shape)
    calib = _unfold_blocks(calib, kernel_width)
    calib = calib.reshape(
        calib.shape[0],
        -1,
        np.prod(kernel_width),
    )
    calib = calib.transpose(1, 0, 2).reshape(-1, calib.shape[0] * np.prod(kernel_width))
    _, S, VH = xp.linalg.svd(calib, full_matrices=False)
    VH = VH[S > thresh * S.max(), :]
    # Get kernels
    kernels = VH.reshape((len(VH), n_coils, *kernel_width))
    # Get covariance matrix in image domain
    AHA = xp.zeros(
        central_kspace.shape[1:][::-1] + (n_coils, n_coils), dtype=kspace_data.dtype
    )
    for kernel in kernels:
        img_kernel = ifft(_crop_or_pad(kernel, central_kspace.shape))
        aH = xp.expand_dims(img_kernel.T, axis=-1)
        a = xp.conj(aH.swapaxes(-1, -2))
        AHA += aH @ a

    AHA *= np.prod(central_kspace.shape[1:]) / np.prod(kernel_width)
    Smaps = xp.ones(central_kspace.shape[::-1] + (1,), dtype=kspace_data.dtype)

    def forward(x):
        return AHA @ x

    def normalize(x):
        return xp.sum(xp.abs(x) ** 2, axis=-2, keepdims=True) ** 0.5

    max_eig, Smaps = power_method(
        max_iter=100,
        operator=forward,
        norm_func=normalize,
        x=Smaps,
        check_convergence=False,
        return_eigvec=True,
    )
    Smaps = Smaps.T[0]
    Smaps *= xp.conj(Smaps[0] / xp.abs(Smaps[0]))
    Smaps *= max_eig.T[0] > crop
    if decim > 1:
        if xp.__name__ == "numpy":
            from scipy.ndimage import zoom

            unwrapped_phase = xp.array(
                [unwrap_phase(smap) for smap in xp.angle(Smaps)], dtype=xp.float32
            )
        else:
            from cupyx.scipy.ndimage import zoom

            unwrapped_phase = xp.array(
                [unwrap_phase(smap.get()) for smap in xp.angle(Smaps)], dtype=xp.float32
            )
        abs_maps = zoom(abs(Smaps), (1,) + (decim,) * (Smaps.ndim - 1), order=1)
        angle_maps = zoom(unwrapped_phase, (1,) + (decim,) * (Smaps.ndim - 1), order=1)
        Smaps = abs_maps * np.exp(1j * angle_maps)
    return Smaps


@with_numpy_cupy
def coil_compression(
    kspace_data: NDArray,
    K: int | float,
    traj: NDArray | None = None,
    krad_thresh: float | None = None,
    return_V: bool = False,
) -> NDArray | tuple[NDArray, NDArray]:
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
    return_V : bool, optional
        Whether to return the compression matrix V.

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
    if return_V:
        return compress_data, V
    return compress_data
