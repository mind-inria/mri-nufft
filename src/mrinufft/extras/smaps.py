"""Smaps module for sensitivity maps estimation.

.. autoregistry:: smaps

"""

from __future__ import annotations

from mrinufft.density.utils import flat_traj
from mrinufft._array_compat import with_numpy_cupy, get_array_module
from mrinufft._utils import MethodRegister, _fill_doc
import numpy as np
from mrinufft.extras.cartesian import fft, ifft
from mrinufft._array_compat import with_numpy
from numpy.typing import NDArray

from collections.abc import Callable


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
    espirit_params="""
calib_width : int or tuple of int, optional
    The calibration region width. By default it is 24.
kernel_width : int or tuple of int, optional
    The kernel width. By default it is 6.
thresh : float, optional
    The threshold for the singular values. By default it is 0.02.
crop : float, optional
    The cropping threshold for the sensitivity maps.
    By default it is 0.95.
decim : int, optional
    The decimation factor for the caluclation of sensitivity maps.
    By default it is 1. This can be used to speed up the computation
    and significantly reduce memory usage. The final result is
    upsampled back to the original size through linear
    interpolation.
Returns
-------
Smaps : NDArray
    The sensitivity maps
""",
    espirit_ref="""
References
----------
    Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS,
    Lustig M. ESPIRiT--an eigenvalue approach to autocalibrating parallel
    MRI: where SENSE meets GRAPPA. Magn Reson Med.
    2014 Mar;71(3):990-1001. doi: 10.1002/mrm.24751.
    PMID: 23649942; PMCID: PMC4142121.
""",
)


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


register_smaps = MethodRegister("smaps", docstring_subs=_smap_docs)
get_smaps = register_smaps.make_getter()


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
        If window_fun is a callable, it takes as input the array (n_samples x n_dims)
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


def _unfold_blocks(calib, calib_width):
    xp = get_array_module(calib)
    return xp.lib.stride_tricks.sliding_window_view(
        calib, calib_width, axis=tuple(range(1, calib.ndim))
    )


@register_smaps
@flat_traj
def espirit(
    traj: NDArray,
    shape: tuple[int, ...],
    kspace_data: NDArray,
    backend: str,
    density: NDArray | None = None,
    max_iter: int = 10,
    calib_width: int | tuple[int, ...] = 24,
    kernel_width: int | tuple[int, ...] = 6,
    thresh: float = 0.02,
    crop: float = 0.95,
    decim: int = 1,
) -> NDArray:
    """ESPIRIT algorithm on non-Cartesian data.

    ${base_params}
    ${espirit_params}

    ${returns}

    ${espirit_ref}
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
    central_kspace = fft(central_kspace_img)
    return cartesian_espirit(
        central_kspace, shape, calib_width, kernel_width, thresh, crop, decim
    )


@_fill_doc(_smap_docs)
def cartesian_espirit(
    kspace: NDArray,
    shape: tuple[int, ...],
    calib_width: int | tuple[int, ...] = 24,
    kernel_width: int | tuple[int, ...] = 6,
    thresh: float = 0.02,
    crop: float = 0.95,
    decim: int = 1,
) -> NDArray:
    """ESPIRIT algorithm on Cartesian data.

    Parameters
    ----------
    kspace: NDArray
        The k-space data in Cartesian grid. Shape (n_coils, *kspace_shape)
    shape : tuple
        The shape of the image.
    ${espirit_params}

    ${returns}

    ${espirit_ref}
    """
    from mrinufft.operators.base import power_method

    if isinstance(calib_width, int):
        calib_width = (calib_width,) * (kspace.ndim - 1)
    if isinstance(kernel_width, int):
        kernel_width = (kernel_width,) * (kspace.ndim - 1)

    xp = get_array_module(kspace)
    n_coils = kspace.shape[0]
    if decim > 1:
        try:
            from skimage.restoration import unwrap_phase
        except ImportError as err:
            raise ImportError(
                "The scikit-image module is not available. Please install "
                "it along with the [extra] dependencies "
                "or using `pip install scikit-image`."
            ) from err
        kspace = _crop_or_pad(
            kspace,
            (kspace.shape[0],) + tuple(sh // decim for i, sh in enumerate(shape)),
        )
    calib_shape = (n_coils, *calib_width)
    calib = _crop_or_pad(kspace, calib_shape)
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
    AHA = xp.zeros(kspace.shape[1:][::-1] + (n_coils, n_coils), dtype=kspace.dtype)
    for kernel in kernels:
        img_kernel = ifft(_crop_or_pad(kernel, kspace.shape))
        aH = xp.expand_dims(img_kernel.T, axis=-1)
        a = xp.conj(aH.swapaxes(-1, -2))
        AHA += aH @ a

    AHA *= np.prod(kspace.shape[1:]) / np.prod(kernel_width)
    Smaps = xp.ones(kspace.shape[::-1] + (1,), dtype=kspace.dtype)

    def forward(x):
        return AHA @ x

    def normalize(x):
        return xp.sum(xp.abs(x) ** 2, axis=-2, keepdims=True) ** 0.5

    max_eig, Smaps = power_method(
        max_iter=100,
        operator=forward,
        norm_func=normalize,
        x=Smaps,
    )
    Smaps = Smaps.T[0]
    Smaps *= xp.conj(Smaps[0] / xp.abs(Smaps[0]))
    if decim > 1:
        if xp is np:
            from scipy.ndimage import zoom
        else:
            from cupyx.scipy.ndimage import zoom
        unwrapped_phase = xp.array(
            [with_numpy(unwrap_phase)(smap) for smap in xp.angle(Smaps)],
            dtype=xp.float32,
        )
        abs_maps = zoom(abs(Smaps), (1,) + (decim,) * (Smaps.ndim - 1), order=1)
        # Phase zoom with 0 order to prevent residual unwrapping causing artifacts
        angle_maps = zoom(unwrapped_phase, (1,) + (decim,) * (Smaps.ndim - 1), order=0)
        max_eig = zoom(max_eig.T[0], (1,) + (decim,) * (Smaps.ndim - 1), order=1)
        Smaps = abs_maps * np.exp(1j * angle_maps)
    Smaps *= max_eig > crop
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
