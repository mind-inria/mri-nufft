"""SMaps module for sensitivity maps estimation."""

from __future__ import annotations

from mrinufft.density.utils import flat_traj
from mrinufft._utils import get_array_module
from .utils import register_smaps
import numpy as np


def _extract_kspace_center(
    kspace_data,
    kspace_loc,
    threshold=None,
    density=None,
    window_fun="ellipse",
):
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
        data_ordered = xp.copy(kspace_data)
        index = xp.linspace(
            0, kspace_loc.shape[0] - 1, kspace_loc.shape[0], dtype=xp.int64
        )
        condition = xp.logical_and.reduce(
            tuple(
                xp.abs(kspace_loc[:, i]) <= threshold[i] for i in range(len(threshold))
            )
        )
        index = xp.extract(condition, index)
        center_locations = kspace_loc[index, :]
        data_thresholded = data_ordered[:, index]
        dc = density[index]
        return data_thresholded, center_locations, dc
    else:
        if callable(window_fun):
            window = window_fun(center_locations)
        else:
            if window_fun in ["hann", "hanning", "hamming"]:
                radius = xp.linalg.norm(kspace_loc, axis=1)
                a_0 = 0.5 if window_fun in ["hann", "hanning"] else 0.53836
                window = a_0 + (1 - a_0) * xp.cos(xp.pi * radius / threshold)
            elif window_fun == "ellipse":
                window = xp.sum(kspace_loc**2 / xp.asarray(threshold) ** 2, axis=1) <= 1
            else:
                raise ValueError("Unsupported window function.")
        data_thresholded = window * kspace_data
        # Return k-space locations & density just for consistency
        return data_thresholded, kspace_loc, density


@register_smaps
@flat_traj
def low_frequency(
    traj,
    shape,
    kspace_data,
    backend,
    threshold: float | tuple[float, ...] = 0.1,
    density=None,
    window_fun: str = "ellipse",
    blurr_factor: int | float | tuple[float, ...] = 0.0,
    mask: bool = False,
):
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
    SOS : numpy.ndarray
        The sum of squares of the sensitivity maps.
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
        samples, shape, density=dc, n_coils=k_space.shape[-2]
    )
    Smaps = smaps_adj_op.adj_op(k_space)
    SOS = np.linalg.norm(Smaps, axis=0)
    if mask:
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
    if mask or np.sum(blurr_factor) > 0:
        # ReCalculate SOS with a minor eps to ensure divide by 0 is ok
        SOS = np.linalg.norm(Smaps, axis=0) + 1e-10
    Smaps = Smaps / SOS
    return Smaps, SOS
