from mrinufft.density.utils import flat_traj
from mrinufft.operators.base import get_array_module
from mrinufft import get_operator
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import convex_hull_image
from .utils import register_smaps
import numpy as np


def _extract_kspace_center(
        kspace_data, kspace_loc, threshold=None, density=None, window_fun="ellipse", 
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
        The samples location in the k-sapec domain (between [-0.5, 0.5[)
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
        index = xp.linspace(0, kspace_loc.shape[0] - 1, kspace_loc.shape[0], dtype=xp.int64)
        condition = xp.logical_and.reduce(tuple(
            xp.abs(kspace_loc[:, i]) <= threshold[i] for i in range(len(threshold))
        ))
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
                window = xp.sum(kspace_loc**2/ xp.asarray(threshold)**2, axis=1) <= 1
            else:
                raise ValueError("Unsupported window function.")
        data_thresholded = window * data_thresholded
        # Return k-space locations & density just for consistency
        return data_thresholded, kspace_loc, density


@register_smaps
@flat_traj
def low_frequency(traj, kspace_data, shape, threshold, backend, density=None,
                  extract_kwargs=None, blurr_factor=0):
    """
    Calculate low-frequency sensitivity maps.

    Parameters
    ----------
    traj : numpy.ndarray
        The trajectory of the samples.
    kspace_data : numpy.ndarray
        The k-space data.
    shape : tuple
        The shape of the image.
    threshold : float
        The threshold used for extracting the k-space center.
    backend : str
        The backend used for the operator.
    density : numpy.ndarray, optional
        The density compensation weights.
    extract_kwargs : dict, optional
        Additional keyword arguments for the `extract_kspace_center` function.
    blurr_factor : float, optional
        The blurring factor for smoothing the sensitivity maps.

    Returns
    -------
    Smaps : numpy.ndarray
        The low-frequency sensitivity maps.
    SOS : numpy.ndarray
        The sum of squares of the sensitivity maps.
    """
    k_space, samples, dc = _extract_kspace_center(
        kspace_data=kspace_data,
        kspace_loc=traj,
        threshold=threshold,
        density=density,
        img_shape=shape,
        **(extract_kwargs or {}),
    )
    smaps_adj_op = get_operator(backend)(
        samples,
        shape,
        density=dc,
        n_coils=k_space.shape[0]
    )
    Smaps_ = smaps_adj_op.adj_op(k_space)
    SOS = np.linalg.norm(Smaps_, axis=0)
    thresh = threshold_otsu(SOS)
    convex_hull = convex_hull_image(SOS>thresh)
    Smaps = Smaps_ * convex_hull / SOS
    # Smooth out the sensitivity maps
    if blurr_factor > 0:
        Smaps = gaussian(Smaps, sigma=blurr_factor * np.asarray(shape))
        SOS = np.linalg.norm(Smaps, axis=0)
        Smaps = Smaps / SOS
    return Smaps, SOS
    