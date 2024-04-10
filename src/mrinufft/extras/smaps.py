from mrinufft._utils import MethodRegister
from mrinufft.density.utils import flat_traj
from mrinufft.operators.base import with_numpy_cupy, get_array_module


register_smaps = MethodRegister("sensitivity_maps")


@flat_traj
def _get_centeral_index(kspace_loc, threshold):
    r"""
    Extract the index of the k-space center.
    
    Parameters
    ----------
    kspace_loc: numpy.ndarray
        The samples location in the k-sapec domain (between [-0.5, 0.5[)
    threshold: tuple or float
        The threshold used to extract the k_space center (between (0, 1])
    
    Returns
    -------
    The index of the k-space center.
    """
    xp = get_array_module(kspace_loc)
    radius = xp.linalg.norm(kspace_loc, axis=-1)
    
    if isinstance(threshold, float):
        threshold = (threshold,) * kspace_loc.shape[-1]
    condition = xp.logical_and.reduce(tuple(
        xp.abs(kspace_loc[:, i]) <= threshold[i] for i in range(len(threshold))
    ))
    index = xp.linspace(0, kspace_loc.shape[0] - 1, kspace_loc.shape[0], dtype=xp.int64)
    index = xp.extract(condition, index)
    return index

def extract_k_space_center_and_locations(
        kspace_data, kspace_loc, threshold=None, window_fun=None, 
    ):
    r"""
    Extract k-space center and corresponding sampling locations.
    
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
    The extracted center of the k-space, i.e. both the kspace locations and
    kspace values. If the density compensators are passed, the corresponding
    compensators for the center of k-space data will also be returned. The
    return stypes for density compensation and kspace data is same as input

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
    radius = xp.linalg.norm(center_locations, axis=1)
    data_ordered = xp.copy(kspace_data)
    if isinstance(threshold, float):
        threshold = (threshold,) * kspace_loc.shape[1]
    condition = xp.logical_and.reduce(tuple(
        xp.abs(kspace_loc[:, i]) <= threshold[i] for i in range(len(threshold))
    ))
    index = xp.linspace(0, kspace_loc.shape[0] - 1, kspace_loc.shape[0], dtype=xp.int64)
    index = xp.extract(condition, index)
    center_locations = kspace_loc[index, :]
    data_thresholded = data_ordered[:, index]
    if window_fun is not None:
        if callable(window_fun):
            window = window_fun(center_locations)
        else:
            if window_fun == "Hann" or window_fun == "Hanning":
                a_0 = 0.5
            elif window_fun == "Hamming":
                a_0 = 0.53836
            else:
                raise ValueError("Unsupported window function.")
            
            window = a_0 + (1 - a_0) * xp.cos(xp.pi * radius / threshold)
        data_thresholded = window * data_thresholded

    if density_comp is not None:
        density_comp = density_comp[index]
        return data_thresholded, center_locations, density_comp
    else:
        return data_thresholded, center_locations


@register_smaps
@with_numpy_cupy   
@flat_traj
def low_frequency(traj, kspace_data, shape, backend, theshold, *args, **kwargs):
    xp = get_array_module(kspace_data)
    k_space, samples, dc = extract_k_space_center_and_locations(
        kspace_data=kspace_data,
        kspace_loc=traj,
        threshold=threshold,
        img_shape=traj_params['img_size'],
    )
    smaps_adj_op = get_operator('gpunufft')(
        samples,
        shape,
        density=dc,
        n_coils=k_space.shape[0]
    )
    Smaps_ = smaps_adj_op.adj_op(k_space)
    SOS = xp.linalg.norm(Smaps_ , axis=0)
    thresh = threshold_otsu(SOS)
    convex_hull = convex_hull_image(SOS>thresh)
    
    