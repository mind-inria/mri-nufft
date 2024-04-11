from mrinufft._utils import MethodRegister
from mrinufft.density.utils import flat_traj
from mrinufft.operators.base import with_numpy_cupy, get_array_module


register_smaps = MethodRegister("sensitivity_maps")


def extract_kspace_center(
        kspace_data, kspace_loc, threshold=None, window_fun="ellipse", 
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
    window_fun: "hann" / "hanning", "hamming", "ellipse", "rect", or a callable, 
        default "ellipse".
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
        # Return k-space locations just for consistency
        return data_thresholded, kspace_loc


@register_smaps
@flat_traj
def low_frequency(traj, kspace_data, shape, backend, threshold, *args, **kwargs):
    xp = get_array_module(kspace_data)
    k_space, traj = extract_kspace_center(
        kspace_data=kspace_data,
        kspace_loc=traj,
        threshold=threshold,
        img_shape=shape,
        **kwargs,
    )
    smaps_adj_op = get_operator(backend)(
        samples,
        shape,
        density=dc,
        n_coils=k_space.shape[0]
    )
    Smaps_ = smaps_adj_op.adj_op(k_space)
    SOS = xp.linalg.norm(Smaps_ , axis=0)
    thresh = threshold_otsu(SOS)
    convex_hull = convex_hull_image(SOS>thresh)
    
    