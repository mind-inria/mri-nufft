"""Random trajectory tools: get_random_loc_1d, stack_random."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from mrinufft.trajectories.tools.transforms import _flip2center
from mrinufft.trajectories.utils import KMAX, VDSorder, VDSpdf


def get_random_loc_1d(
    dim_size: int,
    center_prop: float | int,
    accel: float = 4,
    pdf: Literal["uniform", "gaussian", "equispaced"] | NDArray | VDSpdf = "uniform",
    rng: int | np.random.Generator | None = None,
    order: Literal["center-out", "top-down", "random"] | VDSorder = "center-out",
) -> NDArray:
    """Get slice index at a random position.

    Parameters
    ----------
    dim_size: int
        Dimension size
    center_prop: float or int
        Proportion of center of kspace to continuouly sample
    accel: float
        Undersampling/Acceleration factor
    pdf: str, optional
        Probability density function for the remaining samples.
        "gaussian" (default) or "uniform" or np.array
    rng: int or np.random.Generator
        random state
    order: str
        Order of the lines, "center-out" (default), "random" or "top-down"

    Returns
    -------
    np.ndarray: array of size dim_size/accel.
    """
    order = VDSorder(order)
    pdf = VDSpdf(pdf) if isinstance(pdf, str) else pdf
    if accel == 0 or accel == 1:
        return np.arange(dim_size)  # type: ignore
    elif accel < 0:
        raise ValueError("acceleration factor should be positive.")
    elif isinstance(accel, float):
        raise ValueError("acceleration factor should be an integer.")

    indexes = list(range(dim_size))

    if not isinstance(center_prop, int):
        center_prop = int(center_prop * dim_size)

    center_start = (dim_size - center_prop) // 2
    center_stop = (dim_size + center_prop) // 2
    center_indexes = indexes[center_start:center_stop]
    borders = np.asarray([*indexes[:center_start], *indexes[center_stop:]])

    n_samples_borders = (dim_size - len(center_indexes)) // accel
    if n_samples_borders < 1:
        raise ValueError(
            "acceleration factor, center_prop and dimension not compatible."
            "Edges will not be sampled. "
        )
    rng = np.random.default_rng(rng)  # get RNG from a seed or existing rng.

    def _get_samples(p: NDArray) -> list:
        p = p / np.sum(p)  # automatic casting if needed
        return list(rng.choice(borders, size=n_samples_borders, replace=False, p=p))

    if isinstance(pdf, np.ndarray):
        if len(pdf) == dim_size:
            # extract the borders
            p = pdf[borders]
        elif len(pdf) == len(borders):
            p = pdf
        else:
            raise ValueError("Invalid size for probability.")
        sampled_in_border = _get_samples(p)

    elif pdf == VDSpdf.GAUSSIAN:
        p = norm.pdf(np.linspace(norm.ppf(0.001), norm.ppf(0.999), len(borders)))
        sampled_in_border = _get_samples(p)
    elif pdf == VDSpdf.UNIFORM:
        p = np.ones(len(borders))
        sampled_in_border = _get_samples(p)
    elif pdf == VDSpdf.EQUISPACED:
        sampled_in_border = list(borders[::accel])

    else:
        raise ValueError("Unsupported value for pdf use any of . ")
        # TODO: allow custom pdf as argument (vector or function.)

    line_locs = np.array(sorted(center_indexes + sampled_in_border))
    # apply order of lines
    if order == VDSorder.CENTER_OUT:
        line_locs = _flip2center(sorted(line_locs), dim_size // 2)
    elif order == VDSorder.RANDOM:
        line_locs = rng.permutation(line_locs)
    elif order == VDSorder.TOP_DOWN:
        line_locs = np.array(sorted(line_locs))
    else:
        raise ValueError(f"Unknown direction '{order}'.")
    return (line_locs / dim_size) * 2 * KMAX - KMAX  # rescale to [-0.5,0.5]


def stack_random(
    trajectory: NDArray,
    dim_size: int,
    center_prop: float | int = 0.0,
    accel: float | int = 4,
    pdf: Literal["uniform", "gaussian", "equispaced"] | NDArray = "uniform",
    rng: int | np.random.Generator | None = None,
    order: Literal["center-out", "top-down", "random"] = "center-out",
):
    """Stack a 2D trajectory with random location.

    Parameters
    ----------
    traj: np.ndarray
        Existing 2D trajectory.
    dim_size: int
        Size of the k_z dimension
    center_prop: int or float
        Number of line or proportion of slice to sample in the center of the k-space
    accel: int
        Undersampling/Acceleration factor
    pdf: str or np.array
        Probability density function for the remaining samples.
        "uniform" (default), "gaussian" or np.array
    rng: random state
    order: str
        Order of the lines, "center-out" (default), "random" or "top-down"

    Returns
    -------
    numpy.ndarray
        The 3D trajectory stacked along the :math:`k_z` axis.
    """
    line_locs = get_random_loc_1d(dim_size, center_prop, accel, pdf, rng, order)
    if len(trajectory.shape) == 2:
        Nc, Ns = 1, trajectory.shape[0]
    else:
        Nc, Ns = trajectory.shape[:2]

    new_trajectory = np.zeros((len(line_locs), Nc, Ns, 3))
    for i, loc in enumerate(line_locs):
        new_trajectory[i, :, :, :2] = trajectory[..., :2]
        if trajectory.shape[-1] == 3:
            new_trajectory[i, :, :, 2] = trajectory[..., 2] + loc
        else:
            new_trajectory[i, :, :, 2] = loc

    return new_trajectory.reshape(-1, Ns, 3)
