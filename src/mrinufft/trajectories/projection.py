"""Functions to fit gradient constraints."""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline


def fit_arc_length(
    trajectory: NDArray, order: int | None = None, eps: float = 1e-8
) -> NDArray:
    """Adjust the trajectory to have a uniform arc-length distribution.

    The trajectory is parametrized according to its arc length along a
    cubic-interpolated path and samples are repositioned to minimize
    the gradients amplitude. This solution is optimal with respect to
    gradients but can lead to excessive slew rates, and it will change
    the overall density.

    Parameters
    ----------
    trajectory: NDArray
        A 2D or 3D trajectory of shape (Nc, Ns, Nd), with Nc the number of shots,
        Ns the number of samples per shot, and Nd the number of dimensions.
    order: int | None
        The order of the norm used to compute arc length, based on the convention from
        `numpy.linalg.norm`. Defaults to None (Euclidean norm).
    eps: float
        Convergence threshold for stopping the iterative refinement. Defaults to 1e-8.

    Returns
    -------
    NDArray: The reparameterized trajectory with the same shape as the input.
    """
    Nc, Ns, Nd = trajectory.shape
    new_trajectory = np.copy(trajectory)

    for i in range(Nc):
        time = np.linspace(0, 1, Ns)
        projection = trajectory[i]
        old_projection = 0
        arc_func = CubicSpline(time, projection)

        while (
            np.linalg.norm(projection - old_projection) / np.linalg.norm(projection)
            > eps
        ):
            arc_length = np.cumsum(
                np.linalg.norm(np.diff(projection, axis=0), ord=order, axis=-1), axis=0
            )
            arc_length = np.concatenate([[0], arc_length])
            arc_length = arc_length / arc_length[-1]
            inv_arc_func = CubicSpline(arc_length, time)

            time = inv_arc_func(np.linspace(0, 1, Ns))
            old_projection = np.copy(projection)
            projection = arc_func(time)
        new_trajectory[i] = projection
    return new_trajectory
