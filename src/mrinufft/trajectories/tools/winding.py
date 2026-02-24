"""Winding/rewinding trajectory tools: prewind, rewind."""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline


def prewind(trajectory: NDArray, Ns_transitions: int) -> NDArray:
    """Add pre-winding/positioning to the trajectory.

    The trajectory is extended to start before the readout
    from the k-space center with null gradients and reach
    each shot position with the required gradient strength.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to extend with rewind gradients.
    Ns_transitions : int
        Number of pre-winding/positioning steps used to leave the
        k-space center and prepare for each shot to start.

    Returns
    -------
    NDArray
        Extended trajectory with pre-winding/positioning.
    """
    Nc, Ns, Nd = trajectory.shape
    if Ns_transitions < 3:
        raise ValueError("`Ns_transitions` should be at least 2.")

    # Assemble shots together per concatenation
    assembled_trajectory = []
    source_sample_ids = np.concatenate([[0, 1], Ns_transitions + np.arange(Ns)])
    target_sample_ids = np.arange(Ns_transitions + Ns)

    for i_c in range(Nc):
        spline = CubicSpline(
            source_sample_ids,
            np.concatenate([np.zeros((2, Nd)), trajectory[i_c]], axis=0),
        )
        assembled_trajectory.append(spline(target_sample_ids))
    return np.array(assembled_trajectory)


def rewind(trajectory: NDArray, Ns_transitions: int) -> NDArray:
    """Add rewinding to the trajectory.

    The trajectory is extended to come back to the k-space center
    after the readouts with null gradients.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to extend with rewind gradients.
    Ns_transitions : int
        Number of rewinding steps used to come back to the k-space center.

    Returns
    -------
    NDArray
        Extended trajectory with rewinding.
    """
    Nc, Ns, Nd = trajectory.shape
    if Ns_transitions < 3:
        raise ValueError("`Ns_transitions` should be at least 2.")

    # Assemble shots together per concatenation
    assembled_trajectory = []
    source_sample_ids = np.concatenate(
        [np.arange(Ns), Ns + Ns_transitions - np.arange(3, 1, -1)]
    )
    target_sample_ids = np.arange(Ns_transitions + Ns)

    for i_c in range(Nc):
        spline = CubicSpline(
            source_sample_ids,
            np.concatenate([trajectory[i_c], np.zeros((2, Nd))], axis=0),
        )
        assembled_trajectory.append(spline(target_sample_ids))
    return np.array(assembled_trajectory)
