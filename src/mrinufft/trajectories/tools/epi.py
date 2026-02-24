"""EPI-related trajectory tools: epify, unepify."""

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline


def epify(
    trajectory: NDArray,
    Ns_transitions: int,
    nb_trains: int,
    *,
    reverse_odd_shots: bool = False,
) -> NDArray:
    """Create multi-readout shots from trajectory composed of single-readouts.

    Assemble multiple single-readout shots together by adding transition
    steps in the trajectory to create EPI-like multi-readout shots.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to change by prolonging and merging the shots.
    Ns_transitions : int
        Number of samples/steps between the merged readouts.
    nb_trains : int
        Number of resulting multi-readout shots, or trains.
    reverse_odd_shots : bool, optional
        Whether to reverse every odd shots such that, as in most
        trajectories, even shots end up closer to the start of odd
        shots.

    Returns
    -------
    NDArray
        Trajectory with fewer but longer multi-readout shots.
    """
    Nc, Ns, Nd = trajectory.shape
    if Nc % nb_trains != 0:
        raise ValueError(
            "`nb_trains` should divide the number of shots in `trajectory`."
        )
    nb_shot_per_train = Nc // nb_trains

    # Reverse odd shots to facilitate concatenation if requested
    trajectory = np.copy(trajectory)
    trajectory = trajectory.reshape((nb_trains, -1, Ns, Nd))
    if reverse_odd_shots:
        trajectory[:, 1::2] = trajectory[:, 1::2, ::-1]

    # Assemble shots together per concatenation
    assembled_trajectory = []
    source_sample_ids = np.concatenate(
        [np.arange(Ns) + i * (Ns_transitions + Ns) for i in range(nb_shot_per_train)]
    )
    target_sample_ids = np.arange(
        nb_shot_per_train * Ns + (nb_shot_per_train - 1) * Ns_transitions
    )

    for i_c in range(nb_trains):
        spline = CubicSpline(source_sample_ids, np.concatenate(trajectory[i_c], axis=0))
        assembled_trajectory.append(spline(target_sample_ids))
    return np.array(assembled_trajectory)


def unepify(trajectory: NDArray, Ns_readouts: int, Ns_transitions: int) -> NDArray:
    """Recover single-readout shots from multi-readout trajectory.

    Reformat an EPI-like trajectory with multiple readouts and transitions
    to more single-readout shots by discarding the transition parts.

    Note that it can also be applied to any array of shape
    (Nc, Ns_readouts + Ns_transitions, ...) such as acquired samples
    for example.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to reduce by discarding transitions between readouts.
    Ns_readouts : int
        Number of samples within a single readout.
    Ns_transitions : int
        Number of samples/steps between the readouts.

    Returns
    -------
    NDArray
        Trajectory with more but shorter single shots.
    """
    _, Ns, Nd = trajectory.shape
    if Ns % (Ns_readouts + Ns_transitions) != Ns_readouts:
        raise ValueError(
            "`trajectory` shape does not match `Ns_readouts` or `Ns_transitions`."
        )

    readout_mask = np.zeros(Ns).astype(bool)
    for i in range(1, Ns // (Ns_readouts + Ns_transitions) + 2):
        readout_mask[
            (i - 1) * Ns_readouts
            + (i - 1) * Ns_transitions : i * Ns_readouts
            + (i - 1) * Ns_transitions
        ] = True
    trajectory = trajectory[:, readout_mask, :]
    trajectory = trajectory.reshape((-1, Ns_readouts, Nd))
    return trajectory
