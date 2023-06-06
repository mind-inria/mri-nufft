"""Functions to expands 2D trajectories into 3D trajectories."""
import numpy as np

from .utils import (
    KMAX,
    DEFAULT_CONE_ANGLE,
    DEFAULT_HELIX_ANGLE,
    Rx,
    Ry,
    Rz,
    initialize_tilt,
)


########################
# TRAJECTORY EXPANSION #
########################


def stack_2D_to_3D_expansion(trajectory, nb_repetitions, tilt="intergaps"):
    """Stack 2D trajectories into 3D trajectories.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to expand.
    nb_repetitions : int
        Number of repetitions.
    tilt : str, optional
        Tilt of the planes, by default "intergaps".

    Returns
    -------
    array_like
        Expanded trajectory.
    """
    # Initialize output and z-axis
    Nc, Ns, trajectory = *(trajectory.shape[:2]), trajectory.reshape((-1, 2))
    new_trajectory = np.zeros((nb_repetitions, trajectory.shape[0], 3))
    z_axis = np.linspace(-KMAX, KMAX, nb_repetitions)
    delta_tilt = initialize_tilt(tilt, nb_repetitions)

    # Start stacking the planes
    new_trajectory[0, :, :2] = trajectory
    new_trajectory[0, :, 2] = z_axis[0]
    for i in range(1, nb_repetitions):
        angle_tilt = i * delta_tilt
        rotation_tilt = Rz(angle_tilt)
        new_trajectory[i] = new_trajectory[0] @ rotation_tilt
        new_trajectory[i, :, 2] = z_axis[i]
    return new_trajectory.reshape(nb_repetitions * Nc, Ns, 3)


def rotate_2D_to_3D_expansion(trajectory, nb_repetitions, tilt="intergaps"):
    """Rotate 2D trajectories into 3D trajectories.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to expand.
    nb_repetitions : int
        Number of repetitions.
    tilt : str, optional
        Tilt of the planes, by default "intergaps".

    Returns
    -------
    array_like
        Expanded trajectory.
    """
    # Initialize angle and output
    Nc, Ns, trajectory = *(trajectory.shape[:2]), trajectory.reshape((-1, 2))
    trajectory = trajectory.reshape((-1, 2))
    delta_angle = 2 * np.pi / nb_repetitions / 2
    new_trajectory = np.zeros((nb_repetitions, trajectory.shape[0], 3))
    delta_tilt = initialize_tilt(tilt, nb_repetitions)

    # Start rotating the planes
    new_trajectory[0, :, :2] = trajectory
    for i in range(1, nb_repetitions):
        angle_rep = i * delta_angle
        rotation_rep = Rx(angle_rep)
        angle_tilt = i * delta_tilt
        rotation_tilt = Ry(angle_tilt) @ Rz(angle_tilt)
        new_trajectory[i] = new_trajectory[0] @ (rotation_tilt @ rotation_rep)
    return new_trajectory.reshape(nb_repetitions * Nc, Ns, 3)


def cone_2D_to_3D_expansion(
    trajectory,
    nb_repetitions,
    tilt="intergaps",
    in_out=False,
    max_angle=DEFAULT_CONE_ANGLE,
):
    """Expand 2D trajectories into 3D trajectories using a cone expansion.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to expand.
    nb_repetitions : int
        Number of repetitions.
    tilt : str, optional
        Tilt of the planes, by default "intergaps".
    in_out : bool, optional
        Whether to expand in and out, by default False.
    max_angle : float, optional
        Maximum angle of the cone, by default DEFAULT_CONE_ANGLE.

    Returns
    -------
    array_like
        Expanded trajectory.
    """
    # Initialize angles
    Nc, Ns = trajectory.shape[:2]
    delta_tilt = initialize_tilt(tilt, nb_repetitions)
    alphas = np.linspace(
        -max_angle if (not in_out) else 0,
        +max_angle,
        nb_repetitions // (1 + in_out) + 2,
    )[
        1:-1
    ]  # Avoid 0 angles

    # Start processing the trajectory
    new_trajectory = np.zeros((nb_repetitions, Nc, Ns, 3))
    new_trajectory[..., :2] = trajectory[None]
    new_trajectory = new_trajectory.reshape((nb_repetitions, Nc * Ns, 3))
    for i, alpha in enumerate(alphas):
        # Apply tilt
        angle_tilt = i * delta_tilt
        rotation_tilt = Rz(angle_tilt)
        new_trajectory[i] = new_trajectory[i] @ rotation_tilt

        # Apply cone expansion
        new_trajectory[i, :, 2] = np.sin(alpha) * np.linalg.norm(
            new_trajectory[i, :, :2], axis=-1
        )
        new_trajectory[i, :, 0] = np.cos(alpha) * new_trajectory[i, :, 0]
        new_trajectory[i, :, 1] = np.cos(alpha) * new_trajectory[i, :, 1]

    # Adjust planes to be continuous if in-out
    if in_out:
        # Flatten planes
        new_trajectory = new_trajectory.reshape((nb_repetitions, Nc, 2, Ns // 2, 3))
        even, odd = new_trajectory[..., 0, :, :], new_trajectory[..., 1, :, :]
        odd[..., 2] = -odd[..., 2]
        new_trajectory = np.stack([even, odd], axis=2)
        # Duplicate planes
        fhalf = nb_repetitions // 2
        shalf = nb_repetitions - fhalf  # Handle odd repetition numbers
        new_trajectory[shalf:] = new_trajectory[:fhalf]
        new_trajectory[shalf:, ..., 2] = -new_trajectory[:fhalf, ..., 2]
    return new_trajectory.reshape(nb_repetitions * Nc, Ns, 3)


def helix_2D_to_3D_expansion(
    trajectory, nb_repetitions, in_out=False, max_angle=DEFAULT_HELIX_ANGLE
):
    """
    Expand 2D trajectories into 3D trajectories using a helix expansion.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to expand.
    nb_repetitions : int
        Number of repetitions.
    in_out : bool, optional
        Whether to expand in and out, by default False.
    max_angle : float, optional
        Maximum angle of the helix, by default DEFAULT_HELIX_ANGLE.

    Returns
    -------
    array_like
        Expanded trajectory.
    """
    # TODO: fix max_angle
    # Initialize angles and radius
    Nc, Ns = trajectory.shape[:2]
    alphas = (
        max_angle * np.linspace(0, 1, nb_repetitions * Nc + 2)[1:-1]
    )  # Avoid 0 angles
    radius = np.linspace(-1 if (in_out) else 0, 1, Ns)

    # Start processing the trajectory
    new_trajectory = np.zeros((nb_repetitions * Nc, Ns, 3))
    for i, alpha in enumerate(alphas):
        new_trajectory[i, :, 0] = radius * trajectory[i % Nc, :, 0] * np.sin(alpha)
        new_trajectory[i, :, 1] = radius * trajectory[i % Nc, :, 1] * np.sin(alpha)
        new_trajectory[i, :, 2] = KMAX * radius * np.cos(alpha)

    # Adjust shots to be continuous if in-out
    if in_out:
        # Rotate second halves
        new_trajectory = new_trajectory.reshape((nb_repetitions, Nc, 2, Ns // 2, 3))
        even, odd = new_trajectory[..., 0, :, :], new_trajectory[..., 1, :, :]
        odd = odd.reshape((-1, 3)) @ Rz(np.pi)
        new_trajectory = np.stack([even, odd.reshape(even.shape)], axis=2)
    return new_trajectory
