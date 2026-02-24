"""Trajectory transform tools: stack, rotate, precess, conify, oversample, etc."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
import numpy.linalg as nl
from scipy.interpolate import interp1d

from mrinufft.trajectories.maths import Rv, Rx, Ry, Rz
from mrinufft.trajectories.utils import (
    KMAX,
    initialize_tilt,
)


def stack(
    trajectory: NDArray,
    nb_stacks: int,
    z_tilt: str | float | None = None,
    *,
    hard_bounded: bool = True,
) -> NDArray:
    """Stack 2D or 3D trajectories over the :math:`k_z`-axis.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory in 2D or 3D to stack.
    nb_stacks : int
        Number of stacks repeating the provided trajectory.
    z_tilt : str | float, optional
        Tilt of the stacks, by default `None`.
    hard_bounded : bool, optional
        Whether the stacks should be strictly within the limits of the k-space.

    Returns
    -------
    NDArray
        Stacked trajectory.
    """
    # Check dimensionality and initialize output
    Nc, Ns = trajectory.shape[:2]
    if trajectory.shape[-1] == 2:
        trajectory = np.concatenate([trajectory, np.zeros((Nc, Ns, 1))], axis=-1)
    trajectory = trajectory.reshape((Nc * Ns, 3))
    new_trajectory = np.zeros((nb_stacks, Nc * Ns, 3))

    # Initialize z-axis with boundaries, and z-rotation
    ub, lb = KMAX / nb_stacks, -KMAX / nb_stacks
    if hard_bounded:
        ub = max(np.max(trajectory[..., 2]), ub)
        lb = min(np.min(trajectory[..., 2]), lb)
    z_axis = np.linspace(-KMAX - lb, KMAX - ub, nb_stacks)
    z_rotation = Rz(initialize_tilt(z_tilt, nb_stacks)).T

    # Start stacking the trajectories
    new_trajectory[0] = trajectory
    new_trajectory[0, :, 2] += z_axis[0]
    for i in range(1, nb_stacks):
        new_trajectory[i] = new_trajectory[i - 1] @ z_rotation
        new_trajectory[i, :, 2] = z_axis[i] + trajectory[..., 2]

    return new_trajectory.reshape(nb_stacks * Nc, Ns, 3)


def rotate(
    trajectory: NDArray,
    nb_rotations: int,
    x_tilt: str | float | None = None,
    y_tilt: str | float | None = None,
    z_tilt: str | float | None = None,
) -> NDArray:
    """Rotate 2D or 3D trajectories over the different axes.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory in 2D or 3D to rotate.
    nb_rotations : int
        Number of rotations repeating the provided trajectory.
    x_tilt : str | float, optional
        Tilt of the trajectory over the :math:`k_x`-axis, by default `None`.
    y_tilt : str | float, optional
        Tilt of the trajectory over the :math:`k_y`-axis, by default `None`.
    z_tilt : str | float, optional
        Tilt of the trajectory over the :math:`k_z`-axis, by default `None`.

    Returns
    -------
    NDArray
        Rotated trajectory.
    """
    # Check dimensionality and initialize output
    Nc, Ns = trajectory.shape[:2]
    if trajectory.shape[-1] == 2:
        trajectory = np.concatenate([trajectory, np.zeros((Nc, Ns, 1))], axis=-1)
    trajectory = trajectory.reshape((Nc * Ns, 3))
    new_trajectory = np.zeros((nb_rotations, Nc * Ns, 3))

    # Start rotating the planes
    x_angle = initialize_tilt(x_tilt, nb_rotations)
    y_angle = initialize_tilt(y_tilt, nb_rotations)
    z_angle = initialize_tilt(z_tilt, nb_rotations)
    new_trajectory[0] = trajectory
    for i in range(1, nb_rotations):
        rotation = (Rx(i * x_angle) @ Ry(i * y_angle) @ Rz(i * z_angle)).T
        new_trajectory[i] = new_trajectory[0] @ rotation

    return new_trajectory.reshape(nb_rotations * Nc, Ns, 3)


def precess(
    trajectory: NDArray,
    nb_rotations: int,
    tilt: str | float = "golden",
    half_sphere: bool = False,
    partition: Literal["axial", "polar"] = "axial",
    axis: int | NDArray | None = None,
) -> NDArray:
    """Rotate trajectories as a precession around the :math:`k_z`-axis.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory in 2D or 3D to rotate.
    nb_rotations : int
        Number of rotations repeating the provided trajectory while precessing.
    tilt : str | float, optional
        Angle tilt between consecutive rotations around the :math:`k_z`-axis,
        by default "golden".
    half_sphere : bool, optional
        Whether the precession should be limited to the upper half
        of the k-space sphere.
        It is typically used for in-out trajectories or planes.
    partition : Literal["axial", "polar"], optional
        Partition type between an "axial" or "polar" split of the
        :math:`k_z`-axis, designating whether the axis should be fragmented
        by radius or angle respectively, by default "axial".
    axis : int, NDArray, optional
        Axis selected for alignment reference when rotating the trajectory
        around the :math:`k_z`-axis, generally corresponding to the shot
        direction for single shot ``trajectory`` inputs. It can either
        be an integer for one of the three k-space axes, or directly a 3D
        array. The default behavior when `None` is to select the last
        coordinate of the first shot as the axis, by default `None`.

    Returns
    -------
    NDArray
        Precessed trajectory.
    """
    # Check for partition option error
    if partition not in ["polar", "axial"]:
        raise NotImplementedError(f"Unknown partition type: {partition}")

    # Check dimensionality and initialize output
    Nc, Ns = trajectory.shape[:2]
    if trajectory.shape[-1] == 2:
        trajectory = np.concatenate([trajectory, np.zeros((Nc, Ns, 1))], axis=-1)
    trajectory = trajectory.reshape((Nc * Ns, 3))
    new_trajectory = np.zeros((nb_rotations, Nc * Ns, 3))

    # Determine direction vectors on a sphere
    vectors = np.zeros((nb_rotations, 3))
    phi = initialize_tilt(tilt, nb_rotations) * np.arange(nb_rotations)
    vectors[:, 2] = np.linspace(-1 + half_sphere, 1, nb_rotations)
    if partition == "polar":
        vectors[:, 2] = np.sin(np.pi / 2 * vectors[:, 2])
    radius = np.sqrt(1 - vectors[:, 2] ** 2)
    vectors[:, 0] = np.cos(phi) * radius
    vectors[:, 1] = np.sin(phi) * radius

    # Select rotation axis when axis is not already a vector
    if axis is None:
        axis_vector = np.copy(trajectory[Ns - 1])
        axis_vector /= np.linalg.norm(axis_vector)
    elif isinstance(axis, int):
        axis_vector = np.zeros(3)
        axis_vector[axis] = 1
    else:
        axis_vector = axis

    # Rotate initial trajectory
    for i in np.arange(nb_rotations):
        rotation = Rv(axis_vector, vectors[i], normalize=False).T
        new_trajectory[i] = trajectory @ rotation

    return new_trajectory.reshape((nb_rotations * Nc, Ns, 3))


def conify(
    trajectory: NDArray,
    nb_cones: int,
    z_tilt: str | float | None = None,
    in_out: bool = False,
    max_angle: float = np.pi / 2,
    borderless: bool = True,
) -> NDArray:
    """Distort 2D or 3D trajectories into cones along the :math:`k_z`-axis.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to conify.
    nb_cones : int
        Number of cones repeating the provided trajectory.
    z_tilt : str | float, optional
        Tilt of the trajectory over the :math:`k_z`-axis, by default `None`.
    in_out : bool, optional
        Whether to account for the in-out nature of some trajectories
        to avoid hard angles around the center, by default False.
    max_angle : float, optional
        Maximum angle of the cones, by default pi / 2.
    borderless : bool, optional
        Whether the cones should reach `max_angle` or not,
        and avoid 1D cones if equal to pi / 2, by default True.

    Returns
    -------
    NDArray
        Conified trajectory.
    """
    # Check dimensionality and initialize output
    Nc, Ns = trajectory.shape[:2]
    if trajectory.shape[-1] == 2:
        trajectory = np.concatenate([trajectory, np.zeros((Nc, Ns, 1))], axis=-1)
    trajectory = trajectory.reshape((Nc * Ns, 3))
    new_trajectory = np.zeros((nb_cones, Nc * Ns, 3))

    # Initialize angles
    z_angle = initialize_tilt(z_tilt, nb_cones)
    alphas = np.linspace(-max_angle, +max_angle, nb_cones + 2 * borderless)
    if borderless:
        alphas = alphas[1:-1]  # Remove partition borders

    # Start processing the trajectory
    new_trajectory[:] = trajectory
    for i, alpha in enumerate(alphas):
        # Apply tilt
        rotation = Rz(np.abs(i - nb_cones // 2) * z_angle).T  # Symmetrical for in-out
        new_trajectory[i] = new_trajectory[i] @ rotation

        # Convert to spherical coordinates
        norms = np.linalg.norm(new_trajectory[i], axis=-1)
        polar_angles = np.arccos(
            new_trajectory[i, ..., 2] / np.where(norms == 0, 1, norms)
        )

        # Conify by changing polar angle
        new_trajectory[i, :, 0] = (
            new_trajectory[i, :, 0]
            / np.sin(polar_angles)
            * np.sin(polar_angles + alpha)
        )
        new_trajectory[i, :, 1] = (
            new_trajectory[i, :, 1]
            / np.sin(polar_angles)
            * np.sin(polar_angles + alpha)
        )
        new_trajectory[i, :, 2] = norms * np.cos(polar_angles + alpha)
    new_trajectory = new_trajectory.reshape(nb_cones * Nc, Ns, 3)

    # Handle in-out trajectories to avoid hard transition at the center
    if in_out:
        new_trajectory[:, Ns // 2 :, 2] = -new_trajectory[:, Ns // 2 :, 2]

    return new_trajectory


def oversample(
    trajectory: NDArray,
    new_Ns: int,
    kind: Literal["linear", "quadratic", "cubic"] = "cubic",
) -> NDArray:
    """
    Resample a trajectory to increase the number of samples using interpolation.

    Parameters
    ----------
    trajectory : NDArray
        The original trajectory array, where interpolation
        is applied along the second axis.
    new_Ns : int
        The desired number of samples in the resampled trajectory.
    kind : Literal, optional
        The type of interpolation to use, such as 'linear',
        'quadratic', or 'cubic', by default "cubic".

    Returns
    -------
    NDArray
        The resampled trajectory array with ``new_Ns`` points along the second axis.

    Notes
    -----
    This function uses ``scipy.interpolate.interp1d`` to perform
    the interpolation along the second axis of the input `trajectory` array.

    Warnings
    --------
    Using 'quadratic' or 'cubic' interpolations is likely to generate
    samples located slightly beyond the original k-space limits by
    making smooth transitions.

    See Also
    --------
    scipy.interpolate.interp1d : The underlying interpolation function
        used for resampling.
    """
    f = interp1d(np.linspace(0, 1, trajectory.shape[1]), trajectory, axis=1, kind=kind)
    return f(np.linspace(0, 1, new_Ns))


def duplicate_along_axes(
    trajectory: NDArray, axes: tuple[int, ...] = (0, 1, 2)
) -> NDArray:
    """
    Duplicate a trajectory along the specified axes.

    The provided trajectories are replicated with different orientation,
    with the :math:`k_x`-axis being considered as the default orientation
    of the base trajectory.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to duplicate.
    axes : tuple[int, ...], optional
        Axes along which to duplicate the trajectory, by default (0, 1, 2)

    Returns
    -------
    NDArray
        Duplicated trajectory along the specified axes.
    """
    # Copy input trajectory along other axes
    new_trajectory = []
    if 0 in axes:
        new_trajectory.append(trajectory)
    if 1 in axes:
        dp_trajectory = np.copy(trajectory)
        dp_trajectory[..., [1, 2]] = dp_trajectory[..., [2, 1]]
        new_trajectory.append(dp_trajectory)
    if 2 in axes:
        dp_trajectory = np.copy(trajectory)
        dp_trajectory[..., [2, 0]] = dp_trajectory[..., [0, 2]]
        new_trajectory.append(dp_trajectory)
    return np.concatenate(new_trajectory, axis=0)


def _radialize_center_out(trajectory: NDArray, nb_samples: int) -> NDArray:
    """Radialize a trajectory from the center to the outside.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to radialize.
    nb_samples : int
        Number of samples to radialize from the center.

    Returns
    -------
    NDArray
        Radialized trajectory.
    """
    Nc, Ns = trajectory.shape[:2]
    new_trajectory = np.copy(trajectory)
    for i in range(Nc):
        point = trajectory[i, nb_samples]
        new_trajectory[i, :nb_samples, 0] = np.linspace(0, point[0], nb_samples)
        new_trajectory[i, :nb_samples, 1] = np.linspace(0, point[1], nb_samples)
        new_trajectory[i, :nb_samples, 2] = np.linspace(0, point[2], nb_samples)
    return new_trajectory


def _radialize_in_out(trajectory: NDArray, nb_samples: int) -> NDArray:
    """Radialize a trajectory from the inside to the outside.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to radialize.
    nb_samples : int
        Number of samples to radialize from the inside out.

    Returns
    -------
    NDArray
        Radialized trajectory.
    """
    Nc, Ns = trajectory.shape[:2]
    new_trajectory = np.copy(trajectory)
    first, half, second = (Ns - nb_samples) // 2, Ns // 2, (Ns + nb_samples) // 2
    for i in range(Nc):
        p1 = trajectory[i, first]
        new_trajectory[i, first:half, 0] = np.linspace(0, p1[0], nb_samples // 2)
        new_trajectory[i, first:half, 1] = np.linspace(0, p1[1], nb_samples // 2)
        new_trajectory[i, first:half, 2] = np.linspace(0, p1[2], nb_samples // 2)
        p2 = trajectory[i, second]
        new_trajectory[i, half:second, 0] = np.linspace(0, p2[0], nb_samples // 2)
        new_trajectory[i, half:second, 1] = np.linspace(0, p2[1], nb_samples // 2)
        new_trajectory[i, half:second, 2] = np.linspace(0, p2[2], nb_samples // 2)
    return new_trajectory


def radialize_center(
    trajectory: NDArray, nb_samples: int, in_out: bool = False
) -> NDArray:
    """Radialize a trajectory.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to radialize.
    nb_samples : int
        Number of samples to keep.
    in_out : bool, optional
        Whether the radialization is from the inside to the outside, by default False

    Returns
    -------
    NDArray
        Radialized trajectory.
    """
    # Make nb_samples into straight lines around the center
    if in_out:
        return _radialize_in_out(trajectory, nb_samples)
    return _radialize_center_out(trajectory, nb_samples)


def _flip2center(mask_cols: list, center_value: int) -> np.ndarray:
    """
    Reorder a list by starting by a center_position and alternating left/right.

    Parameters
    ----------
    mask_cols: list or np.array
        List of columns to reorder.
    center_pos: int
        Position of the center column.

    Returns
    -------
    np.array: reordered columns.
    """
    center_pos = np.argmin(np.abs(np.array(mask_cols) - center_value))
    mask_cols = list(mask_cols)
    left = mask_cols[center_pos::-1]
    right = mask_cols[center_pos + 1 :]
    new_cols = []
    while left or right:
        if left:
            new_cols.append(left.pop(0))
        if right:
            new_cols.append(right.pop(0))
    return np.array(new_cols)
