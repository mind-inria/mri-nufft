"""Functions to manipulate/modify trajectories."""

from typing import Any, Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline, interp1d
from scipy.stats import norm

from .maths import Rv, Rx, Ry, Rz
from .utils import KMAX, initialize_tilt, VDSpdf, VDSorder

################
# DIRECT TOOLS #
################


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
    Nc, Ns, Nd = trajectory.shape
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


####################
# FUNCTIONAL TOOLS #
####################


def stack_spherically(
    trajectory_func: Callable[..., NDArray],
    Nc: int,
    nb_stacks: int,
    z_tilt: str | float | None = None,
    hard_bounded: bool = True,
    **traj_kwargs: Any,  # noqa ANN401
) -> NDArray:
    """Stack 2D or 3D trajectories over the :math:`k_z`-axis to make a sphere.

    Parameters
    ----------
    trajectory_func : Callable[..., NDArray]
        Trajectory function that should return an array-like
        with the usual (Nc, Ns, Nd) size.
    Nc : int
        Number of shots to use for the whole spherically stacked trajectory.
    nb_stacks : int
        Number of stacks of trajectories.
    z_tilt : str | float, optional
        Tilt of the stacks, by default `None`.
    hard_bounded : bool, optional
        Whether the stacks should be strictly within the limits
        of the k-space, by default `True`.
    **traj_kwargs : Any
        Trajectory initialization parameters for the function
        provided with `trajectory_func`.

    Returns
    -------
    NDArray
        Stacked trajectory.
    """
    # Handle argument errors
    if Nc < nb_stacks:
        raise ValueError("Nc should be higher than nb_stacks.")

    # Initialize a plane to estimate potential thickness
    trajectory = trajectory_func(Nc=Nc // nb_stacks, **traj_kwargs)
    if trajectory.shape[-1] == 2:
        trajectory = np.concatenate(
            [trajectory, np.zeros((*(trajectory.shape[:2]), 1))], axis=-1
        )

    # Initialize z-axis with boundaries, and z-rotation
    ub, lb = KMAX / nb_stacks, -KMAX / nb_stacks
    if hard_bounded:
        ub = max(np.max(trajectory[..., 2]), ub)
        lb = min(np.min(trajectory[..., 2]), lb)
    z_axis = np.linspace(-KMAX - lb, KMAX - ub, nb_stacks)
    radii = np.cos(np.arcsin(z_axis / KMAX))

    # Attribute shots to stacks following density proportional to surface
    Nc_per_stack = np.ones(nb_stacks).astype(int)
    density = radii**2  # simplified version
    for _ in range(Nc - nb_stacks):
        idx = np.argmax(density / Nc_per_stack)
        Nc_per_stack[idx] += 1

    # Start stacking the trajectories
    new_trajectory = []
    for i in range(nb_stacks):
        # Initialize a single stack
        stack = trajectory_func(Nc=Nc_per_stack[i], **traj_kwargs)
        if stack.shape[-1] == 2:
            stack = np.concatenate([stack, np.zeros((*(stack.shape[:2]), 1))], axis=-1)
        stack[..., :2] = radii[i] * stack[..., :2]
        stack[..., 2] = z_axis[i] + stack[..., 2]

        # Apply z tilt
        rotation = Rz(i * initialize_tilt(z_tilt, nb_stacks)).T
        stack = stack @ rotation
        new_trajectory.append(stack)

    # Concatenate or handle varying Ns value
    Ns_values = np.array([stk.shape[1] for stk in new_trajectory])
    if (Ns_values == Ns_values[0]).all():
        output = np.concatenate(new_trajectory, axis=0)
        return output.reshape(Nc, Ns_values[0], 3)
    return np.concatenate([stk.reshape((-1, 3)) for stk in new_trajectory], axis=0)


def shellify(
    trajectory_func: Callable[..., NDArray],
    Nc: int,
    nb_shells: int,
    z_tilt: str | float = "golden",
    hemisphere_mode: Literal["symmetric", "reversed"] = "symmetric",
    **traj_kwargs: Any,  # noqa ANN401
) -> NDArray:
    """Stack 2D or 3D trajectories over the :math:`k_z`-axis to make a sphere.

    Parameters
    ----------
    trajectory_func : Callable[..., NDArray]
        Trajectory function that should return an array-like with the usual
        (Nc, Ns, Nd) size.
    Nc : int
        Number of shots to use for the whole spherically stacked trajectory.
    nb_shells : int
        Number of shells of distorted trajectories.
    z_tilt : str | float, optional
        Tilt of the shells, by default "golden".
    hemisphere_mode : Literal["symmetric", "reversed"], optional
        Define how the lower hemisphere should be oriented relatively to the
        upper one, with "symmetric" providing a :math:`k_x-k_y` planar symmetry
        by changing the polar angle, and with "reversed" promoting continuity
        (for example in spirals) by reversing the azimuth angle.
        The default is "symmetric".
    **traj_kwargs : Any
        Trajectory initialization parameters for the function
        provided with `trajectory_func`.

    Returns
    -------
    NDArray
        Concentric shell trajectory.
    """
    # Handle argument errors
    if hemisphere_mode not in ["symmetric", "reversed"]:
        raise ValueError(f"Unknown hemisphere_mode: `{hemisphere_mode}`.")
    if Nc < 2 * nb_shells:
        raise ValueError("Nc should be at least twice higher than nb_shells.")

    # Attribute shots to shells following a prescribed density
    Nc_per_shell = np.ones(nb_shells).astype(int)
    density = np.arange(1, nb_shells + 1) ** 2  # simplified version
    for _ in range((Nc - 2 * nb_shells) // 2):
        idx = np.argmax(density / Nc_per_shell)
        Nc_per_shell[idx] += 1

    # Create shells one by one
    radii = (0.5 + np.arange(nb_shells)) / nb_shells
    new_trajectory = []
    for i in range(nb_shells):
        # Initialize trajectory
        shell_upper = trajectory_func(Nc=Nc_per_shell[i], **traj_kwargs)
        if shell_upper.shape[-1] < 3:
            shell_upper = np.concatenate(
                [shell_upper, np.zeros((*(shell_upper.shape[:-1]), 1))], axis=-1
            )

        # Carve upper hemisphere from trajectory
        z_coords = KMAX**2 - shell_upper[..., 0] ** 2 - shell_upper[..., 1] ** 2
        z_signs = np.sign(z_coords)
        shell_upper[..., 2] += z_signs * np.sqrt(np.abs(z_coords))

        # Initialize lower hemisphere from upper
        shell_lower = np.copy(shell_upper)
        if hemisphere_mode in ["symmetric", "reversed"]:
            shell_lower[..., 2] = -shell_lower[..., :, 2]  # Invert polar angle
        if hemisphere_mode in ["reversed"]:
            shell_lower[..., 1] = -shell_lower[..., :, 1]  # Invert azimuthal angle

        # Apply shell tilt
        rotation = Rz(i * initialize_tilt(z_tilt, nb_shells)).T
        shell_upper = shell_upper @ rotation
        shell_lower = shell_lower @ rotation

        # Scale them and add them to the trajectory
        new_trajectory.append(radii[i] * shell_upper)
        new_trajectory.append(radii[i] * shell_lower)

    # Concatenate or handle varying Ns value
    Ns_values = np.array([hem.shape[1] for hem in new_trajectory])
    if (Ns_values == Ns_values[0]).all():
        output = np.concatenate(new_trajectory, axis=0)
        return output.reshape(Nc, Ns_values[0], 3)
    return np.concatenate([hem.reshape((-1, 3)) for hem in new_trajectory], axis=0)


#########
# UTILS #
#########


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


#################
# Randomization #
#################


def _flip2center(mask_cols: list[int], center_value: int) -> np.ndarray:
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


def get_random_loc_1d(
    dim_size: int,
    center_prop: float | int,
    accel: float = 4,
    pdf: Literal["uniform", "gaussian", "equispaced"] | NDArray = "uniform",
    rng: int | np.random.Generator | None = None,
    order: Literal["center-out", "top-down", "random"] = "center-out",
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

    def _get_samples(p: np.typing.ArrayLike) -> list[int]:
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
