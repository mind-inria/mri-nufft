"""Functions to manipulate/modify trajectories."""

import numpy as np
from scipy.interpolate import CubicSpline

from .maths import Rv, Rx, Ry, Rz
from .utils import KMAX, initialize_tilt

################
# DIRECT TOOLS #
################


def stack(trajectory, nb_stacks, z_tilt=None, hard_bounded=True):
    """Stack 2D or 3D trajectories over the :math:`k_z`-axis.

    Parameters
    ----------
    trajectory : array_like
        Trajectory in 2D or 3D to stack.
    nb_stacks : int
        Number of stacks repeating the provided trajectory.
    z_tilt : str, optional
        Tilt of the stacks, by default `None`.
    hard_bounded : bool, optional
        Whether the stacks should be strictly within the limits of the k-space.

    Returns
    -------
    array_like
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


def rotate(trajectory, nb_rotations, x_tilt=None, y_tilt=None, z_tilt=None):
    """Rotate 2D or 3D trajectories over the different axes.

    Parameters
    ----------
    trajectory : array_like
        Trajectory in 2D or 3D to rotate.
    nb_rotations : int
        Number of rotations repeating the provided trajectory.
    x_tilt : str, optional
        Tilt of the trajectory over the :math:`k_x`-axis, by default `None`.
    y_tilt : str, optional
        Tilt of the trajectory over the :math:`k_y`-axis, by default `None`.
    z_tilt : str, optional
        Tilt of the trajectory over the :math:`k_z`-axis, by default `None`.

    Returns
    -------
    array_like
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
    trajectory,
    nb_rotations,
    tilt="golden",
    half_sphere=False,
    partition="axial",
    axis=None,
):
    """Rotate trajectories as a precession around the :math:`k_z`-axis.

    Parameters
    ----------
    trajectory : array_like
        Trajectory in 2D or 3D to rotate.
    nb_rotations : int
        Number of rotations repeating the provided trajectory while precessing.
    tilt : str, optional
        Angle tilt between consecutive rotations around the :math:`k_z`-axis,
        by default "golden".
    half_sphere : bool, optional
        Whether the precession should be limited to the upper half
        of the k-space sphere.
        It is typically used for in-out trajectories or planes.
    partition : str, optional
        Partition type between an "axial" or "polar" split of the
        :math:`k_z`-axis, designating whether the axis should be fragmented
        by radius or angle respectively, by default "axial".
    axis : int, array_like, optional
        Axis selected for alignment reference when rotating the trajectory
        around the :math:`k_z`-axis, generally corresponding to the shot
        direction for single shot ``trajectory`` inputs. It can either
        be an integer for one of the three k-space axes, or directly a 3D
        array. The default behavior when `None` is to select the last
        coordinate of the first shot as the axis, by default `None`.

    Returns
    -------
    array_like
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
    trajectory,
    nb_cones,
    z_tilt=None,
    in_out=False,
    max_angle=np.pi / 2,
    borderless=True,
):
    """Distort 2D or 3D trajectories into cones along the :math:`k_z`-axis.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to conify.
    nb_cones : int
        Number of cones repeating the provided trajectory.
    z_tilt : str, optional
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
    array_like
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


def epify(trajectory, Nc_trains, Ns_transitions, reverse_odd_shots=False):
    """Create multi-readout shots from trajectory composed of single-readouts.

    Assemble multiple single-readout shots together by adding transition
    steps in the trajectory to create EPI-like multi-readout shots.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to change by prolonging and merging the shots.
    Nc_trains : int
        Number of resulting multi-readout shots, or trains.
    Ns_transitions : int
        Number of samples/steps between the merged readouts.
    reverse_odd_shots : bool, optional
        Whether to reverse every odd shots such that, as in most
        trajectories, even shots end up closer to the start of odd
        shots.

    Returns
    -------
    array_like
        Trajectory with fewer but longer multi-readout shots.
    """
    Nc, Ns, Nd = trajectory.shape
    if Nc % Nc_trains != 0:
        raise ValueError(
            "`Nc_trains` should divide the number of shots in `trajectory`."
        )
    nb_shot_per_train = Nc // Nc_trains

    # Reverse odd shots to facilitate concatenation if requested
    trajectory = np.copy(trajectory)
    trajectory = trajectory.reshape((Nc_trains, -1, Ns, Nd))
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

    for i_c in range(Nc_trains):
        spline = CubicSpline(source_sample_ids, np.concatenate(trajectory[i_c], axis=0))
        assembled_trajectory.append(spline(target_sample_ids))
    assembled_trajectory = np.array(assembled_trajectory)

    return assembled_trajectory


def unepify(trajectory, Ns_readouts, Ns_transitions):
    """Recover single-readout shots from multi-readout trajectory.

    Reformat an EPI-like trajectory with multiple readouts and transitions
    to more single-readout shots by discarding the transition parts.

    Note that it can also be applied to any array of shape
    (Nc, Ns_readouts + Ns_transitions, ...) such as acquired samples
    for example.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to reduce by discarding transitions between readouts.
    Ns_readouts : int
        Number of samples within a single readout.
    Ns_transitions : int
        Number of samples/steps between the readouts.

    Returns
    -------
    array_like
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


def prewind(trajectory, Ns_transitions):
    """Add pre-winding/positioning to the trajectory.

    The trajectory is extended to start before the readout
    from the k-space center with null gradients and reach
    each shot position with the required gradient strength.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to extend with rewind gradients.
    Ns_transitions : int
        Number of pre-winding/positioning steps used to leave the
        k-space center and prepare for each shot to start.

    Returns
    -------
    array_like
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
    assembled_trajectory = np.array(assembled_trajectory)

    return assembled_trajectory


def rewind(trajectory, Ns_transitions):
    """Add rewinding to the trajectory.

    The trajectory is extended to come back to the k-space center
    after the readouts with null gradients.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to extend with rewind gradients.
    Ns_transitions : int
        Number of rewinding steps used to come back to the k-space center.

    Returns
    -------
    array_like
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
    assembled_trajectory = np.array(assembled_trajectory)

    return assembled_trajectory


####################
# FUNCTIONAL TOOLS #
####################


def stack_spherically(
    trajectory_func, Nc, nb_stacks, z_tilt=None, hard_bounded=True, **traj_kwargs
):
    """Stack 2D or 3D trajectories over the :math:`k_z`-axis to make a sphere.

    Parameters
    ----------
    trajectory_func : function
        Trajectory function that should return an array-like
        with the usual (Nc, Ns, Nd) size.
    Nc : int
        Number of shots to use for the whole spherically
        stacked trajectory.
    nb_stacks : int
        Number of stacks of trajectories.
    z_tilt : str, optional
        Tilt of the stacks, by default `None`.
    hard_bounded : bool, optional
        Whether the stacks should be strictly within the limits of the k-space,
        by default `True`.
    **kwargs
        Trajectory initialization parameters for the function provided
        with `trajectory_func`.

    Returns
    -------
    array_like
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
        new_trajectory = np.concatenate(new_trajectory, axis=0)
        new_trajectory = new_trajectory.reshape(Nc, Ns_values[0], 3)
    else:
        new_trajectory = np.concatenate(
            [stk.reshape((-1, 3)) for stk in new_trajectory], axis=0
        )

    return new_trajectory


def shellify(
    trajectory_func,
    Nc,
    nb_shells,
    z_tilt="golden",
    hemisphere_mode="symmetric",
    **traj_kwargs,
):
    """Stack 2D or 3D trajectories over the :math:`k_z`-axis to make a sphere.

    Parameters
    ----------
    trajectory_func : function
        Trajectory function that should return an array-like
        with the usual (Nc, Ns, Nd) size.
    Nc : int
        Number of shots to use for the whole spherically
        stacked trajectory.
    nb_shells : int
        Number of shells of distorded trajectories.
    z_tilt : str, float, optional
        Tilt of the shells, by default "golden".
    hemisphere_mode : str, optional
        Define how the lower hemisphere should be oriented
        relatively to the upper one, with "symmetric" providing
        a :math:`k_x-k_y` planar symmetry by changing the polar angle,
        and with "reversed" promoting continuity (for example
        in spirals) by reversing the azimuth angle.
        The default is "symmetric".
    **kwargs
        Trajectory initialization parameters for the function provided
        with `trajectory_func`.

    Returns
    -------
    array_like
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
        new_trajectory = np.concatenate(new_trajectory, axis=0)
        new_trajectory = new_trajectory.reshape(Nc, Ns_values[0], 3)
    else:
        new_trajectory = np.concatenate(
            [hem.reshape((-1, 3)) for hem in new_trajectory], axis=0
        )

    return new_trajectory


#########
# UTILS #
#########


def duplicate_along_axes(trajectory, axes=(0, 1, 2)):
    """
    Duplicate a trajectory along the specified axes.

    The provided trajectories are replicated with different orientation,
    with the :math:`k_x`-axis being considered as the default orientation
    of the base trajectory.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to duplicate.
    axes : tuple, optional
        Axes along which to duplicate the trajectory, by default (0, 1, 2)

    Returns
    -------
    array_like
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
    new_trajectory = np.concatenate(new_trajectory, axis=0)
    return new_trajectory


def _radialize_center_out(trajectory, nb_samples):
    """Radialize a trajectory from the center to the outside."""
    Nc, Ns = trajectory.shape[:2]
    new_trajectory = np.copy(trajectory)
    for i in range(Nc):
        point = trajectory[i, nb_samples]
        new_trajectory[i, :nb_samples, 0] = np.linspace(0, point[0], nb_samples)
        new_trajectory[i, :nb_samples, 1] = np.linspace(0, point[1], nb_samples)
        new_trajectory[i, :nb_samples, 2] = np.linspace(0, point[2], nb_samples)
    return new_trajectory


def _radialize_in_out(trajectory, nb_samples):
    """Radialize a trajectory from the inside to the outside."""
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


def radialize_center(trajectory, nb_samples, in_out=False):
    """Radialize a trajectory.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to radialize.
    nb_samples : int
        Number of samples to keep.
    in_out : bool, optional
        Whether the radialization is from the inside to the outside, by default False
    """
    # Make nb_samples into straight lines around the center
    if in_out:
        return _radialize_in_out(trajectory, nb_samples)
    else:
        return _radialize_center_out(trajectory, nb_samples)
