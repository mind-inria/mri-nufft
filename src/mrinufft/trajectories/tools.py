"""Functions to manipulate trajectories."""
import numpy as np
import numpy.linalg as nl

from .utils import (
    KMAX,
    Rv,
    Rx,
    Ry,
    Rz,
    initialize_tilt,
)

####################
# IMPERATIVE TOOLS #
####################

def stack(trajectory, nb_repetitions, z_tilt="intergaps", hard_bounded=True):
    # Check dimensionality and initialize output
    Nc, Ns = trajectory.shape[:2]
    if (trajectory.shape[-1] == 2):
        trajectory = np.concatenate([trajectory, np.zeros((Nc, Ns, 1))], axis=-1)
    trajectory = trajectory.reshape((Nc * Ns, 3))
    new_trajectory = np.zeros((nb_repetitions, Nc * Ns, 3))

    # Initialize z-axis with boundaries, and z-rotation
    ub, lb = KMAX / nb_repetitions, -KMAX / nb_repetitions
    if (hard_bounded):
        ub = max(np.max(trajectory[..., 2]), ub)
        lb = min(np.min(trajectory[..., 2]), lb)
    z_axis = np.linspace(-KMAX - lb, KMAX - ub, nb_repetitions)
    z_rotation = Rz(initialize_tilt(z_tilt, nb_repetitions)).T

    # Start stacking the trajectories
    new_trajectory[0] = trajectory
    new_trajectory[0, :, 2] += z_axis[0]
    for i in range(1, nb_repetitions):
        new_trajectory[i] = new_trajectory[i - 1] @ z_rotation
        new_trajectory[i, :, 2] = z_axis[i] + trajectory[..., 2]

    return new_trajectory.reshape(nb_repetitions * Nc, Ns, 3)


def rotate(trajectory, nb_repetitions, x_tilt=0, y_tilt=0, z_tilt=0):
    # Check dimensionality and initialize output
    Nc, Ns = trajectory.shape[:2]
    if (trajectory.shape[-1] == 2):
        trajectory = np.concatenate([trajectory, np.zeros((Nc, Ns, 1))], axis=-1)
    trajectory = trajectory.reshape((Nc * Ns, 3))
    new_trajectory = np.zeros((nb_repetitions, Nc * Ns, 3))

    # Start rotating the planes
    x_angle = initialize_tilt(x_tilt, nb_repetitions)
    y_angle = initialize_tilt(y_tilt, nb_repetitions)
    z_angle = initialize_tilt(z_tilt, nb_repetitions)
    new_trajectory[0] = trajectory
    for i in range(1, nb_repetitions):
        rotation = (Rx(i * x_angle) @ Ry(i * y_angle) @ Rz(i * z_angle)).T
        new_trajectory[i] = new_trajectory[0] @ rotation

    return new_trajectory.reshape(nb_repetitions * Nc, Ns, 3)


def precess(trajectory, nb_repetitions, z_tilt="golden", mode="polar", half_sphere=False):
    # Check for requested mode
    if (mode.lower() not in ["axial", "polar"]):
        raise NotImplementedError("Unknown mode name: `{}`.".format(mode))

    # Check dimensionality and initialize output
    Nc, Ns = trajectory.shape[:2]
    if (trajectory.shape[-1] == 2):
        trajectory = np.concatenate([trajectory, np.zeros((Nc, Ns, 1))], axis=-1)
    trajectory = trajectory.reshape((Nc * Ns, 3))
    new_trajectory = np.zeros((nb_repetitions, Nc * Ns, 3))

    # Determine direction vectors on a sphere
    vectors = np.zeros((nb_repetitions, 3))
    phi = initialize_tilt(z_tilt, nb_repetitions) * np.arange(nb_repetitions)
    if (mode.lower() == "axial"):
        vectors[:, 2] = np.linspace(-1 + half_sphere, 1, nb_repetitions)
    else:
        vectors[:, 2] = np.sin(np.pi / 2 * np.linspace(-1 + half_sphere, 1, nb_repetitions))
    radius = np.sqrt(1 - vectors[:, 2] ** 2)
    vectors[:, 0] = np.cos(phi) * radius
    vectors[:, 1] = np.sin(phi) * radius

    # Rotate initial trajectory
    for i in np.arange(nb_repetitions):
        rotation = Rv(np.array((1, 0, 0)), vectors[i], normalize=False).T
        new_trajectory[i] = trajectory @ rotation

    return new_trajectory.reshape((nb_repetitions * Nc, Ns, 3))


def conify(trajectory, nb_repetitions, z_tilt="intergaps", in_out=False, max_angle=np.pi / 2):
    # Check dimensionality and initialize output
    Nc, Ns = trajectory.shape[:2]
    if (trajectory.shape[-1] == 2):
        trajectory = np.concatenate([trajectory, np.zeros((Nc, Ns, 1))], axis=-1)
    trajectory = trajectory.reshape((Nc * Ns, 3))
    new_trajectory = np.zeros((nb_repetitions, Nc * Ns, 3))

    # Initialize angles
    z_tilt = initialize_tilt(z_tilt, nb_repetitions)
    alphas = np.linspace(-max_angle, +max_angle, nb_repetitions + 2)[1:-1]  # Borderless partition

    # Start processing the trajectory
    new_trajectory[:] = trajectory
    for i, alpha in enumerate(alphas):
        # Apply tilt
        rotation = Rz(np.abs((i - nb_repetitions // 2)) * z_tilt).T  # Symmetrical for in-out
        new_trajectory[i] = new_trajectory[i] @ rotation

        # Convert to spherical coordinates
        norms = np.linalg.norm(new_trajectory[i], axis=-1)
        polar_angles = np.arccos(new_trajectory[i, ..., 2] / np.where(norms == 0, 1, norms))

        # Conify by changing polar angle
        new_trajectory[i, :, 0] = new_trajectory[i, :, 0] / np.sin(polar_angles) * np.sin(polar_angles + alpha)
        new_trajectory[i, :, 1] = new_trajectory[i, :, 1] / np.sin(polar_angles) * np.sin(polar_angles + alpha)
        new_trajectory[i, :, 2] = norms * np.cos(polar_angles + alpha)
    new_trajectory = new_trajectory.reshape(nb_repetitions * Nc, Ns, 3)

    # Handle in-out trajectories to avoid hard transition at the center
    if in_out:
        new_trajectory[:, Ns // 2:, 2] = -new_trajectory[:, Ns // 2:, 2]

    return new_trajectory


####################
# FUNCTIONAL TOOLS #
####################

def stack_spherically(trajectory_func, Nc, nb_stacks, z_tilt="golden", hard_bounded=True, **traj_kwargs):
    # Handle argument errors
    if Nc < nb_stacks:
        raise ValueError("Nc should be higher than nb_stacks.")

    # Initialize a plane to estimate potential thickness
    trajectory = trajectory_func(Nc=Nc // nb_stacks, **traj_kwargs)
    if (trajectory.shape[-1] == 2):
        trajectory = np.concatenate([trajectory, np.zeros((*(trajectory.shape[:2]), 1))], axis=-1)

    # Initialize z-axis with boundaries, and z-rotation
    ub, lb = KMAX / nb_stacks, -KMAX / nb_stacks
    if (hard_bounded):
        ub = max(np.max(trajectory[..., 2]), ub)
        lb = min(np.min(trajectory[..., 2]), lb)
    z_axis = np.linspace(-KMAX - lb, KMAX - ub, nb_stacks)
    z_rotation = Rz(initialize_tilt(z_tilt, nb_stacks)).T
    radii = np.cos(np.arcsin(z_axis / KMAX))

    # Attribute shots to stacks following density proportional to surface
    Nc_per_stack = np.ones(nb_stacks).astype(int)
    density = radii ** 2  # simplified version
    for _ in range(Nc - nb_stacks):
        idx = np.argmax(density / Nc_per_stack)
        Nc_per_stack[idx] += 1

    # Start stacking the trajectories
    new_trajectory = []
    for i in range(nb_stacks):
        # Initialize a single stack
        stack = trajectory_func(Nc=Nc_per_stack[i], **traj_kwargs)
        if (stack.shape[-1] == 2):
            stack = np.concatenate([stack, np.zeros((*(stack.shape[:2]), 1))], axis=-1)
        stack[..., :2] = radii[i] * stack[..., :2]
        stack[..., 2] = z_axis[i] + stack[..., 2]

        # Apply z tilt
        rotation = Rz(i * initialize_tilt(z_tilt, nb_stacks))
        stack = stack @ rotation
        new_trajectory.append(stack)

    # Concatenate or handle varying Ns value
    Ns_values = np.array([stk.shape[1] for stk in new_trajectory])
    if (Ns_values == Ns_values[0]).all():
        new_trajectory = np.concatenate(new_trajectory, axis=0)
        new_trajectory = new_trajectory.reshape(Nc, Ns_values[0], 3)
    else:
        new_trajectory = np.concatenate([stk.reshape((-1, 3))
                                         for stk in new_trajectory], axis=0)

    return new_trajectory


def shellify(trajectory_func, Nc, nb_shells, shell_tilt="golden", hemisphere_mode="symmetric", **traj_kwargs):
    # Handle argument errors
    if hemisphere_mode not in ["symmetric", "reversed"]:
        raise ValueError("Unknown hemisphere_mode: `{}`.".format(hemisphere_mode))
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
        # Initialize northern hemisphere
        shell_north = trajectory_func(Nc=Nc_per_shell[i], **traj_kwargs)
        z_coords = KMAX ** 2 - shell_north[..., 0] ** 2 - shell_north[..., 1] ** 2
        z_signs = np.sign(z_coords)
        if (shell_north.shape[-1] < 3):
            shell_north = np.concatenate([shell_north, np.zeros((*(shell_north.shape[:-1]), 1))], axis=-1)
        shell_north[..., 2] += z_signs * np.sqrt(np.abs(z_coords))

        # Initialize southern hemisphere from northern
        shell_south = np.copy(shell_north)
        if hemisphere_mode in ["symmetric", "reversed"]:
            shell_south[..., 2] = -shell_south[..., :, 2]  # Inverse polar angle
        if hemisphere_mode in ["reversed"]:
            shell_south[..., 1] = -shell_south[..., :, 1]  # Inverse azimuthal angle

        # Apply shell tilt
        rotation = Rz(i * initialize_tilt(shell_tilt, nb_shells))
        shell_north = shell_north @ rotation
        shell_south = shell_south @ rotation

        # Scale them and add them to the trajectory
        new_trajectory.append(radii[i] * shell_north)
        new_trajectory.append(radii[i] * shell_south)

    # Concatenate or handle varying Ns value
    Ns_values = np.array([hem.shape[1] for hem in new_trajectory])
    if (Ns_values == Ns_values[0]).all():
        new_trajectory = np.concatenate(new_trajectory, axis=0)
        new_trajectory = new_trajectory.reshape(Nc, Ns_values[0], 3)
    else:
        new_trajectory = np.concatenate([hem.reshape((-1, 3))
                                         for hem in new_trajectory], axis=0)

    return new_trajectory


#########
# UTILS #
#########

def duplicate_along_axes(trajectory, axes=(0, 1, 2)):
    """
    Duplicate a trajectory along the specified axes.

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