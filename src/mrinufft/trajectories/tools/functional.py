"""Functional trajectory tools: stack_spherically, shellify."""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.maths import Rz
from mrinufft.trajectories.utils import KMAX, initialize_tilt


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
        stk = trajectory_func(Nc=Nc_per_stack[i], **traj_kwargs)
        if stk.shape[-1] == 2:
            stk = np.concatenate([stk, np.zeros((*(stk.shape[:2]), 1))], axis=-1)
        stk[..., :2] = radii[i] * stk[..., :2]
        stk[..., 2] = z_axis[i] + stk[..., 2]

        # Apply z tilt
        rotation = Rz(i * initialize_tilt(z_tilt, nb_stacks)).T
        stk = stk @ rotation
        new_trajectory.append(stk)

    # Concatenate or handle varying Ns value
    Ns_values = np.array([s.shape[1] for s in new_trajectory])
    if (Ns_values == Ns_values[0]).all():
        output = np.concatenate(new_trajectory, axis=0)
        return output.reshape(Nc, Ns_values[0], 3)
    return np.concatenate([s.reshape((-1, 3)) for s in new_trajectory], axis=0)


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
