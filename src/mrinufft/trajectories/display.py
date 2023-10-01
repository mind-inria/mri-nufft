"""Display function for trajectories."""
import matplotlib.pyplot as plt
import numpy as np

from .utils import (
    compute_gradients_and_slew_rates,
    KMAX,
    DEFAULT_GMAX,
    DEFAULT_SMAX,
)


COLOR_CYCLE = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:olive",
    "tab:pink",
    "tab:cyan",
]
NB_COLORS = len(COLOR_CYCLE)

LINEWIDTH = 2
POINTSIZE = 10
FONTSIZE = 18


##############
# TICK UTILS #
##############

def _setup_2D_ticks(figsize, ax=None):
    """Add ticks to 2D plot."""
    if ax is None:
        plt.figure(figsize=(figsize, figsize))
        ax = plt.gca()
    ax.grid(True)
    ax.set_xticks([-KMAX, -KMAX / 2, 0, KMAX / 2, KMAX])
    ax.set_yticks([-KMAX, -KMAX / 2, 0, KMAX / 2, KMAX])
    ax.set_xlim((-KMAX, KMAX))
    ax.set_ylim((-KMAX, KMAX))
    ax.set_xlabel("kx", fontsize=FONTSIZE)
    ax.set_ylabel("ky", fontsize=FONTSIZE)
    return ax


def _setup_3D_ticks(figsize, ax=None):
    """Add ticks to 3D plot."""
    if ax is None:
        fig = plt.figure(figsize=(figsize, figsize))
        ax = fig.add_subplot(projection="3d")
    ax.set_xticks([-KMAX, -KMAX / 2, 0, KMAX / 2, KMAX])
    ax.set_yticks([-KMAX, -KMAX / 2, 0, KMAX / 2, KMAX])
    ax.set_zticks([-KMAX, -KMAX / 2, 0, KMAX / 2, KMAX])
    ax.axes.set_xlim3d(left=-KMAX, right=KMAX)
    ax.axes.set_ylim3d(bottom=-KMAX, top=KMAX)
    ax.axes.set_zlim3d(bottom=-KMAX, top=KMAX)
    ax.set_box_aspect((2 * KMAX, 2 * KMAX, 2 * KMAX))
    ax.set_xlabel("kx", fontsize=FONTSIZE)
    ax.set_ylabel("ky", fontsize=FONTSIZE)
    ax.set_zlabel("kz", fontsize=FONTSIZE)
    return ax


######################
# TRAJECTORY DISPLAY #
######################

def display_2D_trajectory(
    trajectory,
    figsize=5,
    one_shot=False,
    subfigure=None,
    show_constraints=False,
    gmax=DEFAULT_GMAX,
    smax=DEFAULT_SMAX,
    constraints_order=None,
    **constraints_kwargs,
):
    # TODO: UPDATE DOCSTRING
    """Display of 2D trajectory.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to display.
    figsize : float, optional
        Size of the figure.
    one_shot : bool, optional
        If True, highlight the trajectory of the middle repetition.
    constraints : bool, optional
        If True, display the points where the gradients and slews are above the
        default limits.
    subfigure: plt.Axes or None, optional
    resolution: Union[float, Tuple[float, ...]], optional
        Resolution of MR image in m.
        The default is DEFAULT_RESOLUTION.

    Returns
    -------
    ax : plt.Axes
        Axes of the figure.
    """
    # Setup figure and ticks
    Nc, Ns = trajectory.shape[:2]
    ax = _setup_2D_ticks(figsize, subfigure)

    # Display every shot
    for i in range(Nc):
        ax.plot(
            trajectory[i, :, 0],
            trajectory[i, :, 1],
            color=COLOR_CYCLE[i % NB_COLORS],
            linewidth=LINEWIDTH,
        )

    # Display one shot in particular if requested
    if one_shot or type(one_shot) == int:
        # Select shot
        shot_id = Nc // 2
        if type(one_shot) == int:
            shot_id = one_shot

        # Highlight the shot in black
        ax.plot(
            trajectory[shot_id, :, 0],
            trajectory[shot_id, :, 1],
            color="k",
            linewidth=2 * LINEWIDTH,
        )

    # Point out violated constraints if requested
    if show_constraints:
        gradients, slews = compute_gradients_and_slew_rates(
            trajectory, **constraints_kwargs)

        # Pad and compute norms
        gradients = np.linalg.norm(np.pad(gradients, ((0, 0), (1, 0), (0, 0))),
                                   axis=-1, ord=constraints_order)
        slews = np.linalg.norm(np.pad(slews, ((0, 0), (2, 0), (0, 0))),
                               axis=-1, ord=constraints_order)

        # Check constraints
        trajectory = trajectory.reshape((-1, 2))
        gradients = trajectory[np.where(gradients.flatten() > gmax)]
        slews = trajectory[np.where(slews.flatten() > smax)]

        # Scatter points with vivid colors
        ax.scatter(gradients[:, 0], gradients[:, 1], color="r", s=POINTSIZE)
        ax.scatter(slews[:, 0], slews[:, 1], color="b", s=POINTSIZE)
    plt.tight_layout()
    return ax


def display_3D_trajectory(
    trajectory,
    nb_repetitions,
    figsize=5,
    per_plane=True,
    one_shot=False,
    subfigure=None,
    show_constraints=False,
    gmax=DEFAULT_GMAX,
    smax=DEFAULT_SMAX,
    constraints_order=None,
    **constraints_kwargs,
):
    # TODO: UPDATE DOCSTRING
    """Display of 3D trajectory.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to display.
    nb_repetitions : int
        Number of repetitions.
    figsize : float, optional
        Size of the figure.
    per_plane : bool, optional
        If True, display the trajectory for each plane.
    one_shot : bool, optional
        If True, highlight the trajectory of the middle repetition.
    constraints : bool, optional
        If True, display the points where the gradients and slews are above the
        default limits.
    subfigure: plt.Axes or None, optional
    resolution: Union[float, Tuple[float, ...]], optional
        Resolution of MR image in m.
        The default is DEFAULT_RESOLUTION.

    Returns
    -------
    ax : plt.Axes
        Axes of the figure.
    """
    # Setup figure and ticks, and handle 2D trajectories
    ax = _setup_3D_ticks(figsize, subfigure)
    if (trajectory.shape[-1] == 2):
        trajectory = np.concatenate([trajectory,
            np.zeros((*(trajectory.shape[:2]), 1))], axis=-1)
    trajectory = trajectory.reshape((nb_repetitions, -1, trajectory.shape[-2], 3))
    Nc, Ns = trajectory.shape[1:3]

    # Display every shot
    for i in range(nb_repetitions):
        for j in range(Nc):
            ax.plot(
                trajectory[i, j, :, 0],
                trajectory[i, j, :, 1],
                trajectory[i, j, :, 2],
                color=COLOR_CYCLE[(i + j * (not per_plane)) % NB_COLORS],
                linewidth=LINEWIDTH,
            )

    # Display one shot in particular if requested
    if one_shot or type(one_shot) == int:
        trajectory = trajectory.reshape((-1, Ns, 3))

        # Select shot
        shot_id = Nc // 2
        if type(one_shot) == int:
            shot_id = one_shot

        # Highlight the shot in black
        ax.plot(
            trajectory[shot_id, :, 0],
            trajectory[shot_id, :, 1],
            trajectory[shot_id, :, 2],
            color="k",
            linewidth=2 * LINEWIDTH,
        )
        trajectory = trajectory.reshape((-1, Nc, Ns, 3))

    # Point out violated constraints if requested
    if show_constraints:
        gradients, slewrates = compute_gradients_and_slew_rates(
            trajectory.reshape((-1, Ns, 3)), **constraints_kwargs)

        # Pad and compute norms
        gradients = np.linalg.norm(np.pad(gradients, ((0, 0), (1, 0), (0, 0))),
                                   axis=-1, ord=constraints_order)
        slewrates = np.linalg.norm(np.pad(slewrates, ((0, 0), (2, 0), (0, 0))),
                                   axis=-1, ord=constraints_order)

        # Check constraints
        gradients = trajectory.reshape((-1, 3))[np.where(gradients.flatten() > gmax)]
        slewrates = trajectory.reshape((-1, 3))[np.where(slewrates.flatten() > smax)]

        # Scatter points with vivid colors
        ax.scatter(*(gradients.T), color="r", s=POINTSIZE)
        ax.scatter(*(slewrates.T), color="b", s=POINTSIZE)
    plt.tight_layout()
    return ax
