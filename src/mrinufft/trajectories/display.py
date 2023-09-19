"""Display function for trajectories."""
import matplotlib.pyplot as plt
import numpy as np

from .utils import compute_gradients, DEFAULT_GMAX, DEFAULT_SMAX
from .utils import KMAX, DEFAULT_RESOLUTION


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


def setup_2D_ticks(size, ax=None):
    """Add ticks to 2D plot."""
    if ax is None:
        plt.figure(figsize=(size, size))
        ax = plt.gca()
    ax.grid(True)
    ax.set_xticks([-KMAX, -KMAX / 2, 0, KMAX / 2, KMAX])
    ax.set_yticks([-KMAX, -KMAX / 2, 0, KMAX / 2, KMAX])
    ax.set_xlim((-KMAX, KMAX))
    ax.set_ylim((-KMAX, KMAX))
    ax.set_xlabel("kx", fontsize=FONTSIZE)
    ax.set_ylabel("ky", fontsize=FONTSIZE)
    return ax


def setup_3D_ticks(size):
    """Add ticks to 3D plot."""
    fig = plt.figure(figsize=(size, size))
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


def display_2D_trajectory(
    trajectory,
    size=5,
    one_shot=False,
    constraints=False,
    subfigure=None,
    resolution=DEFAULT_RESOLUTION,
):
    """Display of 2D trajectory.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to display.
    size : float, optional
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
    Nc, Ns = trajectory.shape[:2]
    ax = setup_2D_ticks(size, subfigure)
    trajectory = trajectory.reshape((Nc, -1, 2))
    for i in range(Nc):
        ax.plot(
            trajectory[i, :, 0],
            trajectory[i, :, 1],
            color=COLOR_CYCLE[i % NB_COLORS],
            linewidth=LINEWIDTH,
        )
    if one_shot:
        ax.plot(
            trajectory[Nc // 2, :, 0],
            trajectory[Nc // 2, :, 1],
            color="k",
            linewidth=2 * LINEWIDTH,
        )
    if constraints:
        gradients, _, slews = compute_gradients(
            trajectory.reshape((-1, Ns, 2)),
            resolution=resolution,
        )
        gradients = np.linalg.norm(np.pad(gradients, ((0, 0), (1, 0), (0, 0))), axis=-1)
        slews = np.linalg.norm(np.pad(slews, ((0, 0), (2, 0), (0, 0))), axis=-1)
        gradients = trajectory.reshape((-1, 2))[
            np.where(gradients.flatten() > DEFAULT_GMAX)
        ]
        slews = trajectory.reshape((-1, 2))[np.where(slews.flatten() > DEFAULT_SMAX)]
        ax.scatter(gradients[:, 0], gradients[:, 1], color="r", s=POINTSIZE)
        ax.scatter(slews[:, 0], slews[:, 1], color="b", s=POINTSIZE)
    plt.tight_layout()
    return ax


def display_3D_trajectory(
    trajectory,
    nb_repetitions,
    Nc,
    Ns,
    size=5,
    per_plane=True,
    one_shot=False,
    constraints=False,
    resolution=DEFAULT_RESOLUTION,
):
    """Display of 3D trajectory.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to display.
    nb_repetitions : int
        Number of repetitions.
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shots.
    size : float, optional
        Size of the figure.
    per_plane : bool, optional
        If True, display the trajectory for each plane.
    one_shot : bool, optional
        If True, highlight the trajectory of the middle repetition.
    constraints : bool, optional
        If True, display the points where the gradients and slews are above the
        default limits.
    resolution: Union[float, Tuple[float, ...]], optional
        Resolution of MR image in m.
        The default is DEFAULT_RESOLUTION.


    Returns
    -------
    ax : plt.Axes
        Axes of the figure.
    """
    ax = setup_3D_ticks(size)
    trajectory = trajectory.reshape((nb_repetitions, Nc, Ns, 3))
    for i in range(nb_repetitions):
        for j in range(Nc):
            ax.plot(
                trajectory[i, j, :, 0],
                trajectory[i, j, :, 1],
                trajectory[i, j, :, 2],
                color=COLOR_CYCLE[(i + j * (not per_plane)) % NB_COLORS],
                linewidth=LINEWIDTH,
            )
    if one_shot:
        ax.plot(
            trajectory[nb_repetitions // 2, Nc // 2, :, 0],
            trajectory[nb_repetitions // 2, Nc // 2, :, 1],
            trajectory[nb_repetitions // 2, Nc // 2, :, 2],
            color="k",
            linewidth=2 * LINEWIDTH,
        )
    if constraints:
        gradients, slews = compute_gradients(
            trajectory.reshape((-1, Ns, 3)),
            resolution=resolution,
        )
        gradients = np.linalg.norm(np.pad(gradients, ((0, 0), (1, 0), (0, 0))), axis=-1)
        slews = np.linalg.norm(np.pad(slews, ((0, 0), (2, 0), (0, 0))), axis=-1)
        gradients = trajectory.reshape((-1, 3))[
            np.where(gradients.flatten() > DEFAULT_GMAX)
        ]
        slews = trajectory.reshape((-1, 3))[np.where(slews.flatten() > DEFAULT_SMAX)]
        ax.scatter(
            gradients[:, 0], gradients[:, 1], gradients[:, 2], color="r", s=POINTSIZE
        )
        ax.scatter(slews[:, 0], slews[:, 1], slews[:, 2], color="b", s=POINTSIZE)
    plt.tight_layout()
    return ax
