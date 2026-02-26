"""Trajectory display functions: display_2D_trajectory, display_3D_trajectory."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.display.config import displayConfig
from mrinufft.trajectories.utils import (
    Acquisition,
    compute_gradients_and_slew_rates,
)


def _setup_2D_ticks(
    figsize: float,
    fig: plt.Figure | None = None,
    acq: Acquisition | None = None,
    scale_fov: bool = False,
) -> plt.Axes:
    """Add ticks to 2D plot."""
    acq = acq or Acquisition.default
    KMAX = acq.norm_factor * (2 * np.array(acq.res) if scale_fov else np.array([1, 1]))
    if fig is None:
        fig = plt.figure(figsize=(figsize, figsize))
    ax = fig if (isinstance(fig, plt.Axes)) else fig.subplots()
    ax.grid(True)
    ax.set_xticks([-KMAX[0], -KMAX[0] / 2, 0, KMAX[0] / 2, KMAX[0]])
    ax.set_yticks([-KMAX[1], -KMAX[1] / 2, 0, KMAX[1] / 2, KMAX[1]])
    ax.set_xlim((-KMAX[0], KMAX[0]))
    ax.set_ylim((-KMAX[1], KMAX[1]))
    ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
    ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
    return ax


def _setup_3D_ticks(
    figsize: float,
    fig: plt.FigureBase | None = None,
    acq: Acquisition = None,
    scale_fov: bool = False,
) -> plt.Axes:
    """Add ticks to 3D plot."""
    acq = acq or Acquisition.default
    KMAX = acq.norm_factor * (
        2 * np.array(acq.res) if scale_fov else np.array([1, 1, 1])
    )
    if fig is None:
        fig = plt.figure(figsize=(figsize, figsize))
    ax = fig if (isinstance(fig, plt.Axes)) else fig.add_subplot(projection="3d")
    ax.set_xticks([-KMAX[0] - KMAX[0] / 2, 0, KMAX[0] / 2, KMAX[0]])
    ax.set_yticks([-KMAX[1] - KMAX[1] / 2, 0, KMAX[1] / 2, KMAX[1]])
    ax.set_zticks([-KMAX[2] - KMAX[2] / 2, 0, KMAX[2] / 2, KMAX[2]])
    ax.axes.set_xlim3d(left=-KMAX[0], right=KMAX[0])
    ax.axes.set_ylim3d(bottom=-KMAX[1], top=KMAX[1])
    ax.axes.set_zlim3d(bottom=-KMAX[2], top=KMAX[2])
    ax.set_box_aspect((2 * KMAX[0], 2 * KMAX[1], 2 * KMAX[2]))
    ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
    ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
    ax.set_zlabel("kz", fontsize=displayConfig.fontsize)
    return ax


def display_2D_trajectory(
    trajectory: NDArray,
    figsize: float = 5,
    one_shot: bool | int = False,
    subfigure: plt.Figure | plt.Axes | None = None,
    show_constraints: bool = False,
    acq: Acquisition | None = None,
    constraints_order: float | Literal["fro"] | None = None,
    scale_fov: bool = False,
) -> plt.Axes:
    """Display 2D trajectories.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to display.
    figsize : float, optional
        Size of the figure.
    one_shot : bool or int, optional
        State if a specific shot should be highlighted in bold black.
        If `True`, highlight the middle shot.
        If `int`, highlight the shot at that index.
        The default is `False`.
    subfigure: plt.Figure, plt.SubFigure or plt.Axes, optional
        The figure where the trajectory should be displayed.
        The default is `None`.
    show_constraints : bool, optional
        Display the points where the gradients and slew rates
        are above the `gmax` and `smax` limits, respectively.
        The default is `False`.
    acq: Acquisition, optional
        Acquisition configuration to use.
        If `None`, the default acquisition is used.
    constraint_order: int, str, optional
        Norm order defining how the constraints are checked,
        typically 2 or `np.inf`, following the `numpy.linalg.norm`
        conventions on parameter `ord`.
        The default is None.
    scale_fov: bool, optional
        If True the ticks are scaled to represent the k-space in m^-1
        If False (default) the ticks are left in the normalized k-space [-0.5, 0.5]

    Returns
    -------
    ax : plt.Axes
        Axes of the figure.
    """
    acq = acq or Acquisition.default
    # Setup figure and ticks
    Nc, _ = trajectory.shape[:2]
    ax = _setup_2D_ticks(figsize, subfigure, acq, scale_fov=scale_fov)
    colors = displayConfig.get_colorlist()
    # Display every shot
    for i in range(Nc):
        ax.plot(
            trajectory[i, :, 0],
            trajectory[i, :, 1],
            color=colors[i % displayConfig.nb_colors],
            linewidth=displayConfig.linewidth,
        )

    # Display one shot in particular if requested
    if one_shot is not False:  # If True or int
        # Select shot
        shot_id = Nc // 2
        if one_shot is not True:  # If int
            shot_id = one_shot

        # Highlight the shot in black
        ax.plot(
            trajectory[shot_id, :, 0],
            trajectory[shot_id, :, 1],
            color=displayConfig.one_shot_color,
            linewidth=displayConfig.one_shot_linewidth_factor * displayConfig.linewidth,
        )

    # Point out violated constraints if requested
    if show_constraints:
        gradients, slews = compute_gradients_and_slew_rates(trajectory, acq=acq)

        # Pad and compute norms
        gradients = np.linalg.norm(
            np.pad(gradients, ((0, 0), (1, 0), (0, 0))), axis=-1, ord=constraints_order
        )
        slews = np.linalg.norm(
            np.pad(slews, ((0, 0), (2, 0), (0, 0))), axis=-1, ord=constraints_order
        )

        # Check constraints
        trajectory = trajectory.reshape((-1, 2))
        gradients = trajectory[np.where(gradients.flatten() > acq.gmax)]
        slews = trajectory[np.where(slews.flatten() > acq.smax)]

        # Scatter points with vivid colors
        ax.scatter(
            gradients[:, 0],
            gradients[:, 1],
            color=displayConfig.gradient_point_color,
            s=displayConfig.pointsize,
        )
        ax.scatter(
            slews[:, 0],
            slews[:, 1],
            color=displayConfig.slewrate_point_color,
            s=displayConfig.pointsize,
        )
    return ax


def display_3D_trajectory(
    trajectory: NDArray,
    nb_repetitions: int | None = None,
    figsize: float = 5,
    per_plane: bool = True,
    one_shot: bool | int = False,
    subfigure: plt.Figure | plt.Axes | None = None,
    show_constraints: bool = False,
    acq: Acquisition | None = None,
    constraints_order: int | str | None = None,
) -> plt.Axes:
    """Display 3D trajectories.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to display.
    nb_repetitions : int
        Number of repetitions (planes, cones, shells, etc).
        The default is `None`.
    figsize : float, optional
        Size of the figure.
    per_plane : bool, optional
        If True, display the trajectory with a different color
        for each plane.
    one_shot : bool or int, optional
        State if a specific shot should be highlighted in bold black.
        If `True`, highlight the middle shot.
        If `int`, highlight the shot at that index.
        The default is `False`.
    subfigure: plt.Figure, plt.SubFigure or plt.Axes, optional
        The figure where the trajectory should be displayed.
        The default is `None`.
    show_constraints : bool, optional
        Display the points where the gradients and slew rates
        are above the `gmax` and `smax` limits, respectively.
        The default is `False`.
    acq: Acquisition, optional
        Acquisition configuration to use.
        If `None`, the default acquisition is used.
    constraint_order: int, str, optional
        Norm order defining how the constraints are checked,
        typically 2 or `np.inf`, following the `numpy.linalg.norm`
        conventions on parameter `ord`.
        The default is None.

    Returns
    -------
    ax : plt.Axes
        Axes of the figure.
    """
    # Setup figure and ticks, and handle 2D trajectories
    acq = acq or Acquisition.default

    ax = _setup_3D_ticks(figsize, subfigure)
    if nb_repetitions is None:
        nb_repetitions = trajectory.shape[0]
    if trajectory.shape[-1] == 2:
        trajectory = np.concatenate(
            [trajectory, np.zeros((*(trajectory.shape[:2]), 1))], axis=-1
        )
    trajectory = trajectory.reshape((nb_repetitions, -1, trajectory.shape[-2], 3))
    Nc, Ns = trajectory.shape[1:3]

    colors = displayConfig.get_colorlist()
    # Display every shot
    for i in range(nb_repetitions):
        for j in range(Nc):
            ax.plot(
                trajectory[i, j, :, 0],
                trajectory[i, j, :, 1],
                trajectory[i, j, :, 2],
                color=colors[(i + j * (not per_plane)) % displayConfig.nb_colors],
                linewidth=displayConfig.linewidth,
            )

    # Display one shot in particular if requested
    if one_shot is not False:  # If True or int
        trajectory = trajectory.reshape((-1, Ns, 3))

        # Select shot
        shot_id = Nc // 2
        if one_shot is not True:  # If int
            shot_id = one_shot

        # Highlight the shot in black
        ax.plot(
            trajectory[shot_id, :, 0],
            trajectory[shot_id, :, 1],
            trajectory[shot_id, :, 2],
            color=displayConfig.one_shot_color,
            linewidth=displayConfig.one_shot_linewidth_factor * displayConfig.linewidth,
        )
        trajectory = trajectory.reshape((-1, Nc, Ns, 3))

    # Point out violated constraints if requested
    if show_constraints:
        gradients, slewrates = compute_gradients_and_slew_rates(
            trajectory.reshape((-1, Ns, 3)),
            acq=acq,
        )

        # Pad and compute norms
        gradients = np.linalg.norm(
            np.pad(gradients, ((0, 0), (1, 0), (0, 0))), axis=-1, ord=constraints_order
        )
        slewrates = np.linalg.norm(
            np.pad(slewrates, ((0, 0), (2, 0), (0, 0))), axis=-1, ord=constraints_order
        )

        # Check constraints
        gradients = trajectory.reshape((-1, 3))[
            np.where(gradients.flatten() > acq.gmax)
        ]
        slewrates = trajectory.reshape((-1, 3))[
            np.where(slewrates.flatten() > acq.smax)
        ]

        # Scatter points with vivid colors
        ax.scatter(
            *(gradients.T),
            color=displayConfig.gradient_point_color,
            s=displayConfig.pointsize,
        )
        ax.scatter(
            *(slewrates.T),
            color=displayConfig.slewrate_point_color,
            s=displayConfig.pointsize,
        )
    return ax
