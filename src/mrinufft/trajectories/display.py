"""Display functions for trajectories."""

from __future__ import annotations

import itertools
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from numpy.typing import NDArray

from .utils import (
    DEFAULT_GMAX,
    DEFAULT_RASTER_TIME,
    DEFAULT_SMAX,
    KMAX,
    compute_gradients_and_slew_rates,
    convert_trajectory_to_gradients,
)


class displayConfig:
    """
    A container class used to share arguments related to display.

    The values can be updated either directy (and permanently) or temporarily by using
    a context manager.

    Examples
    --------
    >>> from mrinufft.trajectories.display import displayConfig
    >>> displayConfig.alpha
    0.2
    >>> with displayConfig(alpha=0.5):
            print(displayConfig.alpha)
    0.5
    >>> displayConfig.alpha
    0.2
    """

    alpha: float = 0.2
    """Transparency used for area plots, by default ``0.2``."""
    linewidth: float = 2
    """Width for lines or curves, by default ``2``."""
    pointsize: int = 10
    """Size for points used to show constraints, by default ``10``."""
    fontsize: int = 18
    """Font size for most labels and texts, by default ``18``."""
    small_fontsize: int = 14
    """Font size for smaller texts, by default ``14``."""
    nb_colors: int = 10
    """Number of colors to use in the color cycle, by default ``10``."""
    palette: str = "tab10"
    """Name of the color palette to use, by default ``"tab10"``.
    This can be any of the matplotlib colormaps, or a list of colors."""
    one_shot_color: str = "k"
    """Matplotlib color for the highlighted shot, by default ``"k"`` (black)."""
    one_shot_linewidth_factor: float = 2
    """Factor to multiply the linewidth of the highlighted shot, by default ``2``."""
    gradient_point_color: str = "r"
    """Matplotlib color for gradient constraint points, by default ``"r"`` (red)."""
    slewrate_point_color: str = "b"
    """Matplotlib color for slew rate constraint points, by default ``"b"`` (blue)."""

    def __init__(self, **kwargs: Any) -> None:  # noqa ANN401
        """Update the display configuration."""
        self.update(**kwargs)

    def update(self, **kwargs: Any) -> None:  # noqa ANN401
        """Update the display configuration."""
        self._old_values = {}
        for key, value in kwargs.items():
            self._old_values[key] = getattr(displayConfig, key)
            setattr(displayConfig, key, value)

    def reset(self) -> None:
        """Restore the display configuration."""
        for key, value in self._old_values.items():
            setattr(displayConfig, key, value)
        delattr(self, "_old_values")

    def __enter__(self) -> displayConfig:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: Any) -> None:  # noqa ANN401
        """Exit the context manager."""
        self.reset()

    @classmethod
    def get_colorlist(cls) -> list[str | NDArray]:
        """Extract a list of colors from a matplotlib palette.

        If the palette is continuous, the colors will be sampled from it.
        If its a categorical palette, the colors will be used in cycle.

        Parameters
        ----------
        palette : str, or list of colors, or matplotlib colormap
            Name of the palette to use, or list of colors, or matplotlib colormap.
        nb_colors : int, optional
            Number of colors to extract from the palette.
            The default is -1, and the value will be read from displayConfig.nb_colors.

        Returns
        -------
        colorlist : list of matplotlib colors.
        """
        if isinstance(cls.palette, str):
            cm = mpl.colormaps[cls.palette]
        elif isinstance(cls.palette, mpl.colors.Colormap):
            cm = cls.palette
        elif isinstance(cls.palette, list):
            cm = mpl.cm.ListedColormap(cls.palette)
        colorlist = []
        colors = getattr(cm, "colors", [])
        if 0 < len(colors) < cls.nb_colors:
            colorlist = [
                c for _, c in zip(range(cls.nb_colors), itertools.cycle(cm.colors))
            ]
        else:
            colorlist = cm(np.linspace(0, 1, cls.nb_colors))
        return colorlist


##############
# TICK UTILS #
##############


def _setup_2D_ticks(figsize: float, fig: plt.Figure | None = None) -> plt.Axes:
    """Add ticks to 2D plot."""
    if fig is None:
        fig = plt.figure(figsize=(figsize, figsize))
    ax = fig if (isinstance(fig, plt.Axes)) else fig.subplots()
    ax.grid(True)
    ax.set_xticks([-KMAX, -KMAX / 2, 0, KMAX / 2, KMAX])
    ax.set_yticks([-KMAX, -KMAX / 2, 0, KMAX / 2, KMAX])
    ax.set_xlim((-KMAX, KMAX))
    ax.set_ylim((-KMAX, KMAX))
    ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
    ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
    return ax


def _setup_3D_ticks(figsize: float, fig: plt.Figure | None = None) -> plt.Axes:
    """Add ticks to 3D plot."""
    if fig is None:
        fig = plt.figure(figsize=(figsize, figsize))
    ax = fig if (isinstance(fig, plt.Axes)) else fig.add_subplot(projection="3d")
    ax.set_xticks([-KMAX, -KMAX / 2, 0, KMAX / 2, KMAX])
    ax.set_yticks([-KMAX, -KMAX / 2, 0, KMAX / 2, KMAX])
    ax.set_zticks([-KMAX, -KMAX / 2, 0, KMAX / 2, KMAX])
    ax.axes.set_xlim3d(left=-KMAX, right=KMAX)
    ax.axes.set_ylim3d(bottom=-KMAX, top=KMAX)
    ax.axes.set_zlim3d(bottom=-KMAX, top=KMAX)
    ax.set_box_aspect((2 * KMAX, 2 * KMAX, 2 * KMAX))
    ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
    ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
    ax.set_zlabel("kz", fontsize=displayConfig.fontsize)
    return ax


######################
# TRAJECTORY DISPLAY #
######################


def display_2D_trajectory(
    trajectory: NDArray,
    figsize: float = 5,
    one_shot: bool | int = False,
    subfigure: plt.Figure | plt.Axes | None = None,
    show_constraints: bool = False,
    gmax: float = DEFAULT_GMAX,
    smax: float = DEFAULT_SMAX,
    constraints_order: int | str | None = None,
    **constraints_kwargs: Any,  # noqa ANN401
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
    gmax: float, optional
        Maximum constraint on the gradients in T/m.
        The default is `DEFAULT_GMAX`.
    smax: float, optional
        Maximum constraint on the slew rates in T/m/ms.
        The default is `DEFAULT_SMAX`.
    constraint_order: int, str, optional
        Norm order defining how the constraints are checked,
        typically 2 or `np.inf`, following the `numpy.linalg.norm`
        conventions on parameter `ord`.
        The default is None.
    **constraints_kwargs
        Acquisition parameters used to check on hardware constraints,
        following the parameter convention from
        `mrinufft.trajectories.utils.compute_gradients_and_slew_rates`.

    Returns
    -------
    ax : plt.Axes
        Axes of the figure.
    """
    # Setup figure and ticks
    Nc, Ns = trajectory.shape[:2]
    ax = _setup_2D_ticks(figsize, subfigure)
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
        gradients, slews = compute_gradients_and_slew_rates(
            trajectory, **constraints_kwargs
        )

        # Pad and compute norms
        gradients = np.linalg.norm(
            np.pad(gradients, ((0, 0), (1, 0), (0, 0))), axis=-1, ord=constraints_order
        )
        slews = np.linalg.norm(
            np.pad(slews, ((0, 0), (2, 0), (0, 0))), axis=-1, ord=constraints_order
        )

        # Check constraints
        trajectory = trajectory.reshape((-1, 2))
        gradients = trajectory[np.where(gradients.flatten() > gmax)]
        slews = trajectory[np.where(slews.flatten() > smax)]

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
    gmax: float = DEFAULT_GMAX,
    smax: float = DEFAULT_SMAX,
    constraints_order: int | str | None = None,
    **constraints_kwargs: Any,  # noqa ANN401
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
    gmax: float, optional
        Maximum constraint on the gradients in T/m.
        The default is `DEFAULT_GMAX`.
    smax: float, optional
        Maximum constraint on the slew rates in T/m/ms.
        The default is `DEFAULT_SMAX`.
    constraint_order: int, str, optional
        Norm order defining how the constraints are checked,
        typically 2 or `np.inf`, following the `numpy.linalg.norm`
        conventions on parameter `ord`.
        The default is None.
    **kwargs
        Acquisition parameters used to check on hardware constraints,
        following the parameter convention from
        `mrinufft.trajectories.utils.compute_gradients_and_slew_rates`.

    Returns
    -------
    ax : plt.Axes
        Axes of the figure.
    """
    # Setup figure and ticks, and handle 2D trajectories
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
            trajectory.reshape((-1, Ns, 3)), **constraints_kwargs
        )

        # Pad and compute norms
        gradients = np.linalg.norm(
            np.pad(gradients, ((0, 0), (1, 0), (0, 0))), axis=-1, ord=constraints_order
        )
        slewrates = np.linalg.norm(
            np.pad(slewrates, ((0, 0), (2, 0), (0, 0))), axis=-1, ord=constraints_order
        )

        # Check constraints
        gradients = trajectory.reshape((-1, 3))[np.where(gradients.flatten() > gmax)]
        slewrates = trajectory.reshape((-1, 3))[np.where(slewrates.flatten() > smax)]

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


####################
# GRADIENT DISPLAY #
####################


def display_gradients_simply(
    trajectory: NDArray,
    shot_ids: tuple[int, ...] = (0,),
    figsize: float = 5,
    fill_area: bool = True,
    show_signal: bool = True,
    uni_signal: str | None = "gray",
    uni_gradient: str | None = None,
    subfigure: plt.Figure | None = None,
) -> tuple[plt.Axes]:
    """Display gradients based on trajectory of any dimension.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to display.
    shot_ids : tuple[int, ...], optional
        Indices of the shots to display.
        The default is `[0]`.
    figsize : float, optional
        Size of the figure.
    fill_area : bool, optional
        Fills the area under the curve for improved visibility and
        representation of the integral, aka trajectory.
        The default is `True`.
    show_signal : bool, optional
        Show an additional illustration of the signal as
        the modulated distance to the center.
        The default is `True`.
    uni_signal : str or None, optional
        Define whether the signal should be represented by a
        unique color given as argument or just by the default
        color cycle when `None`.
        The default is `"gray"`.
    uni_signal : str or None, optional
        Define whether the gradients should be represented by a
        unique color given as argument or just by the default
        color cycle when `None`.
        The default is `None`.
    subfigure: plt.Figure, optional
        The figure where the trajectory should be displayed.
        The default is `None`.

    Returns
    -------
    axes : plt.Axes
        Axes of the figure.
    """
    # Setup figure and labels
    Nd = trajectory.shape[-1]
    if subfigure is None:
        fig = plt.figure(figsize=(figsize, figsize * (Nd + show_signal) / Nd))
    else:
        fig = subfigure
    axes = fig.subplots(Nd + show_signal, 1)
    for i, ax in enumerate(axes[:Nd]):
        ax.set_ylabel("G{}".format(["x", "y", "z"][i]), fontsize=displayConfig.fontsize)
    axes[-1].set_xlabel("Time", fontsize=displayConfig.fontsize)

    # Setup axes ticks
    for ax in axes:
        ax.grid(True)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

    # Plot the curves for each axis
    gradients = np.diff(trajectory, axis=1)
    vmax = 1.1 * np.max(np.abs(gradients[shot_ids, ...]))
    x_axis = np.arange(gradients.shape[1])
    colors = displayConfig.get_colorlist()
    for j, s_id in enumerate(shot_ids):
        for i, ax in enumerate(axes[:Nd]):
            ax.set_ylim((-vmax, vmax))
            color = (
                uni_gradient
                if uni_gradient is not None
                else colors[j % displayConfig.nb_colors]
            )
            ax.plot(x_axis, gradients[s_id, ..., i], color=color)
            if fill_area:
                ax.fill_between(
                    x_axis,
                    gradients[s_id, ..., i],
                    alpha=displayConfig.alpha,
                    color=color,
                )

    # Return axes alone
    if not show_signal:
        return axes

    # Show signal as modulated distance to center
    distances = np.linalg.norm(trajectory[shot_ids, 1:-1], axis=-1)
    distances = np.tile(distances.reshape((len(shot_ids), -1, 1)), (1, 1, 10))
    signal = 1 - distances.reshape((len(shot_ids), -1)) / np.max(distances)
    signal = (
        signal * np.exp(2j * np.pi * figsize / 100 * np.arange(signal.shape[1]))
    ).real
    signal = signal * np.abs(signal) ** 3

    colors = displayConfig.get_colorlist()
    # Show signal for each requested shot
    axes[-1].set_ylim((-1, 1))
    axes[-1].set_ylabel("Signal", fontsize=displayConfig.fontsize)
    for j in range(len(shot_ids)):
        color = (
            uni_signal
            if (uni_signal is not None)
            else colors[j % displayConfig.nb_colors]
        )
        axes[-1].plot(np.arange(signal.shape[1]), signal[j], color=color)
    return axes


def display_gradients(
    trajectory: NDArray,
    shot_ids: tuple[int, ...] = (0,),
    figsize: float = 5,
    fill_area: bool = True,
    show_signal: bool = True,
    uni_signal: str | None = "gray",
    uni_gradient: str | None = None,
    subfigure: plt.Figure | plt.Axes | None = None,
    show_constraints: bool = False,
    gmax: float = DEFAULT_GMAX,
    smax: float = DEFAULT_SMAX,
    constraints_order: int | str | None = None,
    raster_time: float = DEFAULT_RASTER_TIME,
    **constraints_kwargs: Any,  # noqa ANN401
) -> tuple[plt.Axes]:
    """Display gradients based on trajectory of any dimension.

    Parameters
    ----------
    trajectory : NDArray
        Trajectory to display.
    shot_ids : list of int
        Indices of the shots to display.
        The default is `(0,)`.
    figsize : float, optional
        Size of the figure.
    fill_area : bool, optional
        Fills the area under the curve for improved visibility and
        representation of the integral, aka trajectory.
        The default is `True`.
    show_signal : bool, optional
        Show an additional illustration of the signal as
        the modulated distance to the center.
        The default is `True`.
    uni_signal : str or None, optional
        Define whether the signal should be represented by a
        unique color given as argument or just by the default
        color cycle when `None`.
        The default is `"gray"`.
    uni_signal : str or None, optional
        Define whether the gradients should be represented by a
        unique color given as argument or just by the default
        color cycle when `None`.
        The default is `None`.
    subfigure: plt.Figure or plt.SubFigure, optional
        The figure where the trajectory should be displayed.
        The default is `None`.
    show_constraints : bool, optional
        Display the points where the gradients and slew rates
        are above the `gmax` and `smax` limits, respectively.
        The default is `False`.
    gmax: float, optional
        Maximum constraint on the gradients in T/m.
        The default is `DEFAULT_GMAX`.
    smax: float, optional
        Maximum constraint on the slew rates in T/m/ms.
        The default is `DEFAULT_SMAX`.
    constraint_order: int, str, optional
        Norm order defining how the constraints are checked,
        typically 2 or `np.inf`, following the `numpy.linalg.norm`
        conventions on parameter `ord`.
        The default is None.
    raster_time: float, optional
        Amount of time between the acquisition of two
        consecutive samples in ms.
        The default is `DEFAULT_RASTER_TIME`.
    **constraints_kwargs
        Acquisition parameters used to check on hardware constraints,
        following the parameter convention from
        `mrinufft.trajectories.utils.compute_gradients_and_slew_rates`.

    Returns
    -------
    axes : plt.Axes
        Axes of the figure.
    """
    # Initialize figure with a simpler version
    axes = display_gradients_simply(
        trajectory,
        shot_ids,
        figsize,
        fill_area,
        show_signal,
        uni_signal,
        uni_gradient,
        subfigure,
    )

    # Setup figure and labels
    Nd = trajectory.shape[-1]
    for i, ax in enumerate(axes[:Nd]):
        ax.set_ylabel(
            "G{} (mT/m)".format(["x", "y", "z"][i]),
            fontsize=displayConfig.small_fontsize,
        )
    axes[-1].set_xlabel("Time (ms)", fontsize=displayConfig.small_fontsize)
    if show_signal:
        axes[-1].set_ylabel("Signal (a.u.)", fontsize=displayConfig.small_fontsize)

    # Update axis ticks with rescaled values
    for i, ax in enumerate(axes):
        # Update xtick labels with time values
        if ax == axes[-1]:
            ax.xaxis.set_tick_params(labelbottom=True)
            ticks = ax.get_xticks()
            scale = (0.1 if (show_signal and ax == axes[-1]) else 1) * raster_time
            locator = mticker.FixedLocator(ticks)
            formatter = mticker.FixedFormatter(np.around(scale * ticks, 2))
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        # Update ytick labels with gradient values
        ax.yaxis.set_tick_params(labelleft=True)
        ticks = ax.get_yticks()
        idx = min(i, Nd - 1)
        norms = np.diff(trajectory[:1, :2, idx]).squeeze()
        norms = np.where(norms != 0, norms, 1)
        scale = (
            convert_trajectory_to_gradients(
                trajectory[:1, :2], raster_time=raster_time, **constraints_kwargs
            )[0][0, 0, idx]
            / norms
        )
        scale = 1e3 * scale  # Convert from T/m to mT/m
        locator = mticker.FixedLocator(ticks)
        formatter = mticker.FixedFormatter(np.around(scale * ticks, 1))
        if not show_signal or ax != axes[-1]:
            ax.yaxis.set_major_locator(locator)
            ax.yaxis.set_major_formatter(formatter)

    # Move on with constraints if requested
    if not show_constraints:
        return axes

    # Compute true gradients and slew rates
    gradients, slewrates = compute_gradients_and_slew_rates(
        trajectory[shot_ids, :], **constraints_kwargs
    )
    gradients = np.linalg.norm(gradients, axis=-1, ord=constraints_order)
    slewrates = np.linalg.norm(slewrates, axis=-1, ord=constraints_order)
    slewrates = np.pad(slewrates, ((0, 0), (0, 1)))

    # Point out hardware constraint violations
    for ax in axes[:Nd]:
        pts = np.where(gradients > gmax)
        ax.scatter(
            pts,
            np.zeros_like(pts),
            color=displayConfig.gradient_point_color,
            s=displayConfig.pointsize,
        )
        pts = np.where(slewrates > smax)
        ax.scatter(
            pts,
            np.zeros_like(pts),
            color=displayConfig.slewrate_point_color,
            s=displayConfig.pointsize,
        )
    return axes
