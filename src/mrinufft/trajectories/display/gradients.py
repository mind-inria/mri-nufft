"""Gradient display functions."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.utils import (
    Acquisition,
    compute_gradients_and_slew_rates,
    convert_trajectory_to_gradients,
)
from mrinufft.trajectories.display.config import displayConfig


def display_gradients_simply(
    trajectory: NDArray,
    shot_ids: Sequence = (0,),
    figsize: float = 5,
    fill_area: bool = True,
    uni_signal: str | None = "gray",
    uni_gradient: str | None = None,
    subfigure: plt.Figure | Sequence = None,
) -> tuple:
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
    nb_axes = trajectory.shape[-1] + 1
    if subfigure is None:
        fig = plt.figure(figsize=(figsize, figsize * nb_axes / (nb_axes - 1)))
        axes = fig.subplots(nb_axes, 1)
    elif isinstance(subfigure, Sequence) and len(subfigure) == nb_axes:
        axes = subfigure
    else:
        fig = subfigure
        axes = fig.subplots(nb_axes, 1)
    for i, ax in enumerate(axes[: nb_axes - 1]):
        ax.set_ylabel("G{}".format(["x", "y", "z"][i]), fontsize=displayConfig.fontsize)

    axes[-1].set_xlabel("Time", fontsize=displayConfig.fontsize)

    # Setup axes ticks
    for ax in axes:
        ax.grid(True)
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

    # Plot the curves for each axis
    gradients = np.diff(trajectory, axis=1)
    vmax = 1.1 * np.max(np.linalg.norm(gradients[shot_ids, ...], axis=-1, ord=1))
    for ax in axes[:-1]:
        ax.set_ylim((-vmax, vmax))
    axes[-1].set_ylim(-0.1 * vmax, vmax)

    time_axis = np.arange(gradients.shape[1])
    colors = displayConfig.get_colorlist()
    for j, s_id in enumerate(shot_ids):
        color = (
            uni_gradient
            if uni_gradient is not None
            else colors[j % displayConfig.nb_colors]
        )

        # Set each axis individually
        for i, ax in enumerate(axes[:-1]):
            ax.plot(time_axis, gradients[s_id, ..., i], color=color)
            if fill_area:
                ax.fill_between(
                    time_axis,
                    gradients[s_id, ..., i],
                    alpha=displayConfig.alpha,
                    color=color,
                )

        # Set the norm axis if requested
        gradient_norm = np.linalg.norm(gradients[s_id], axis=-1)
        axes[-1].set_ylabel("|G|", fontsize=displayConfig.fontsize)
        axes[-1].plot(gradient_norm, color=color)
        if fill_area:
            axes[-1].fill_between(
                time_axis,
                gradient_norm,
                alpha=displayConfig.alpha,
                color=color,
            )
    return axes


def display_gradients(
    trajectory: NDArray,
    shot_ids: tuple = (0,),
    figsize: float = 5,
    fill_area: bool = True,
    show_norm: bool = True,
    uni_signal: str | None = "gray",
    uni_gradient: str | None = None,
    subfigure: plt.Figure | plt.Axes | None = None,
    data_type: str = "grad",
    show_constraints: bool = False,
    acq: Acquisition | None = None,
    constraints_order: int | str | None = None,
) -> tuple:
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
    show_norm : bool, optional
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
    data_type: str, optional
        Type of data to display, either 'grad' for gradients or 'slew' for
        slew rates. This is used to determine the y-axis labels and colors
        for constraint violations. The default is 'grad'.
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
    axes : plt.Axes
        Axes of the figure.
    """
    acq = acq or Acquisition.default

    # Initialize figure with a simpler version
    axes = display_gradients_simply(
        trajectory if data_type == "grad" else np.diff(trajectory, axis=1),
        shot_ids,
        figsize,
        fill_area,
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
    if show_norm:
        axes[-1].set_ylabel("Signal (a.u.)", fontsize=displayConfig.small_fontsize)

    # Update axis ticks with rescaled values
    for i, ax in enumerate(axes):
        # Update xtick labels with time values
        if ax == axes[-1]:
            ax.xaxis.set_tick_params(labelbottom=True)
            ticks = ax.get_xticks()
            scale = acq.raster_time * 1e3
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
                trajectory[:1, :2],
                acq=acq,
            )[
                0
            ][0, 0, idx]
            / norms
        )
        scale = scale * (
            1e3 if data_type == "grad" else 1e5
        )  # Convert from T/m to mT/m
        locator = mticker.FixedLocator(ticks)
        formatter = mticker.FixedFormatter(np.around(scale * ticks, 1))
        if not show_norm or ax != axes[-1]:
            ax.yaxis.set_major_locator(locator)
            ax.yaxis.set_major_formatter(formatter)

    # Move on with constraints if requested
    if not show_constraints:
        return axes

    # Compute true gradients and slew rates
    gradients, slewrates = compute_gradients_and_slew_rates(
        trajectory[shot_ids, :], acq=acq
    )
    gradients = np.linalg.norm(gradients, axis=-1, ord=constraints_order)
    slewrates = np.linalg.norm(slewrates, axis=-1, ord=constraints_order)
    slewrates = np.pad(slewrates, ((0, 0), (0, 1)))

    # Point out hardware constraint violations
    for ax in axes:
        if data_type == "grad":
            ax.plot(
                np.ones_like(gradients.flatten()) * acq.gmax / scale * 1e3,
                "--",
                color=displayConfig.gradient_point_color,
            )
            ax.plot(
                -np.ones_like(slewrates.flatten()) * acq.smax / scale,
                "--",
                color=displayConfig.slewrate_point_color,
            )
            pts = np.where(gradients > acq.gmax)
            ax.scatter(
                pts,
                np.zeros_like(pts),
                color=displayConfig.gradient_point_color,
                s=displayConfig.pointsize,
            )
        if data_type == "slew":
            ax.plot(
                np.ones_like(slewrates.flatten()) * acq.smax / scale,
                "--",
                color=displayConfig.slewrate_point_color,
            )
            ax.plot(
                -np.ones_like(slewrates.flatten()) * acq.smax / scale,
                "--",
                color=displayConfig.slewrate_point_color,
            )
            pts = np.where(slewrates > acq.smax)
            ax.scatter(
                pts,
                np.zeros_like(pts),
                color=displayConfig.slewrate_point_color,
                s=displayConfig.pointsize,
            )
    return axes
