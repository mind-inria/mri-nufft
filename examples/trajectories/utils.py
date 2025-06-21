"""
This module contains visualisation functions only relevant to
the examples.
"""

# External imports
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

# Internal imports
from mrinufft import display_2D_trajectory, display_3D_trajectory, displayConfig
from mrinufft.trajectories.utils import KMAX


def show_trajectory(trajectory, one_shot, figure_size):
    if trajectory.shape[-1] == 2:
        ax = display_2D_trajectory(
            trajectory, size=figure_size, one_shot=one_shot % trajectory.shape[0]
        )
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()
    else:
        ax = display_3D_trajectory(
            trajectory,
            size=figure_size,
            one_shot=one_shot % trajectory.shape[0],
            per_plane=False,
        )
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        plt.show()


def show_trajectories(
    function, arguments, one_shot, subfig_size, dim="3D", axes=(0, 1)
):
    # Initialize trajectories with varying option
    trajectories = [function(arg) for arg in arguments]

    # Plot the trajectories side by side
    fig = plt.figure(
        figsize=(len(trajectories) * subfig_size, subfig_size),
        constrained_layout=True,
    )
    subfigs = fig.subfigures(1, len(trajectories), wspace=0)
    for subfig, arg, traj in zip(subfigs, arguments, trajectories):
        if dim == "3D" and traj.shape[-1] == 3:
            ax = display_3D_trajectory(
                traj,
                size=subfig_size,
                one_shot=one_shot % traj.shape[0],
                subfigure=subfig,
                per_plane=False,
            )
        else:
            ax = display_2D_trajectory(
                traj[..., axes],
                size=subfig_size,
                one_shot=one_shot % traj.shape[0],
                subfigure=subfig,
            )
        labels = ["kx", "ky", "kz"]
        ax.set_xlabel(labels[axes[0]], fontsize=displayConfig.fontsize)
        ax.set_ylabel(labels[axes[1]], fontsize=displayConfig.fontsize)
        ax.set_aspect("equal")
        ax.set_title(str(arg), fontsize=displayConfig.fontsize)
    plt.show()


def show_density(density, figure_size, *, log_scale=False):
    density = density.T[::-1]

    plt.figure(figsize=(figure_size, figure_size))
    if log_scale:
        plt.imshow(density, cmap="jet", norm=colors.LogNorm())
    else:
        plt.imshow(density, cmap="jet")

    ax = plt.gca()
    ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
    ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
    ax.set_aspect("equal")

    plt.axis(False)
    plt.colorbar()
    plt.show()


def show_densities(function, arguments, subfig_size, *, log_scale=False):
    # Initialize k-space densities with varying option
    densities = [function(arg).T[::-1] for arg in arguments]

    # Plot the trajectories side by side
    fig, axes = plt.subplots(
        1,
        len(densities),
        figsize=(len(densities) * subfig_size, subfig_size),
        constrained_layout=True,
    )

    for ax, arg, density in zip(axes, arguments, densities):
        ax.set_title(str(arg), fontsize=displayConfig.fontsize)
        ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
        ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
        ax.set_aspect("equal")
        if log_scale:
            ax.imshow(density, cmap="jet", norm=colors.LogNorm())
        else:
            ax.imshow(density, cmap="jet")
        ax.axis(False)
    plt.show()


def show_locations(function, arguments, subfig_size, *, log_scale=False):
    # Initialize k-space locations with varying option
    locations = [function(arg) for arg in arguments]

    # Plot the trajectories side by side
    fig, axes = plt.subplots(
        1,
        len(locations),
        figsize=(len(locations) * subfig_size, subfig_size),
        constrained_layout=True,
    )

    for ax, arg, location in zip(axes, arguments, locations):
        ax.set_title(str(arg), fontsize=displayConfig.fontsize)
        ax.set_xlim(-1.05 * KMAX, 1.05 * KMAX)
        ax.set_ylim(-1.05 * KMAX, 1.05 * KMAX)
        ax.set_xlabel("kx", fontsize=displayConfig.fontsize)
        ax.set_ylabel("ky", fontsize=displayConfig.fontsize)
        ax.set_aspect("equal")
        ax.scatter(location[..., 0], location[..., 1], s=3)
    plt.show()
