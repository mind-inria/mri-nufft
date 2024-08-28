"""
This module contains visualisation functions only relevant to
the examples.
"""

# External imports
import numpy as np
import matplotlib.pyplot as plt

# Internal imports
from mrinufft import displayConfig, display_2D_trajectory, display_3D_trajectory


def show_argument(function, arguments, one_shot, subfig_size, dim="3D", axes=(0, 1)):
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
        ax.set_title(str(arg), fontsize=4 * subfig_size)
    plt.show()


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
