"""
================
Trajectory tools
================

A collection of tools to manipulate and develop non-Cartesian trajectories.

"""

# %%
# Hereafter we detail and illustrate different generic tools that can
# be used and combined to create custom trajectories. Most of them are
# already present in the proposed trajectories from the literature.
# Since most arguments are redundant across the different patterns,
# some of the documentation will refer to previous patterns for explanation.
#
# In this page, we invite the user to manually run the script to be able
# to manipulate the plot orientations with the matplotlib interface to better
# visualize the 3D volumes.
#

# External
import matplotlib.pyplot as plt
import numpy as np

# Internal
import mrinufft as mn
import mrinufft.trajectories.tools as tools

from mrinufft import display_2D_trajectory, display_3D_trajectory


# Util function to display varying arguments
def show_argument(function, arguments, one_shot, subfig_size, dim="3D"):
    # Initialize trajectories with varying option
    trajectories = [function(arg) for arg in arguments]

    # Plot the trajectories side by side
    fig = plt.figure(
        figsize=(len(trajectories) * subfigure_size, subfigure_size),
        constrained_layout=True,
    )
    subfigs = fig.subfigures(1, len(trajectories), wspace=0)
    for subfig, arg, traj in zip(subfigs, arguments, trajectories):
        if dim == "3D":
            ax = display_3D_trajectory(
                traj,
                size=subfigure_size,
                one_shot=one_shot,
                subfigure=subfig,
                per_plane=False,
            )
        else:
            ax = display_2D_trajectory(
                traj[..., :2],
                size=subfigure_size,
                one_shot=one_shot,
                subfigure=subfig,
            )
        ax.set_aspect("equal")
        ax.set_title(str(arg), fontsize=4 * subfigure_size)
    plt.show()


def show_trajectory(trajectory, one_shot, figure_size):
    ax = display_3D_trajectory(
        trajectory, size=figure_size, one_shot=one_shot, per_plane=False
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


# %%
# Script options
# ==============
# These options are used in the examples below as default values for all trajectories.

# Trajectory parameters
Nc = 100  # Number of shots
Ns = 500  # Number of samples per shot
in_out = True  # Choose between in-out or center-out trajectories
nb_repetitions = 8  # Number of stacks, rotations, cones, shells etc.
nb_revolutions = 5  # Number of revolutions for base trajectories

# Display parameters
figure_size = 10  # Figure size for trajectory plots
subfigure_size = 6  # Figure size for subplots
one_shot = 2 * Nc // 3  # Highlight one shot in particular


# %%
# Direct tools
# ============
#
# In this section are presented the tools to apply over already
# instanciated trajectories, i.e. arrays of size :math:`(N_c, N_s, N_d)`
# with :math:`N_c` the number of shots, :math:`N_s` the number of samples
# per shot and :math:`N_d` the number of dimensions (2 or 3).

# %%
# Preparation
# -----------
# 
# Single shots
# ~~~~~~~~~~~~
#
# We can define a few simple trajectories to use in later examples:
# single shots from 2D radial, spiral, cones and 3D cones.

single_trajectories = {
    "Radial": mn.initialize_2D_radial(1, Ns, in_out=in_out),
    "Spiral": mn.initialize_2D_spiral(1, Ns, in_out=in_out),
    "2D Cones": mn.initialize_2D_cones(Nc // nb_repetitions, Ns, in_out=in_out, nb_zigzags=nb_revolutions)[:1],
    "3D Cones": mn.initialize_3D_cones(Nc, Ns, in_out=in_out, nb_zigzags=nb_revolutions)[:1],
}

# Adjust the trajectory direction
single_trajectories["3D Cones"] = np.roll(single_trajectories["3D Cones"], axis=-1, shift=1)

# %%

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: single_trajectories[x]
show_argument(function, arguments, one_shot=bool(one_shot), subfig_size=subfigure_size)


# %%
# Planes
# ~~~~~~
#
# We will also be using them as planes, or thick planes for 3D shots.

Nc_planes = Nc // nb_repetitions
z_tilt = 2 * np.pi / Nc_planes / (1 + in_out)

planar_trajectories = {
    "Radial": tools.rotate(single_trajectories["Radial"], nb_rotations=Nc_planes, z_tilt=z_tilt),
    "Spiral": tools.rotate(single_trajectories["Spiral"], nb_rotations=Nc_planes, z_tilt=z_tilt),
    "2D Cones": tools.rotate(single_trajectories["2D Cones"], nb_rotations=Nc_planes, z_tilt=z_tilt),
    "3D Cones": tools.rotate(single_trajectories["3D Cones"], nb_rotations=Nc_planes, z_tilt=z_tilt),
}

# %%

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: planar_trajectories[x]
show_argument(function, arguments, one_shot=bool(one_shot), subfig_size=subfigure_size)


# %%
# Stack
# -----
#
# Arguments:
#
# - ``trajectory (int)``: array of k-space coordinates of
#   size :math:`(N_c, N_s, N_d)`
# - ``nb_stacks (int)``: number of stacks repeating ``trajectory``
#   over the :math:`k_z`-axis.
# - ``z_tilt (float)``: angle tilt between consecutive stacks
#   over the :math:`k_z`-axis. ``(default None)``
# - ``hard_bounded (bool)``: whether the stacks should be
#   strictly bounded to k-space. ``(default True)``
#


# %%

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.stack(planar_trajectories[x], nb_stacks=nb_repetitions)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.stack(
    np.roll(
            tools.stack(single_trajectories[x], nb_stacks=Nc_planes),
            axis=-1,
            shift=1,
    ),
    nb_stacks=nb_repetitions,
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# Rotate
# ------
#
# Arguments:
#
# - ``trajectory (int)``: array of k-space coordinates of
#   size :math:`(N_c, N_s, N_d)`
# - ``nb_rotations (int)``: number of rotations repeating ``trajectory``.
# - ``x_tilt (float)``: angle tilt between consecutive stacks
#   over the :math:`k_x`-axis. ``(default None)``
# - ``y_tilt (float)``: angle tilt between consecutive stacks
#   over the :math:`k_y`-axis. ``(default None)``
# - ``z_tilt (float)``: angle tilt between consecutive stacks
#   over the :math:`k_z`-axis. ``(default None)``
#

# %%

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.rotate(planar_trajectories[x],
                                  nb_rotations=nb_repetitions,
                                  x_tilt="uniform")
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# Precess
# -------
#
# Arguments:
#
# - ``trajectory (int)``: array of k-space coordinates of
#   size :math:`(N_c, N_s, N_d)`
# - ``nb_rotations (int)``: number of rotations repeating ``trajectory``
#   over the :math:`k_z`-axis.
# - ``z_tilt (float)``: angle tilt between consecutive stacks
#   over the :math:`k_z`-axis. ``(default "golden")``
# - ``mode (str)``: whether the precession should align over an "axial"
#   or "polar" partition of the :math:`k_z`-axis. ``(default "polar")``
# - ``half_sphere (bool)``: whether the precession should be limited
#   to the upper half of the k-space sphere, typically for in-out
#   trajectories or planes. ``(default False)``
#

# %%

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.precess(planar_trajectories[x],
                                   nb_rotations=nb_repetitions,
                                   z_tilt="golden")
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.precess(single_trajectories[x],
                                   nb_rotations=Nc,
                                   z_tilt="golden")
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# Conify
# ------
#
# Arguments:
#
# - ``trajectory (int)``: array of k-space coordinates of
#   size :math:`(N_c, N_s, N_d)`
# - ``nb_cones (int)``: number of cones repeating ``trajectory``
#   with conical distortion over the :math:`k_z`-axis.
# - ``z_tilt (float)``: angle tilt between consecutive stacks
#   over the :math:`k_z`-axis. ``(default "golden")``
# - ``in_out (bool)``: whether to account for the in-out
#   nature of some trajectories to avoid hard angles
#   around the center, ``(default False)``
# - ``max_angle (float)``: maximum angle of the cones. ``(default pi / 2)``
#


arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.conify(planar_trajectories[x],
                                  nb_cones=nb_repetitions,
                                  in_out=in_out)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%


arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.conify(single_trajectories[x],
                                  nb_cones=Nc,
                                  z_tilt="golden",
                                  in_out=in_out)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# Functional tools
# ================

# %%
# Preparation
# -----------
# 
# Functions
# ~~~~~~~~~

init_trajectories = {
    "Radial": lambda Nc: mn.initialize_2D_radial(Nc, Ns, in_out=in_out),
    "Spiral": lambda Nc: mn.initialize_2D_spiral(Nc, Ns, in_out=in_out),
    "2D Cones": lambda Nc: mn.initialize_2D_cones(Nc, Ns, in_out=in_out),
    "3D Cones": lambda Nc: tools.rotate(single_trajectories["3D Cones"],
        nb_rotations=Nc, z_tilt=2 * np.pi / Nc / (1 + in_out)),
}


# %%
# Stack spherically
# -----------------
#
# Arguments:
#
# - ``trajectory_func (function)``: trajectory function that
#   should return an array-like with the usual :math:`(N_c, N_s, N_d)` size.
# - ``Nc (int)``: number of shots to use for the whole spherically
#   stacked trajectory.
# - ``nb_stacks (int)``: number of stacks repeating ``trajectory``
#   over the :math:`k_z`-axis.
# - ``z_tilt (float)``: angle tilt between consecutive stacks
#   around the :math:`k_z`-axis. ``(default None)``
# - ``hard_bounded (bool)``: whether the stacks should be
#   strictly bounded to k-space. ``(default True)``
# - ``**kwargs``: trajectory initialization parameters for the
#   function provided with ``trajectory_func``.
#


arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.stack_spherically(init_trajectories[x],
                                             Nc=Nc, nb_stacks=nb_repetitions)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)



# %%
# Stack spherically
# -----------------
#
# Arguments:
#
# - ``trajectory_func (function)``: trajectory function that
#   should return an array-like with the usual :math:`(N_c, N_s, N_d)` size.
# - ``Nc (int)``: number of shots to use for the whole spherically
#   stacked trajectory.
# - ``nb_shells (int)``: number of shells repeating ``trajectory``
#   with spherical distortion over the :math:`k_z`-axis.
# - ``z_tilt (float)``: angle tilt between concentric shells
#   around the :math:`k_z`-axis. ``(default None)``
# - ``hemisphere_mode``: define how the lower hemisphere should
#   be oriented relatively to the upper one, with "symmetric" providing
#   a kx-ky planar symmetry by changing the polar angle, and with
#   "reversed" promoting continuity (for example in spirals) by
#   reversing the azimuth angle. ``(default "symmetric")``.
# - ``**kwargs``: trajectory initialization parameters for the
#   function provided with ``trajectory_func``.
#


arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.shellify(init_trajectories[x],
                                    Nc=Nc, nb_shells=nb_repetitions)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%
