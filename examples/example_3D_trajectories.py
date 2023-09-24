"""
===============
3D Trajectories
===============

A collection of 3D non-Cartesian trajectories with analytical definitions.

"""

# %%
# Hereafter we detail and illustrate the different arguments used in the
# parameterization of 3D non-Cartesian trajectories. Since most arguments
# are redundant across the different patterns, some of the documentation
# will refer to previous patterns for explanation.
#
# Note that most sources have not been added yet, but will be in the near
# future. Also the examples hereafter only cover natively 3D trajectories
# or famous 3D trajectories obtained from 2D. Examples on how to use
# 2D-to-3D expansion methods will be presented over another page.
#
# In this page in particular, we invite the user to manually run the script
# to be able to manipulate the plot orientations with the matplotlib interface
# to better visualize the 3D volumes.
#

# External
import matplotlib.pyplot as plt
import numpy as np

# Internal
import mrinufft as mn

from mrinufft import display_3D_trajectory


# Util function to display varying arguments
def show_argument(function, arguments, one_shot, subfigure_size):
    # Initialize trajectories with varying option
    trajectories = [function(arg) for arg in arguments]

    # Plot the trajectories side by side
    _, axes = plt.subplots(
        1,
        len(trajectories),
        figsize=(len(trajectories) * subfigure_size, subfigure_size),
        subplot_kw=dict(projection="3d"),
    )
    for axi, arg, traj in zip(axes, arguments, trajectories):
        display_3D_trajectory(
            traj,
            nb_repetitions=1,
            Nc=traj.shape[0],
            Ns=traj.shape[1],
            size=subfigure_size,
            one_shot=one_shot,
            per_plane=False,
            subfigure=axi,
        )
        axi.set_title(str(arg))
    plt.show()


# %%
# Script options
# ==============
# These options are used in the examples below as default values for all trajectories.

# Trajectory parameters
Nc = 72  # Number of shots
Ns = 512  # Number of samples per shot
in_out = True  # Choose between in-out or center-out trajectories
tilt = "uniform"  # Choose the angular distance between shots
nb_shells = 8  # Number of concentric shells for shell-type trajectories

# Display parameters
figure_size = 5  # Figure size for trajectory plots
subfigure_size = 3  # Figure size for subplots
one_shot = True  # Highlight one shot in particular


# %%
# Freeform trajectories
# =====================
#
# In this section are presented trajectories in all kinds of shapes
# and relying on different principles.
#
# 3D Cones
# ------
#
# A common pattern composed of 3D cones oriented all over within a sphere.
#
# Arguments:
#
# - ``Nc (int)``: number of individual shots
# - ``Ns (int)``: number of samples per shot
# - ``tilt (str, float)``: angle between each consecutive shot (in radians). ``(default "uniform")``
# - ``in_out (bool)``: define whether the shots should travel toward the center then outside
#   (in-out) or not (center-out). ``(default False)``
# - ``nb_zigzags (float)``: number of revolutions over a center-out shot. ``(default 5)``
# - ``width (float)``: cone width factor, normalized to cover the k-space by default.
#   ``(default 1)``
#

trajectory = mn.initialize_3D_cones(Nc, Ns, in_out=in_out)
display_3D_trajectory(
    trajectory,
    nb_repetitions=1,
    Nc=Nc,
    Ns=Ns,
    size=figure_size,
    one_shot=one_shot,
    per_plane=False,
)
plt.show()


# %%
# ``Nc (int)``
# ~~~~~~~~~~~~
#
# The number of individual shots, here 3D cones, used to cover the
# k-space. More shots means better coverage but also longer acquisitions.
#

arguments = [Nc // 4, Nc // 2, Nc, Nc * 2]
function = lambda x: mn.initialize_3D_cones(x, Ns, in_out=in_out)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# ``Ns (int)``
# ~~~~~~~~~~~~
#
# The number of samples per shot. More samples means the cones are split
# into more smaller segments, and therefore either the acquisition window
# is lengthened or the sampling rate is increased.
#

arguments = [20, 50, 80, 200]
function = lambda x: mn.initialize_3D_cones(Nc, x, in_out=in_out)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# ``tilt (str, float)``
# ~~~~~~~~~~~~~~~~~~~~~
#
# The angle between each consecutive shots, either in radians or as a
# string defining some default mods such as “uniform” for
# :math:`2 \pi / N_c`, or “golden” and “mri golden” for the different
# common definitions of golden angles. The angle is automatically adapted
# when the ``in_out`` argument is switched to keep the same behavior.
#

arguments = ["uniform", "golden", "mri golden", np.pi / 17]
function = lambda x: mn.initialize_3D_cones(Nc, Ns, tilt=x, in_out=in_out)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# ``in_out (bool)``
# ~~~~~~~~~~~~~~~~~
#
# It allows switching between different ways to define how the shot should
# travel through the k-space:
#
# - in-out: starting from the outer regions, then passing through the center
#   then going back to outer regions, often on the opposite side (radial, cones)
# - center-out or center-center: when ``in_out=False`` the trajectory will start
#   at the center, but depending on the specific trajectory formula the path might
#   end up in the outer regions (radial, spiral, cones, etc) or back to the center (rosette,
#   lissajous).
#
# Note that the behavior of both ``tilt`` and ``width`` are automatically adapted
# to the changes to avoid having to update them too when switching ``in_out``.
#

arguments = [True, False]
function = lambda x: mn.initialize_3D_cones(Nc, Ns, in_out=x)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# ``nb_zigzags (float)``
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The number of “zigzags”, or revolutions around the 3D cone on a center-out shot
# (doubled overall for in-out trajectories)
#

arguments = [0.5, 2, 5, 10]
function = lambda x: mn.initialize_3D_cones(Nc, Ns, in_out=in_out, nb_zigzags=x)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# ``width (float)``
# ~~~~~~~~~~~~~~~~~
#
# The cone width normalized such that ``width = 1`` corresponds to
# non-overlapping cones covering the whole k-space sphere, and
# therefore ``width > 1`` creates overlap between cone regions and
# ``width < 1`` tends to more radial patterns.
#

arguments = [0.2, 1, 2, 3]
function = lambda x: mn.initialize_3D_cones(Nc, Ns, in_out=in_out, width=x)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# Shell trajectories
# ==================
#
# In this section are presented trajectories that are composed of concentric
# shells, i.e. shots arranged over spherical surfaces.
#
# Helical shells
# --------------
#
# An arrangement of spirals covering sphere surfaces, often referred to as
# concentric shells. Here the name was changed to avoid confusion with
# other trajectories sharing this principle.
#
# This implementation follows the proposition from [YRB06]_ but the idea
# is much older and can be traced back at least to [PN95]_.
#
# Arguments:
#
# - ``Nc (int)``: number of individual shots. See 3D cones
# - ``Ns (int)``: number of samples per shot. See 3D cones
# - ``nb_shells (int)``: number of shells used to partition the k-space.
#   It should be lower than or equal to ``Nc``.
# - ``spiral_reduction (float)``: factor to reduce the automatic number of
#   spiral revolution per shot. ``(default 1)``
# - ``shell_tilt (str, float)``: angle between each consecutive shell (in radians).
#   ``(default "intergaps")``
# - ``shot_tilt (str, float)``: angle between each consecutive shot
#   over a sphere (in radians). ``(default "uniform")``
#

trajectory = mn.initialize_3D_helical_shells(Nc, Ns, nb_shells=nb_shells)
display_3D_trajectory(
    trajectory,
    nb_repetitions=1,
    Nc=Nc,
    Ns=Ns,
    size=figure_size,
    one_shot=one_shot,
    per_plane=False,
)
plt.show()


# %%
# ``nb_shells (int)``
# ~~~~~~~~~~~~~~~~~~~
#
# Number of shells, i.e. concentric spheres, used to partition the k-space sphere.
#

arguments = [1, 2, nb_shells // 2, nb_shells]
function = lambda x: mn.initialize_3D_helical_shells(
    Nc=x, Ns=Ns, nb_shells=x, spiral_reduction=2
)
show_argument(function, arguments, one_shot=False, subfigure_size=subfigure_size)


# %%
# ``spiral_reduction (float)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Normalized factor controlling the curvature of the spirals over the sphere surfaces.
# The curvature is determined by ``Nc`` and ``Ns`` automatically based on [YRB06]_
# in order to provide a coverage with minimal aliasing, but the curve velocities and
# accelerations might make them incompatible with gradient and slew rate constraints.
# Therefore we provided ``spiral_reduction`` to reduce (or increase) the pre-determined
# spiral curvature.
#

arguments = [0.5, 1, 2, 4]
function = lambda x: mn.initialize_3D_helical_shells(
    Nc=Nc, Ns=Ns, nb_shells=nb_shells, spiral_reduction=x
)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# ``shell_tilt (str, float)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Angle between each consecutive shells (in radians).
#

arguments = ["uniform", "intergaps", "golden", 3.1415]
function = lambda x: mn.initialize_3D_helical_shells(
    Nc=Nc, Ns=Ns, nb_shells=nb_shells, spiral_reduction=2, shell_tilt=x
)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# ``shot_tilt (str, float)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Angle between each consecutive shot over a shell/sphere (in radians).
# Note that since the number of shots per shell is determined automatically
# for each individual shell following a density provided in [YRB06]_, it
# is advised to use adaptive keywords such as "uniform" rather than hard values.
#

arguments = ["uniform", "intergaps", "golden", 0.1]
function = lambda x: mn.initialize_3D_helical_shells(
    Nc=Nc, Ns=Ns, nb_shells=nb_shells, spiral_reduction=2, shot_tilt=x
)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# Annular shells
# --------------
#
# An exclusive trajectory composed of re-arranged rings covering
# concentric shells with minimal redundancy, based on the work from [HM11]_.
# The rings are cut in halves and recombined in order to provide
# more homogeneous shot lengths as compared to a spherical stack
# of rings.
#
# Arguments:
#
# - ``Nc (int)``: number of individual shots. See 3D cones
# - ``Ns (int)``: number of samples per shot. See 3D cones
# - ``nb_shells (int)``: number of shells used to partition the k-space.
#   It should be lower than or equal to ``Nc``. See helical shells.
# - ``shell_tilt (str, float)``: angle between each consecutive shell (in radians).
#   ``(default pi)``. See helical shells.
# - ``ring_tilt (str, float)``: angle used to rotate the half-sphere of rings
#   (in radians). ``(default pi / 2)``
#

trajectory = mn.initialize_3D_annular_shells(Nc, Ns, nb_shells)
display_3D_trajectory(
    trajectory,
    nb_repetitions=1,
    Nc=Nc,
    Ns=Ns,
    size=figure_size,
    one_shot=one_shot,
    per_plane=False,
)
plt.show()


# %%
# ``ring_tilt (float)``
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Angle (in radians) defining the rotation between the two halves of
# each spheres, and therefore also the rings recombination. A zero angle,
# as seen on the first example, results in a simple stack-of-rings, while
# an angle of :math:`\pi / 2` on the third example makes the ring take
# a right angle.
#
# Note that the angle is discretized over each sphere depending on the
# number of rings, and therefore the angle might be inaccurate over smaller
# shells.
#
# An angle of :math:`\pi / 2` allows reaching the best shot length homogeneity,
# and partition the spheres into several connex curves composed of exactly
# two shots.
#

arguments = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
function = lambda x: mn.initialize_3D_annular_shells(
    Nc, Ns, nb_shells=nb_shells, ring_tilt=x
)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# References
# ==========
#
# .. [PN95] Irarrazabal, Pablo, and Dwight G. Nishimura.
#    "Fast three dimensional magnetic resonance imaging."
#    Magnetic Resonance in Medicine 33, no. 5 (1995): 656-662
# .. [YRB06] Shu, Yunhong, Stephen J. Riederer, and Matt A. Bernstein.
#    "Three‐dimensional MRI with an undersampled spherical shells trajectory."
#    Magnetic Resonance in Medicine 56, no. 3 (2006): 553-562.
# .. [HM11] Gerlach, Henryk, and Heiko von der Mosel.
#    "On sphere-filling ropes."
#    The American Mathematical Monthly 118, no. 10 (2011): 863-876