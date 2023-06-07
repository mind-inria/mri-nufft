"""
===============
2D Trajectories
===============

A collection of 2D non-Cartesian trajectories with analytical definitions.

"""

# External
import matplotlib.pyplot as plt
import numpy as np

# Internal
import mrinufft as nct2d

from mrinufft import display_2D_trajectory


def show_argument(function, arguments, one_shot, subfigure_size):
    # Initialize trajectories with varying option
    trajectories = [function(arg) for arg in arguments]

    # Plot the trajectories side by side
    _, axes = plt.subplots(
        1,
        len(trajectories),
        figsize=(len(trajectories) * subfigure_size, subfigure_size),
    )
    for axi, arg, traj in zip(axes, arguments, trajectories):
        display_2D_trajectory(
            traj, size=subfigure_size, one_shot=one_shot, subfigure=axi
        )
        axi.set_title(str(arg))
    plt.show()


# %%
# Options
# =======
#

# Trajectory parameters
Nc = 24  # Number of shots
Ns = 256  # Number of samples per shot
in_out = True  # Choose between in-out or center-out trajectories
tilt = "uniform"  # Choose the angular distance between shots

# Display parameters
figure_size = 5  # Figure size for trajectory plots
subfigure_size = 3  # Figure size for subplots
one_shot = True  # Highlight one shot in particular


# %%
# Documentation
# =============
#


# %%
# Hereafter we detail and illustrate the different arguments used in the
# parameterization of 2D non-Cartesian trajectories. Since most arguments
# are redundant across the different patterns, some of the documentation
# will refer to previous patterns for explanation.
#
# Note that most sources have not been added yet, but will be in the near
# future.
#


# %%
# Radial
# ------
#


# %%
# The most basic non-Cartesian trajectory composed of straight lines with
# no customization arguments besides the common ones.
#
# Arguments:
# - ``Nc (int)``: number of individual shots
# - ``Ns (int)``: number of samples per shot
# - ``tilt (str, float)``: angle between each consecutive shot (in radians) ``(default "uniform")``
# - ``in_out (bool)``: define whether the shots should travel toward the center then outside
# or not (center-out). ``(default False)``
#

trajectory = nct2d.initialize_2D_radial(Nc, Ns, tilt=tilt, in_out=in_out)
display_2D_trajectory(trajectory, size=figure_size, one_shot=one_shot)
plt.show()


# %%
# ``Nc (int)``
# ~~~~~~~~~~~~
#


# %%
# The number of individual shots, here straight lines, to cover the
# k-space. More lines means better coverage but also longer acquisitions.
#

arguments = [8, 16, 32, 64]
function = lambda x: nct2d.initialize_2D_radial(x, Ns, tilt=tilt, in_out=in_out)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# ``Ns (int)``
# ~~~~~~~~~~~~
#


# %%
# The number of samples per shot. More samples means the lines are split
# into more smaller segments and therefore either the acquisition window
# is lengthened or the sampling rate is increased.
#

arguments = [8, 16, 32, 64]
function = lambda x: nct2d.initialize_2D_radial(Nc, x, tilt=tilt, in_out=in_out)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# ``tilt (str, float)``
# ~~~~~~~~~~~~~~~~~~~~~
#


# %%
# The angle between each consecutive shots, either in radians or as a
# string defining some default mods such as “uniform” for
# :math:`2 \pi / N_c`, or “golden” or “mri golden” for the different
# common definitions of golden angles. The angle is automatically adapted
# when the ``in_out`` argument is switched to keep the same behavior.
#

arguments = ["uniform", "golden", "mri golden", np.pi / 17]
function = lambda x: nct2d.initialize_2D_radial(Nc, Ns, tilt=x, in_out=in_out)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# ``in_out (bool)``
# ~~~~~~~~~~~~~~~~~
#


# %%
# It allows to switch between different ways to define how the shot should
# travel through the k-space:
# - in-out: starting from the outer regions, then passing through the center
# then going back to outer regions, often on the opposite side (radial, cones)
# - center-out or center-center: when ``in_out=False`` the trajectory will start
# at the center, but depending on the specific trajectory formula the path might
# end up in the outer regions (radial, spiral, cones, etc) or back to the center (rosette,
# lissajous).
#

arguments = [True, False]
function = lambda x: nct2d.initialize_2D_radial(Nc, Ns, tilt=tilt, in_out=x)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# Spiral
# ------
#


# %%
# A generalized function that generates spirals defined through the
# :math:`r = a \theta^{1/n}` equality, with :math:`r` the radius and
# :math:`\theta` the polar angle. Note that the most common spirals,
# Archimedes and Fermat, are subcases of this equation.
#
# Arguments:
# - ``Nc (int)``: number of individual shots. See radial
# - ``Ns (int)``: number of samples per shot. See radial
# - ``tilt (str, float)``: angle between each consecutive shot (in radians).
# ``(default "uniform")``. See radial
# - ``in_out (bool)``: define whether the shots should travel toward the center
# then outside or not (center-out). ``(default False)``. See radial
# - ``nb_revolutions (float)``: number of revolutions performed from the
# center. ``(default 1)``
# - ``spiral (str, float)``: type of spiral defined through the above-mentionned equation.
# ``(default "archimedes")``
#

trajectory = nct2d.initialize_2D_spiral(Nc, Ns, tilt=tilt, in_out=in_out)
display_2D_trajectory(trajectory, size=figure_size, one_shot=one_shot)
plt.show()


# %%
# ``nb_revolutions (float)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#


# %%
# The number of revolutions performed from the center (i.e. performed
# twice for in-out trajectories).
#

arguments = [1 / 8, 1 / 2, 1, 3]
function = lambda x: nct2d.initialize_2D_spiral(
    Nc, Ns, tilt=tilt, nb_revolutions=x, in_out=in_out
)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# ``spiral (str, float)``
# ~~~~~~~~~~~~~~~~~~~~~~~
#


# %%
# The shape of the spiral defined through :math:`n` in the
# :math:`r = a \theta^{1/n}` equality, with :math:`r` the radius and
# :math:`\theta` the polar angle. Both ``"archimedes"`` and ``"fermat"``
# are available as string options for simplicity.
#

arguments = ["archimedes", "fermat", 0.5, 1.5]
function = lambda x: nct2d.initialize_2D_spiral(
    Nc, Ns, tilt=tilt, spiral=x, in_out=in_out
)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# Cones
# -----
#


# %%
# A radial-like trajectory zigzaging within cones over the k-space to
# offer a better coverage with more customization parameters.
#
# Arguments:
# - ``Nc (int)``: number of individual shots. See radial
# - ``Ns (int)``: number of samples per shot. See radial
# - ``tilt (str, float)``: angle between each consecutive shot (in radians).
# ``(default "uniform")``. See radial
# - ``in_out (bool)``: define whether the shots should travel toward the
# center then outside or not (center-out). ``(default False)``. See radial
# - ``nb_zigzags (float)``: number of times a cone border will be reached
# from the center. ``(default 5)``
# - ``width (float)``: cone width, normalized by the default uniform width. 
# ``(default 1)``
#

trajectory = nct2d.initialize_2D_cones(Nc, Ns, tilt=tilt, in_out=in_out)
display_2D_trajectory(trajectory, size=figure_size, one_shot=one_shot)
plt.show()


# %%
# ``nb_zigzags (float)``
# ~~~~~~~~~~~~~~~~~~~~~~
#


# %%
# The number of “zigzags”, aka the number of times the shot will touch a
# same side of the cone, from the center (i.e twice as much overall for
# in-out trajectories)
#

arguments = [0.5, 2, 5, 10]
function = lambda x: nct2d.initialize_2D_cones(
    Nc, Ns, tilt=tilt, in_out=in_out, nb_zigzags=x
)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# ``width (float)``
# ~~~~~~~~~~~~~~~~~
#


# %%
# The cone width normalized such that ``width = 1`` corresponds to
# non-overlapping cones covering uniformly the whole k-space, and
# therefore ``width > 1`` creates overlap between cone regions and
# ``width < 1`` tends to radial patterns.
#

arguments = [0.2, 1, 2, 3]
function = lambda x: nct2d.initialize_2D_cones(
    Nc, Ns, tilt=tilt, in_out=in_out, width=x
)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# Sinusoide
# ---------
#


# %%
# Another radial-like trajectory zigzaging similarly to cones, but over a
# whole band rather than cones reducing around the center.
#
# Arguments: 
# - ``Nc (int)``: number of individual shots. See radial
# - ``Ns (int)``: number of samples per shot. See radial
# - ``tilt (str, float)``: angle between each consecutive shot (in radians).
# - ``(default "uniform")``. See radial
# - ``in_out (bool)``: define whether the shots should travel toward the center
# then outside or not (center-out). ``(default False)``. See radial
# - ``nb_zigzags (float)``: number of times a cone border will be reached
# from the center. ``(default 5)``. See cones
# - ``width (float)``: band width, normalized by the default uniform width.
# ``(default 1)``. See cones
#

trajectory = nct2d.initialize_2D_sinusoide(Nc, Ns, tilt=tilt, in_out=in_out)
display_2D_trajectory(trajectory, size=figure_size, one_shot=one_shot)
plt.show()


# %%
# Rosette
# -------
#


# %%
# A repeating pattern composed of a single long curve going through the
# center multiple times and split into multiple shots.
#
# Arguments:
# - ``Nc (int)``: number of individual shots. See radial
# - ``Ns (int)``: number of samples per shot. See radial
# - ``in_out (bool)``: define whether the shots should travel toward the
# center then outside or not (center-center). ``(default False)``. See radial
# - ``coprime_index (int)``: the index of the coprime factor used
# to define the shot curvature. ``(default 0)``
#

trajectory = nct2d.initialize_2D_rosette(Nc, Ns, in_out=in_out)
display_2D_trajectory(trajectory, size=figure_size, one_shot=one_shot)
plt.show()


# %%
# ``coprime_index (int)``
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# The index used to select a compatible coprime factor, parameterized such
# that trajectories keep :math:`N_c` petals while increasing their width,
# i.e. increasing the curvature of the shots. This argument is quite
# complex with regard to the original formula in order to remain easily
# interpretable, user-friendly and optimal for MR use cases. For more
# details, please consult
# https://en.wikipedia.org/wiki/Rose\_(mathematics)#Roses_with_rational_number_values_for_k.
#

arguments = [0, 1, 5, 10]
function = lambda x: nct2d.initialize_2D_rosette(Nc, Ns, in_out=in_out, coprime_index=x)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# Polar Lissajous (WIP)
# ---------------
#


# %%
# A polar version of the Lissajous curve, repeating pattern composed of a
# single long curve going through the center multiple times and split into
# multiple shots.
#
# Arguments:
# - ``Nc (int)``: number of individual shots. See radial
# - ``Ns (int)``: number of samples per shot. See radial
# - ``in_out (bool)``: define whether the shots should travel toward the
# center then outside or not (center-center). ``(default False)``. See radial
# - ``coprime_index (int)``: the index of the coprime factor used # to define
# the shot curvature. ``(default 0)``
# - ``nb_segments (int)``: number of indepedent Lissajous curves covering
# different segments of the k-space. ``(default 1)``
#

trajectory = nct2d.initialize_2D_polar_lissajous(Nc, Ns, in_out=in_out)
display_2D_trajectory(trajectory, size=figure_size, one_shot=one_shot)
plt.show()


# %%
# ``coprime_index (int)``
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# The index used to select a compatible coprime factor, and impacting the
# shot curvature. For now, it is less trivial to select than for rosette
# but it will be updated in the future.
#

arguments = [0, 3, 12, 15]
function = lambda x: nct2d.initialize_2D_polar_lissajous(
    Nc, Ns, in_out=in_out, coprime_index=x
)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# ``nb_segments (int)``
# ~~~~~~~~~~~~~~~~~~~~~
#
# The number of Lissajous curves and segmented regions of the k-space. The
# polar Lissajous curve natively puts emphasis on the center and along the
# ``ky`` axis, but can be parameterized to rather emphasize
# ``nb_segments`` axes by reducing the coverage and duplicating a shorter
# curve.
#
# In the example below, ``nb_segments = 2`` emphasizes the diagonals as
# two Lissajous curves were created with each of them only covering two
# opposing quarters of the k-space. It implies that ``nb_segments`` should
# be a divider of ``Nc``.
#

arguments = [1, 2, 3, 4]
function = lambda x: nct2d.initialize_2D_polar_lissajous(
    Nc, Ns, in_out=in_out, nb_segments=x
)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)

#

arguments = [6, 8, 12]
function = lambda x: nct2d.initialize_2D_polar_lissajous(
    Nc, Ns, in_out=in_out, nb_segments=x
)
show_argument(function, arguments, one_shot=one_shot, subfigure_size=subfigure_size)


# %%
# Comments
# ~~~~~~~~
#


# %%
# This specific curve has never been used in MRI to the best of our
# knowledge, and was inspired by this MathCurve webpage. It is heavily
# related to the rosette trajectory but parameterized in a much more
# complex way, as shown below when varying both ``coprime_index`` and
# ``nb_segments``. It is not necessarily fit for MR applications, but was
# added out of personal interest in an effort to explore potentially
# unexploited geometries.
#

for io in [True, False]:
    for cpi in [0, 6]:
        arguments = [1, 2, 4, 12]
        function = lambda x: nct2d.initialize_2D_polar_lissajous(
            Nc, Ns, in_out=io, coprime_index=cpi, nb_segments=x
        )
        show_argument(
            function, arguments, one_shot=one_shot, subfigure_size=subfigure_size
        )
