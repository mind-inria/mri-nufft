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
Nc = 120  # Number of shots
Ns = 500  # Number of samples per shot
in_out = False  # Choose between in-out or center-out trajectories
tilt = "uniform"  # Angular distance between shots
nb_repetitions = 6  # Number of stacks, rotations, cones, shells etc.
nb_revolutions = 1  # Number of revolutions for base trajectories

# Display parameters
figure_size = 10  # Figure size for trajectory plots
subfigure_size = 6  # Figure size for subplots
one_shot = -5  # Highlight one shot in particular


# %%
# Freeform trajectories
# =====================
#
# In this section are presented trajectories in all kinds of shapes
# and relying on different principles.
#
# 3D Cones
# --------
#
# A common pattern composed of 3D cones oriented all over within a sphere.
#
# Arguments:
#
# - ``Nc (int)``: number of individual shots
# - ``Ns (int)``: number of samples per shot
# - ``tilt (str, float)``: angle between each consecutive shot (in radians).
#   ``(default "uniform")``
# - ``in_out (bool)``: define whether the shots should travel toward
#   the center then outside (in-out) or not (center-out). ``(default False)``
# - ``nb_zigzags (float)``: number of revolutions over a center-out shot.
#   ``(default 5)``
# - ``width (float)``: cone width factor, normalized to densely cover the k-space
#   by default. ``(default 1)``
#

trajectory = mn.initialize_3D_cones(Nc, Ns, in_out=in_out)
show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)


# %%
# ``Nc (int)``
# ~~~~~~~~~~~~
#
# The number of individual shots, here 3D cones, used to cover the
# k-space. More shots means better coverage but also longer acquisitions.
#

arguments = [Nc // 4, Nc // 2, Nc, Nc * 2]
function = lambda x: mn.initialize_3D_cones(x, Ns, in_out=in_out)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# ``Ns (int)``
# ~~~~~~~~~~~~
#
# The number of samples per shot. More samples means the cones are split
# into more smaller segments, and therefore either the acquisition window
# is lengthened or the sampling rate is increased.
#

arguments = [10, 25, 40, 100]
function = lambda x: mn.initialize_3D_cones(Nc, x, in_out=in_out)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


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

arguments = ["uniform", "golden", "mri-golden", np.pi / 17]
function = lambda x: mn.initialize_3D_cones(Nc, Ns, tilt=x, in_out=in_out)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


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
#   end up in the outer regions (radial, spiral, cones, etc)
#   or back to the center (rosette, lissajous).
#
# Note that the behavior of both ``tilt`` and ``width`` are automatically adapted
# to the changes to avoid having to update them too when switching ``in_out``.
#

arguments = [True, False]
function = lambda x: mn.initialize_3D_cones(Nc, Ns, in_out=x)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# ``nb_zigzags (float)``
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The number of “zigzags”, or revolutions around the 3D cone on a center-out shot
# (doubled overall for in-out trajectories)
#

arguments = [0.5, 2, 5, 10]
function = lambda x: mn.initialize_3D_cones(Nc, Ns, in_out=in_out, nb_zigzags=x)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


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
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# FLORET
# ------
#
# A pattern introduced in [Pip+11]_ composed of Fermat spirals
# folded into cones. The acronym stands for Fermat Looped, Orthogonally
# Encoded Trajectories. Most arguments are related either to
# ``initialize_2D_spiral`` or to ``tools.conify``.
#
# Arguments:
#
# - ``Nc (int)``: number of individual shots. See 3D cones
# - ``Ns (int)``: number of samples per shot. See 3D cones
# - ``in_out (bool)``: define whether the shots should travel toward
#   the center then outside (in-out) or not (center-out).
#   ``(default False)``. See 3D cones or 2D spiral
# - ``nb_revolutions (float)``: number of revolutions performed from the
#   center. ``(default 1)``. See 2D spiral
# - ``spiral_tilt (str, float)``: angle between each spiral within a plane
#   (in radians). ``(default "uniform")``. See 2D spiral
# - ``spiral (str, float)``: type of spiral defined through the general
#   archimedean equation. ``(default "fermat")``. See 2D spiral
# - ``nb_cones (int)``: number of cones around the :math:`k_z`-axis.
#   See ``tools.conify``
# - ``cone_tilt (float)``: angle tilt between consecutive cones
#   around the :math:`k_z`-axis. ``(default "golden")``. See ``tools.conify``
# - ``max_angle (float)``: maximum angle of the cones. ``(default pi / 2)``.
#   See ``tools.conify``
# - ``axes (tuple)``: axes over which cones are created, by default (2,)
#

trajectory = mn.initialize_3D_floret(
    Nc,
    Ns,
    in_out=in_out,
    nb_revolutions=nb_revolutions,
    nb_cones=nb_repetitions,
    max_angle=np.pi / 2,
)[::-1]
show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)

# %%
#
# ``axes (tuple)``
# ~~~~~~~~~~~~~~~~
#
# Indices of the different axes over which cones are created,
# with 0, 1, 2 corresponding to :math:`k_x, k_y, k_z` respectively.
# The ``Nc`` shots and ``nb_cones`` are distributed
# over all axes, and therefore should be divisible by ``len(axes)``.
#
# The point is to provide an efficient coverage by reducing ``max_angle``
# to avoid redundancy around one axis, but still cover the whole
# k-space sphere by duplicating cones along several axes, as initially
# proposed by [Pip+11]_.
#

arguments = [(2,), (0,), (0, 2), (0, 1, 2)]
function = lambda x: mn.initialize_3D_floret(
    Nc,
    Ns,
    in_out=in_out,
    nb_revolutions=nb_revolutions,
    nb_cones=nb_repetitions,
    max_angle=np.pi / 4,
    axes=x,
)[::-1]
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# Wave-CAIPI
# ----------
#
# A pattern introduced in [Bil+15]_ composed of helices evolving
# in the same direction and packed together,
# inherited from trajectories such as CAIPIRINHA and
# Bunched Phase-Encoding (BPE) designed to better spread aliasing
# and facilitate reconstruction.
#
# Arguments:
#
# - ``Nc (int)``: number of individual shots. See 3D cones
# - ``Ns (int)``: number of samples per shot. See 3D cones
# - ``nb_revolutions (str, float)``: number of revolution of the helices.
#   ``(default 5)``
# - ``width (float)``: helix width factor, normalized to densely
#   cover the k-space by default. ``(default 1)``.
# - ``packing (str)``: packing method used to position the helices.
#   ``(default "triangular")``
# - ``shape (str, float)``: shape over the 2D kx-ky plane to pack with shots.
#   ``(default "circle")``
# - ``spacing (tuple(int, int))``: Spacing between helices over the
#   2D :math:`k_x`-:math:`k_y` plane normalized similarly to `width`. ``(default (1, 1))``

trajectory = mn.initialize_3D_wave_caipi(Nc, Ns)
show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)

# %%
# ``nb_revolutions (float)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The number of revolutions of the helices from bottom to top.
#

arguments = [0.5, 2.5, 5, 10]
function = lambda x: mn.initialize_3D_wave_caipi(Nc, Ns, nb_revolutions=x)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%
# ``width (float)``
# ~~~~~~~~~~~~~~~~~
#
# The helix diameter normalized such that ``width = 1`` corresponds to
# non-overlapping shots densely covering the k-space shape (for square packing),
# and therefore ``width > 1`` creates overlap between cone regions and
# ``width < 1`` tends to more radial patterns.
#
# See ``packing`` for more details about coverage.
#

arguments = [0.2, 1, 2, 3]
function = lambda x: mn.initialize_3D_wave_caipi(Nc, Ns, width=x)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%
# ``packing (str)``
# ~~~~~~~~~~~~~~~~~
#
# The method used to pack circles of same size within an arbitrary ``shape``.
# The available methods are ``"triangular"`` and ``"square"`` for regular tiling
# over dense grids, and ``"circular"`` and ``"random"`` for irregular packing.
# Different aliases are available, such as ``"triangle"``, ``"hexagon"`` instead
# of ``"triangular"``.
#
# Note that ``"triangular"`` packing has slightly overlapping helices,
# as it corresponds to a triangular/hexagonal grid.
# The ``"random"`` packing also naturally overlaps as the positions are determined
# following a uniform distribution over :math:`k_x` and :math:`k_y` dimensions.
#

arguments = ["triangular", "square", "circular", "random"]
function = lambda x: mn.initialize_3D_wave_caipi(Nc, Ns, packing=x)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%

show_argument(
    function, arguments, one_shot=one_shot, subfig_size=subfigure_size, dim="2D"
)

# %%
# ``shape (str, float)``
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The 2D shape defined over the :math:`k_x`-:math:`k_y` plane
# and where the helices should be packed. Aliases are available for convenience,
# namely ``"circle"``, ``"square"``, ``"diamond"``, but shapes are primarily
# defined through the p-norm of the 2D coordinates following the convention
# of the ``ord`` parameter from ``numpy.linalg.norm``.
#
# The shapes are approximately respected depending on the available ``Nc``
# parameter, and extra shots on the edges will be placed in priority to have
# a minimal 2-norm (eliminating the diagonals) except for circles with infinity-norm
# (accumulating over the diagonals).
#

arguments = ["circle", "square", "diamond", 0.5]
function = lambda x: mn.initialize_3D_wave_caipi(Nc, Ns, shape=x)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%

show_argument(
    function, arguments, one_shot=one_shot, subfig_size=subfigure_size, dim="2D"
)

# %%
# ``spacing (tuple(int, int))``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The spacing between helices over the :math:`k_x`-:math:`k_y` plane, mostly
# defined for ``"square"`` packing. It is defined to correspond to the ``width``
# unit, itself automatically matching the helix diameters, which can cause more
# complex behaviors for other packing methods as the diameters are normalized to
# fit within the cubic k-space.
#

arguments = [(1, 1), (2, 1), (1, 2), (2.3, 1.8)]
function = lambda x: mn.initialize_3D_wave_caipi(Nc, Ns, packing="square", spacing=x)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%

show_argument(
    function, arguments, one_shot=one_shot, subfig_size=subfigure_size, dim="2D"
)


# %%
# Seiffert spirals / Yarnball
# ---------------------------
#
# A recent pattern with tightly controlled gradient norms using radially
# modulated Seiffert spirals, based on Jacobi elliptic functions.
# Note that Seiffert spirals more commonly refer to a curve evolving
# over a sphere surface rather than a volume, with the advantage of
# having a constant speed and angular velocity. The MR trajectory
# is obtained by increasing progressively the radius of the sphere.
#
# This implementation follows the proposition from [SMR18]_ based on
# works from [Er00]_ and [Br09]_. The pattern is also referred to as
# Yarnball by a different team [SB21]_, as a nod to the Yarn trajectory
# pictured in [IN95]_, even though both admittedly share little in common.
#
# Arguments:
#
# - ``Nc (int)``: number of individual shots. See 3D cones
# - ``Ns (int)``: number of samples per shot. See 3D cones
# - ``curve_index (float)``: Index controlling curvature from 0 (flat) to 1 (curvy).
#   ``(default 0.3)``
# - ``nb_revolutions (float)``: number of revolutions or elliptic periods.
#   ``(default 1)``
# - ``tilt (str, float)``: angle between each consecutive shot (in radians).
#   ``(default "uniform")``. See 3D cones
# - ``in_out (bool)``: define whether the shots should travel toward the center
#   then outside (in-out) or not (center-out). ``(default False)``. See 3D cones
#

trajectory = mn.initialize_3D_seiffert_spiral(Nc, Ns, in_out=in_out)
show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)


# %%
# ``curve_index (float)``
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# An index defined over :math:`[0, 1)` controling the curvature, with :math:`0`
# corresponding to a planar spiral, and increasing the length and exploration of
# the curve while asymptotically approaching :math:`1`.
#

arguments = [0, 0.3, 0.9, 0.99]
function = lambda x: mn.initialize_3D_seiffert_spiral(
    Nc, Ns, in_out=in_out, curve_index=x
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# ``nb_revolutions (float)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Number of revolutions, or simply the number of times a curve reaches its
# original orientation. For regular Seiffert spirals, it corresponds to the
# number of times the shot reaches the starting pole of the sphere. It
# subsequently defines the length of the curve.
#

arguments = [0.5, 1, 1.5, 2]
function = lambda x: mn.initialize_3D_seiffert_spiral(
    Nc, Ns, in_out=in_out, nb_revolutions=x
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


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
# is much older and can be traced back at least to [IN95]_.
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

trajectory = mn.initialize_3D_helical_shells(Nc, Ns, nb_shells=nb_repetitions)
show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)


# %%
# ``nb_shells (int)``
# ~~~~~~~~~~~~~~~~~~~
#
# Number of shells, i.e. concentric spheres, used to partition the k-space sphere.
#

arguments = [1, 2, nb_repetitions // 2, nb_repetitions]
function = lambda x: mn.initialize_3D_helical_shells(
    Nc=x, Ns=Ns, nb_shells=x, spiral_reduction=2
)
show_argument(function, arguments, one_shot=False, subfig_size=subfigure_size)


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
    Nc=Nc, Ns=Ns, nb_shells=nb_repetitions, spiral_reduction=x
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# ``shell_tilt (str, float)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Angle between each consecutive shells (in radians).
#

arguments = ["uniform", "intergaps", "golden", 3.1415]
function = lambda x: mn.initialize_3D_helical_shells(
    Nc=Nc, Ns=Ns, nb_shells=nb_repetitions, spiral_reduction=2, shell_tilt=x
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


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
    Nc=Nc, Ns=Ns, nb_shells=nb_repetitions, spiral_reduction=2, shot_tilt=x
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


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

trajectory = mn.initialize_3D_annular_shells(Nc, Ns, nb_shells=nb_repetitions)
show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)


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
# and it partitions the spheres into several connex curves composed of exactly
# two shots.
#

arguments = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
function = lambda x: mn.initialize_3D_annular_shells(
    Nc, Ns, nb_shells=nb_repetitions, ring_tilt=x
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# Seiffert shells
# ---------------
#
# An exclusive trajectory composed of re-arranged Seiffert spirals
# covering concentric shells. All curves have a constant speed and
# angular velocity, depending on the size of the sphere they belong to.
#
# This implementation is inspired by the propositions from [YRB06]_ and [SMR18]_,
# and also based on works from [Er00]_ and [Br09]_.
#
# Arguments:
#
# - ``Nc (int)``: number of individual shots. See 3D cones
# - ``Ns (int)``: number of samples per shot. See 3D cones
# - ``curve_index (float)``: Index controlling curvature from 0 (flat) to 1 (curvy).
#   ``(default 0.3)``. See Seiffert spirals
# - ``nb_revolutions (float)``: number of revolutions or elliptic periods.
#   ``(default 1)``.  See Seiffert spirals
# - ``shell_tilt (str, float)``: angle between each consecutive shell (in radians).
#   ``(default "intergaps")``. See helical shells
# - ``shot_tilt (str, float)``: angle between each consecutive shot
#   over a sphere (in radians). ``(default "uniform")``. See helical shells
#

trajectory = mn.initialize_3D_seiffert_shells(Nc, Ns, nb_shells=nb_repetitions)
show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)


# %%
# References
# ==========
#
# .. [IN95] Irarrazabal, Pablo, and Dwight G. Nishimura.
#    "Fast three dimensional magnetic resonance imaging."
#    Magnetic Resonance in Medicine 33, no. 5 (1995): 656-662.
# .. [Er00] Erdös, Paul.
#    "Spiraling the earth with C. G. J. Jacobi."
#    American Journal of Physics 68, no. 10 (2000): 888-895.
# .. [YRB06] Shu, Yunhong, Stephen J. Riederer, and Matt A. Bernstein.
#    "Three‐dimensional MRI with an undersampled spherical shells trajectory."
#    Magnetic Resonance in Medicine 56, no. 3 (2006): 553-562.
# .. [Br09] Brizard, Alain J.
#    "A primer on elliptic functions with applications in classical mechanics."
#    European journal of physics 30, no. 4 (2009): 729.
# .. [HM11] Gerlach, Henryk, and Heiko von der Mosel.
#    "On sphere-filling ropes."
#    The American Mathematical Monthly 118, no. 10 (2011): 863-876
# .. [Pip+11] Pipe, James G., Nicholas R. Zwart, Eric A. Aboussouan,
#    Ryan K. Robison, Ajit Devaraj, and Kenneth O. Johnson.
#    "A new design and rationale for 3D orthogonally
#    oversampled k‐space trajectories."
#    Magnetic resonance in medicine 66, no. 5 (2011): 1303-1311.
# .. [Bil+15] Bilgic, Berkin, Borjan A. Gagoski, Stephen F. Cauley, Audrey P. Fan,
#    Jonathan R. Polimeni, P. Ellen Grant, Lawrence L. Wald, and Kawin Setsompop.
#    "Wave‐CAIPI for highly accelerated 3D imaging."
#    Magnetic resonance in medicine 73, no. 6 (2015): 2152-2162.
# .. [SMR18] Speidel, Tobias, Patrick Metze, and Volker Rasche.
#    "Efficient 3D Low-Discrepancy k-Space Sampling
#    Using Highly Adaptable Seiffert Spirals."
#    IEEE Transactions on Medical Imaging 38, no. 8 (2018): 1833-1840.
# .. [SB21] Stobbe, Robert W., and Christian Beaulieu.
#    "Three‐dimensional Yarnball k‐space acquisition for accelerated MRI."
#    Magnetic Resonance in Medicine 85, no. 4 (2021): 1840-1854.
