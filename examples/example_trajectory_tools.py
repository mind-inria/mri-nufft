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
# Also note that not all possibilities are relevant for MR applications
# since these functions are only tools to simplify trajectory design.
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

from mrinufft.trajectories.utils import KMAX
from utils import show_argument, show_trajectory


# %%
# Script options
# ==============
# These options are used in the examples below as default values for all trajectories.

# Trajectory parameters
Nc = 100  # Number of shots
Ns = 500  # Number of samples per shot
in_out = True  # Choose between in-out or center-out trajectories
nb_repetitions = 8  # Number of stacks, rotations, cones, shells etc.
nb_revolutions = 1  # Number of revolutions for base trajectories
nb_zigzags = 5  # Number of zigzags for base trajectories

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
# Single shots from 2D radial, Fermat's spiral, and 2D/3D cones.
#

single_trajectories = {
    "Radial": mn.initialize_2D_radial(1, Ns, in_out=in_out),
    "Spiral": mn.initialize_2D_spiral(
        1, Ns, in_out=in_out, spiral="fermat", nb_revolutions=nb_revolutions
    ),
    "2D Cones": mn.initialize_2D_cones(
        Nc // nb_repetitions, Ns, in_out=in_out, nb_zigzags=nb_zigzags
    )[:1],
    "3D Cones": mn.initialize_3D_cones(Nc, Ns, in_out=in_out, nb_zigzags=nb_zigzags)[
        :1
    ],
}

# %%

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: single_trajectories[x]
show_argument(function, arguments, one_shot=bool(one_shot), subfig_size=subfigure_size)


# %%
# Planes
# ~~~~~~
#
# We will also be using them as planes, or thick planes for 3D shots.
#

Nc_planes = Nc // nb_repetitions
z_tilt = 2 * np.pi / Nc_planes / (1 + in_out)

planar_trajectories = {
    "Radial": tools.rotate(
        single_trajectories["Radial"], nb_rotations=Nc_planes, z_tilt=z_tilt
    ),
    "Spiral": tools.rotate(
        single_trajectories["Spiral"], nb_rotations=Nc_planes, z_tilt=z_tilt
    ),
    "2D Cones": tools.rotate(
        single_trajectories["2D Cones"], nb_rotations=Nc_planes, z_tilt=z_tilt
    ),
    "3D Cones": tools.rotate(
        single_trajectories["3D Cones"], nb_rotations=Nc_planes, z_tilt=z_tilt
    ),
}

# %%

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: planar_trajectories[x]
show_argument(function, arguments, one_shot=bool(one_shot), subfig_size=subfigure_size)


# %%
# Stack
# -----
#
# The oldest and most widely used method is to simply stack any plane
# on top of itself, until it reaches the desired number of slices.
#
# Arguments:
#
# - ``trajectory (array)``: array of k-space coordinates of
#   size :math:`(N_c, N_s, N_d)`
# - ``nb_stacks (int)``: number of stacks repeating ``trajectory``
#   over the :math:`k_z`-axis.
# - ``z_tilt (float)``: angle tilt between consecutive stacks
#   over the :math:`k_z`-axis. ``(default None)``
# - ``hard_bounded (bool)``: whether the stacks should be
#   strictly bounded to k-space. ``(default True)``
#

trajectory = tools.stack(planar_trajectories["Radial"], nb_stacks=nb_repetitions)
show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)

# %%
# ``trajectory (array)``
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The main use case is to stack trajectories consisting of
# flat or thick planes that will match the image slices.
# Some stacks can also be removed afterward to create GRAPPA-like
# patterns that will rely on parallel imaging and sensitivity maps
# to recover the missing information.
# Stacking planes without tilting them is notably compatible
# with stacked-NUFFT operators, reducing time and memory
# consumption during reconstruction.
#

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.stack(planar_trajectories[x], nb_stacks=nb_repetitions)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)
# %%
show_argument(
    function,
    arguments,
    one_shot=one_shot,
    subfig_size=subfigure_size,
    dim="2D",
    axes=(0, 2),
)

# %%
#
# It can also be applied twice to single shots to create
# a plane before stacking it over the :math:`k_z`-axis.
# Note here that is does not make a lot of sense for
# non-radial trajectories such as spirals.
#

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
# ``hard_bounded (bool)``
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# The stack is ``hard_bounded`` when the positions of the stacks
# over the :math:`k_z`-axis are squeezed such that the lower and upper
# stacks don't exceed the k-space boundaries when the plane is thick.
# In the example below, the thickness is deliberately increased to
# emphasize this point.
#

arguments = [True, False]
trajectory = np.copy(planar_trajectories["3D Cones"])
trajectory[..., 2] *= 2
function = lambda x: tools.stack(trajectory, nb_stacks=nb_repetitions, hard_bounded=x)
show_argument(
    function,
    arguments,
    one_shot=one_shot,
    subfig_size=subfigure_size,
    dim="2D",
    axes=(0, 2),
)


# %%
# Rotate
# ------
#
# A simple method to duplicate any trajectory with a rotation over
# one or multiple axes at the same time.
#
# Arguments:
#
# - ``trajectory (array)``: array of k-space coordinates of
#   size :math:`(N_c, N_s, N_d)`
# - ``nb_rotations (int)``: number of rotations repeating ``trajectory``.
# - ``x_tilt (float)``: angle tilt between consecutive stacks
#   over the :math:`k_x`-axis. ``(default None)``
# - ``y_tilt (float)``: angle tilt between consecutive stacks
#   over the :math:`k_y`-axis. ``(default None)``
# - ``z_tilt (float)``: angle tilt between consecutive stacks
#   over the :math:`k_z`-axis. ``(default None)``
#

trajectory = tools.rotate(
    planar_trajectories["Radial"], nb_rotations=nb_repetitions, x_tilt="uniform"
)
show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)

# %%
# ``trajectory (array)``
# ~~~~~~~~~~~~~~~~~~~~~~
#
# A common application is to rotate a single shot to create a plane
# as used earlier to initialize the planar trajectories. It has also
# been used in the literature to rotate planes around one axis to
# create 3D trajectories, but the density (and redundancy) along that
# axis is then much greater than anywhere else.
#

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.rotate(
    planar_trajectories[x],
    nb_rotations=nb_repetitions,
    x_tilt="uniform",
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)
# %%
show_argument(
    function,
    arguments,
    one_shot=one_shot,
    subfig_size=subfigure_size,
    dim="2D",
    axes=(1, 2),
)


# %%
# Precess
# -------
#
# A method to duplicate a trajectory while applying a
# precession-like rotation around a provided axis.
#
# Arguments:
#
# - ``trajectory (array)``: array of k-space coordinates of
#   size :math:`(N_c, N_s, N_d)`
# - ``nb_rotations (int)``: number of rotations repeating ``trajectory``
#   over the :math:`k_z`-axis.
# - ``tilt (float)``: angle tilt between consecutive rotations
#   around the :math:`k_z`-axis. ``(default "golden")``
# - ``half_sphere (bool)``: whether the precession should be limited
#   to the upper half of the k-space sphere, typically for in-out
#   trajectories or planes. ``(default False)``
# - ``partition (str)``: partition type between an "axial" or "polar"
#   split of the :math:`k_z`-axis, designating whether the axis should
#   be fragmented by radius or angle respectively. ``(default "axial")``
# - ``axis (int, array)``: axis selected for alignment reference
#   when rotating the trajectory around the :math:`k_z`-axis,
#   generally corresponding to the shot direction for
#   single shot ``trajectory`` inputs. It can either be an integer for
#   one of the three k-space axes, or directly a 3D array.
#   The default behavior when ``None`` is to select the last coordinate
#   of the first shot as the axis. ``(default None)``
#

trajectory = tools.precess(
    planar_trajectories["Radial"],
    nb_rotations=nb_repetitions,
    tilt="golden",
    half_sphere=in_out,
    axis=2,
)
show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)

# %%
# ``trajectory (array)``
# ~~~~~~~~~~~~~~~~~~~~~~
#
# This method provides a way to distribute duplicated trajectories
# (single shots, planes or anything else) to cover evenly a provided
# axis tilting the azimuthal orientation.
#

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.precess(
    planar_trajectories[x],
    nb_rotations=nb_repetitions,
    tilt="golden",
    half_sphere=in_out,
    axis=2,
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%
#
# It is however most often used with single shots to
# cover more evenly the k-space sphere, such as with 3D cones
# or Seiffert spirals. Indeed, applying a precession with
# the golden angle is known to approximate an even distribution
# of points over a sphere surface.
#

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.precess(
    single_trajectories[x],
    nb_rotations=Nc,
    tilt="golden",
    half_sphere=in_out,
    axis=0,
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# ``half_sphere (bool)``
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Whether the precession should be limited to the upper half
# of the k-space sphere (with respect to the provided axis).
# It is typically used for in-out trajectories or planes, as
# otherwise shots would likely be stacked in a redundant way.
#
# In the example hereafter, center-out shots are shown for clarity.
#


arguments = [True, False]
function = lambda x: tools.precess(
    single_trajectories["Radial"][:, Ns // (1 + in_out) :],
    nb_rotations=Nc,
    tilt="golden",
    half_sphere=x,
    axis=0,
)
show_argument(
    function,
    arguments,
    one_shot=one_shot,
    subfig_size=subfigure_size,
    dim="2D",
    axes=(0, 2),
)


# %%
# ``partition (str)``
# ~~~~~~~~~~~~~~~~~~~
#
# Partition type between an "axial" or "polar"
# split of the :math:`k_z`-axis, designating whether the axis should
# be fragmented by radius or angle respectively.
#

arguments = ["axial", "polar"]
function = lambda x: tools.precess(
    single_trajectories["Radial"],
    nb_rotations=Nc,
    tilt=None,
    partition=x,
    axis=0,
)
show_argument(
    function,
    arguments,
    one_shot=one_shot,
    subfig_size=subfigure_size,
    dim="2D",
    axes=(0, 2),
)

# %%
#
# While "polar" looks more natural in the absence of rotation (``tilt=None``),
# it results in too many shots close to the rotation axis, and therefore
# a non-uniform density. The best approximation of a uniform distribution
# is obtained with an "axial" partition and "golden" tilt along
# the provided axis.
#

arguments = ["axial", "polar"]
function = lambda x: tools.precess(
    single_trajectories["Radial"],
    nb_rotations=Nc,
    tilt="golden",
    partition=x,
    axis=0,
)
show_argument(
    function,
    arguments,
    one_shot=one_shot,
    subfig_size=subfigure_size,
    dim="2D",
    axes=(0, 2),
)

# %%
#
# The distribution over the k-space sphere surface can be shown by
# displaying only the tip of the shots.
#

arguments = ["axial", "polar"]
function = lambda x: tools.precess(
    single_trajectories["Radial"][:, -5:],
    nb_rotations=Nc,
    tilt="golden",
    partition=x,
    axis=0,
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%
# ``axis (int, array)``
# ~~~~~~~~~~~~~~~~~~~~~
#
# Axis selected for alignment reference when rotating the trajectory
# around the :math:`k_z`-axis, generally corresponding to the
# shot direction for single shot ``trajectory`` inputs.
# It can either be an integer for one of the three k-space axes,
# or directly a 3D array. The default behavior when `None`
# is to select the last coordinate of the first shot as the axis.
#
# This argument is simple to select but still important, as the
# precession relies on Rodrigues' rotation coefficients that enable
# a rotation from one vector to another to align the trajectory
# through the provided axis with the precession vectors all over
# the k-space sphere. However, misalignement between shots and the
# provided axis will result in a non-uniform distribution, as the
# rotation around the axis is unfavorably deterministic.
#
# The first case is single shots, where the provided axis should
# simply correspond to the shot axis.
#

arguments = [None, 0, 1, 2]
function = lambda x: tools.precess(
    single_trajectories["Radial"],
    nb_rotations=Nc,
    tilt="golden",
    half_sphere=in_out,
    axis=x,
)
show_argument(
    function,
    arguments,
    one_shot=one_shot,
    subfig_size=subfigure_size,
    dim="2D",
    axes=(1, 2),
)

# %%
#
# The second case is planar trajectories, where the axis orthogonal
# to the shots plane is preferred.
#

arguments = [None, 0, 1, 2]
function = lambda x: tools.precess(
    planar_trajectories["Radial"],
    nb_rotations=nb_repetitions,
    tilt="golden",
    half_sphere=in_out,
    axis=x,
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%
#
# Some trickier cases exist in the literature, with the example of Seiffert spirals.
# Those 3D spirals neither correspond to a single-axis shot or a plane, so the authors
# chose to use the center-out axis of each shot as a reference axis for the rotation.
# In order to handle the redundant distribution, they added a pseudo-random rotation
# within the shot axes.
#


# %%
# Conify
# ------
#
# A tool to distort trajectories into multiple cones
# positioned to cover the k-space sphere.
#
# Arguments:
#
# - ``trajectory (array)``: array of k-space coordinates of
#   size :math:`(N_c, N_s, N_d)`
# - ``nb_cones (int)``: number of cones repeating ``trajectory``
#   with conical distortion over the :math:`k_z`-axis.
# - ``z_tilt (float)``: angle tilt between consecutive cones
#   around the :math:`k_z`-axis. ``(default "golden")``
# - ``in_out (bool)``: whether to account for the in-out
#   nature of some trajectories to avoid hard angles
#   around the center, ``(default False)``
# - ``max_angle (float)``: maximum angle of the cones. ``(default pi / 2)``
# - ``borderless (bool)``: Whether the cones should reach `max_angle` or not,
#   mostly to avoid 1D cones if ``max_angle`` is equal to pi / 2, by default True.
#

trajectory = tools.conify(
    planar_trajectories["Radial"], nb_cones=nb_repetitions, in_out=in_out
)
show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)

# %%
# ``trajectory (array)``
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The trajectory is folded toward the :math:`k_z`-axis to shape cones,
# and is therefore expected to be planar over the :math:`k_x-k_y` axes.
# Other configuration might result in irrelevant trajectories.
# Also, the distortion is likely to increase the required gradient amplitudes
# and slew rates.
#

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.conify(
    planar_trajectories[x], nb_cones=nb_repetitions, in_out=in_out
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)
# %%
show_argument(
    function,
    arguments,
    one_shot=one_shot,
    subfig_size=subfigure_size,
    dim="2D",
    axes=(0, 2),
)

# %%
#
# Similarly to other tools, it can be used with single shots.
# In that case, ``nb_cones`` is set to ``Nc`` to create as many
# individual cones.
#

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.conify(
    single_trajectories[x], nb_cones=Nc, z_tilt="golden", in_out=in_out
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)
# %%
show_argument(
    function,
    arguments,
    one_shot=one_shot,
    subfig_size=subfigure_size,
    dim="2D",
    axes=(0, 2),
)

# %%
# ``max_angle (float)``
# ~~~~~~~~~~~~~~~~~~~~~
#
# Polar angle of the most folded cone. As pointed out in [Pip+11]_,
# folding planes over the whole sphere would result in inefficient
# distributions near the :math:`k_z`-axis, and it may be more relevant
# to reduce the maximum angle but duplicate all of the cones along
# another axis to still cover the whole k-space.
#

arguments = [np.pi / 2, np.pi / 3, np.pi / 4, np.pi / 5]
function = lambda x: tools.conify(
    planar_trajectories["Radial"],
    nb_cones=nb_repetitions,
    in_out=in_out,
    max_angle=x,
)
show_argument(
    function,
    arguments,
    one_shot=one_shot,
    subfig_size=subfigure_size,
    dim="2D",
    axes=(0, 2),
)


# %%
# ``borderless (bool)``
# ~~~~~~~~~~~~~~~~~~~~~
#
# Define whether or not the edge cones should reach ``max_angle``
# when equal to ``False``, or instead simply partition the
# sphere over a polar split.
#

arguments = [True, False]
function = lambda x: tools.conify(
    planar_trajectories["Radial"],
    nb_cones=nb_repetitions,
    in_out=in_out,
    max_angle=np.pi / 2,
    borderless=x,
)
show_argument(
    function,
    arguments,
    one_shot=one_shot,
    subfig_size=subfigure_size,
    dim="2D",
    axes=(0, 2),
)


# %%
# Functional tools
# ================

# %%
# Preparation
# -----------
#
# We can define a few functions that will be used in the following
# examples, using again 2D radial, Fermat's spiral, and 2D/3D cones:
#

init_trajectories = {
    "Radial": lambda Nc: mn.initialize_2D_radial(Nc, Ns, in_out=in_out),
    "Spiral": lambda Nc: mn.initialize_2D_spiral(
        Nc, Ns, in_out=in_out, spiral="fermat", nb_revolutions=nb_revolutions
    ),
    "2D Cones": lambda Nc: mn.initialize_2D_cones(
        Nc, Ns, in_out=in_out, nb_zigzags=nb_zigzags
    ),
    "3D Cones": lambda Nc: tools.rotate(
        single_trajectories["3D Cones"],
        nb_rotations=Nc,
        z_tilt=2 * np.pi / Nc / (1 + in_out),
    ),
}


# %%
# Stack spherically
# -----------------
#
# A tool similar to ``tools.stack`` but with stacks shrinked
# in order to cover the k-space sphere and a variable number
# of shot per stack to improve the coverage over larger stacks.
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

trajectory = tools.stack_spherically(
    init_trajectories["Radial"], Nc=Nc, nb_stacks=nb_repetitions
)
show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)

# %%
# ``trajectory_func (function)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A function that takes at least one argument ``Nc`` to control
# the number of shots, in order to adapt that value for each stack
# and focus more ressources over larger areas. In opposition to
# ``tools.stack``, it is not possible to use stacked-NUFFT operators
# with ``tools.stack_spherically``.
#

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.stack_spherically(
    init_trajectories[x], Nc=Nc, nb_stacks=nb_repetitions
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)
# %%
show_argument(
    function,
    arguments,
    one_shot=one_shot,
    subfig_size=subfigure_size,
    dim="2D",
    axes=(0, 2),
)

# %%
#
# In the previous example, we can observe that spirals and cones
# are nicely adapted for each stack, while shrinking the shots
# for the radial trajectory is quite irrelevant (coverage is not
# improved). Instead, each radial disc could be normalized to
# cover a cylinder with variable density over :math:`k_z`.
#

traj_classic = tools.stack_spherically(
    init_trajectories["Radial"], Nc=Nc, nb_stacks=nb_repetitions
)
traj_normal = np.copy(traj_classic)
traj_normal[..., :2] = (
    KMAX
    * traj_normal[..., :2]
    / np.max(
        np.linalg.norm(traj_classic[..., :2], axis=2, keepdims=True),
        axis=1,
        keepdims=True,
    )
)

trajectories = {"Classic": traj_classic, "Normalized": traj_normal}
arguments = ["Classic", "Normalized"]
function = lambda x: trajectories[x]
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# Shellify
# --------
#
# A tool to carve trajectories into half-spheres/domes and duplicate
# them into concentric shells composed of a variable number
# of shots depending on their size.
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
# - ``hemisphere_mode (str)``: define how the lower hemisphere should
#   be oriented relatively to the upper one, with "symmetric" providing
#   a kx-ky planar symmetry by changing the polar angle, and with
#   "reversed" promoting continuity (for example in spirals) by
#   reversing the azimuthal angle. ``(default "symmetric")``.
# - ``**kwargs``: trajectory initialization parameters for the
#   function provided with ``trajectory_func``.
#

trajectory = tools.shellify(
    init_trajectories["Radial"], Nc=Nc, nb_shells=nb_repetitions
)
show_trajectory(trajectory, figure_size=figure_size, one_shot=one_shot)

# %%
# ``trajectory_func (function)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A function that takes at least one argument ``Nc`` to control
# the number of shots, in order to adapt that value for each shell
# and focus more ressources over larger spheres.
#
# Gradient amplitudes and slew rates are likely to be increased
# near the edges, and it should be accounted for.
# Companion functions will be added in the future in order
# to manipulate individual spheres.
#

arguments = ["Radial", "Spiral", "2D Cones", "3D Cones"]
function = lambda x: tools.shellify(
    init_trajectories[x], Nc=Nc, nb_shells=nb_repetitions
)
show_argument(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%
# ``hemisphere_mode (str)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Define how the lower hemisphere should be oriented relatively
# to the upper one, with "symmetric" providing a :math:`k_x-k_y`
# planar symmetry by changing the polar angle, and with "reversed"
# promoting continuity (for example in spirals) by reversing
# the azimuthal angle.
#

arguments = ["symmetric", "reversed"]
function = lambda x: tools.shellify(
    init_trajectories["Spiral"], Nc=Nc, nb_shells=nb_repetitions, hemisphere_mode=x
)
show_argument(
    function,
    arguments,
    one_shot=one_shot,
    subfig_size=subfigure_size,
    dim="2D",
    axes=(0, 2),
)


# %%
# References
# ==========
#
# .. [Pip+11] Pipe, James G., Nicholas R. Zwart, Eric A. Aboussouan,
#    Ryan K. Robison, Ajit Devaraj, and Kenneth O. Johnson.
#    "A new design and rationale for 3D orthogonally
#    oversampled k‐space trajectories."
#    Magnetic resonance in medicine 66, no. 5 (2011): 1303-1311.
