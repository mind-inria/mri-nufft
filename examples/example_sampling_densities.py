"""
==================
Sampling densities
==================

A collection of sampling densities and density-based non-Cartesian trajectories.

"""

# %%
# In this example we illustrate the use of different sampling densities,
# and show how to generate trajectories based on them, such as random
# walks and travelling-salesman trajectories.
#

# External
import brainweb_dl as bwdl
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    show_density,
    show_densities,
    show_locations,
    show_trajectory,
    show_trajectories,
)

# Internal
import mrinufft as mn
from mrinufft import display_2D_trajectory, display_3D_trajectory


# %%
# Script options
# ==============
#
# These options are used in the examples below as default values for
# all densities and trajectories.

# Density parameters
shape_2d = (100, 100)
shape_3d = (100, 100, 100)

# Trajectory parameters
Nc = 10  # Number of shots
Ns = 50  # Number of samples per shot

# Display parameters
figure_size = 5.5  # Figure size for trajectory plots
subfigure_size = 3  # Figure size for subplots
one_shot = 0  # Highlight one shot in particular

# %%
# Densities
# =========
#
# In this section we present different sampling densities
# with various properties.
#
# Cutoff/decay density
# --------------------
#
# Create a density composed of a central constant-value ellipsis
# defined by a cutoff ratio, followed by a polynomial decay over
# outer regions as defined in [Cha+22]_.

cutoff_density = mn.create_cutoff_decay_density(shape=shape_2d, cutoff=0.2, decay=2)
show_density(cutoff_density, figure_size=figure_size)

# %%
# ``cutoff (float)``
# ~~~~~~~~~~~~~~~~~~
#
# The k-space radius cutoff ratio between 0 and 1 within
# which density remains uniform and beyond which it decays.
# It is modulated by ``resolution`` to create ellipsoids.
#
# The ``mrinufft.trajectories.sampling.create_polynomial_density``
# simply calls this function with ``cutoff=0``.

arguments = [0, 0.1, 0.2, 0.3]
function = lambda x: mn.create_cutoff_decay_density(
    shape=shape_2d,
    cutoff=x,
    decay=2,
)
show_densities(
    function,
    arguments,
    subfig_size=subfigure_size,
)


# %%
# ``decay (float)``
# ~~~~~~~~~~~~~~~~~
#
# The polynomial decay in density beyond the cutoff ratio.
# It can be zero or negative as shown below, but most applications
# are expected have decays in the positive range.


arguments = [-1, 0, 0.5, 2]
function = lambda x: mn.create_cutoff_decay_density(
    shape=shape_2d,
    cutoff=0.2,
    decay=x,
)
show_densities(
    function,
    arguments,
    subfig_size=subfigure_size,
)

# %%
# ``resolution (tuple)``
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Resolution scaling factors for each dimension of the density grid,
# by default ``None``. Note on the example below that the unit doesn't
# matter because ``cutoff`` is a ratio and ``decay`` is an exponent,
# so only the relative factor between the dimensions is important.
#
# This argument can be used to handle anisotropy but also to produce
# ellipsoidal densities.


arguments = [None, (1, 1), (1, 2), (1e-3, 0.5e-3)]
function = lambda x: mn.create_cutoff_decay_density(
    shape=shape_2d,
    cutoff=0.2,
    decay=2,
    resolution=x,
)
show_densities(
    function,
    arguments,
    subfig_size=subfigure_size,
)

# %%
# Energy-based density
# --------------------
#
# A common intuition is to consider that the sampling density
# should be proportional to the k-space amplitude. It can be
# learned from existing datasets and used for new acquisitions.

dataset = bwdl.get_mri(4, "T1")[:, ::2, ::2]
energy_density = mn.create_energy_density(dataset=dataset)
show_density(energy_density, figure_size=figure_size, log_scale=True)

# %%
# ``dataset (np.ndarray)``
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# The dataset from which to calculate the density
# based on its Fourier transform, with an expected
# shape (nb_volumes, dim_1, ..., dim_N).
# An N-dimensional Fourier transform is then performed.
#
# In the example below, we show the resulting densities
# from different slices of a single volume for convenience.
# More relevant use cases would be to learn densities for
# different organs and/or contrasts.


arguments = [50, 100, 150]
function = lambda x: mn.create_energy_density(dataset=bwdl.get_mri(4, "T1")[x : x + 20])
show_densities(
    function,
    arguments,
    subfig_size=subfigure_size,
    log_scale=True,
)


# %%
# Chauffert's density
# -------------------
#


chauffert_density = mn.create_chauffert_density(
    shape=shape_2d,
    wavelet_basis="haar",
    nb_wavelet_scales=3,
)
show_density(chauffert_density, figure_size=figure_size)


# %%
# ``wavelet_basis (str)``
# ~~~~~~~~~~~~~~~~~~~~~~~

arguments = ["haar", "rbio2.2", "coif4", "sym8"]
function = lambda x: mn.create_chauffert_density(
    shape=shape_2d,
    wavelet_basis=x,
    nb_wavelet_scales=3,
)
show_densities(
    function,
    arguments,
    subfig_size=subfigure_size,
)

# %%
# ``nb_wavelet_scales (int)``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

arguments = [1, 2, 3, 4]
function = lambda x: mn.create_chauffert_density(
    shape=shape_2d,
    wavelet_basis="haar",
    nb_wavelet_scales=x,
)
show_densities(
    function,
    arguments,
    subfig_size=subfigure_size,
)


# %%
# Custom density
# --------------

# Linear gradient
density = np.tile(np.linspace(0, 1, shape_2d[1])[:, None], (1, shape_2d[0]))
# Square center
density[
    3 * shape_2d[0] // 8 : 5 * shape_2d[0] // 8,
    3 * shape_2d[1] // 8 : 5 * shape_2d[1] // 8,
] = 2
# Outer circle
density = np.where(
    np.linalg.norm(np.indices(shape_2d) - shape_2d[0] / 2, axis=0) < shape_2d[0] / 2,
    density,
    0,
)
# Normalization
custom_density = density / np.sum(density)

show_density(custom_density, figure_size=figure_size)


# %%
# Sampling
# ========
#
# In this section we present random, pseudo-random and
# algorithm-based sampling methods. The examples are based
# on a few densities picked from the ones presented above.
#

densities = {
    "Cutoff/Decay": cutoff_density,
    "Energy": energy_density,
    "Chauffert": chauffert_density,
    "Custom": custom_density,
}

arguments = densities.keys()
function = lambda x: densities[x]
show_densities(function, arguments, subfig_size=subfigure_size)


# %%
# Random sampling
# ---------------

arguments = densities.keys()
function = lambda x: mn.sample_from_density(Nc * Ns, densities[x], method="random")
show_locations(function, arguments, subfig_size=subfigure_size)


# %%
# Lloyd's sampling
# ----------------

arguments = densities.keys()
function = lambda x: mn.sample_from_density(Nc * Ns, densities[x], method="lloyd")
show_locations(function, arguments, subfig_size=subfigure_size)


# %%
# Density-based trajectories
# ==========================
#
# In this section we present 2D and 3D trajectories based
# on arbitrary densities, and also sampling for some of them.
#
# Random walks
# ------------
#

arguments = densities.keys()
function = lambda x: mn.initialize_2D_random_walk(
    Nc, Ns, density=densities[x][::4, ::4]
)
show_trajectories(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%
#

arguments = densities.keys()
function = lambda x: mn.initialize_2D_random_walk(
    Nc, Ns, density=densities[x][::4, ::4], method="lloyd"
)
show_trajectories(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%
#
# Oversampled

arguments = densities.keys()
function = lambda x: mn.oversample(
    mn.initialize_2D_random_walk(
        Nc, Ns, density=densities[x][::4, ::4], method="lloyd"
    ),
    4 * Ns,
)
show_trajectories(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# Travelling Salesman
# -------------------
#

arguments = densities.keys()
function = lambda x: mn.initialize_2D_travelling_salesman(
    Nc,
    Ns,
    density=densities[x],
)
show_trajectories(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%
#

arguments = densities.keys()
function = lambda x: mn.initialize_2D_travelling_salesman(
    Nc,
    Ns,
    density=densities[x],
    method="lloyd",
)
show_trajectories(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%
#
# Oversampled

arguments = densities.keys()
function = lambda x: mn.oversample(
    mn.initialize_2D_travelling_salesman(Nc, Ns, density=densities[x], method="lloyd"),
    4 * Ns,
)
show_trajectories(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)

# %%
#

arguments = ((None, None, None), ("y", None, "x"), ("phi", None, "r"), ("y", "x", "r"))
function = lambda x: mn.initialize_2D_travelling_salesman(
    Nc,
    Ns,
    density=densities["Custom"],
    first_cluster_by=x[0],
    second_cluster_by=x[1],
    sort_by=x[2],
    method="lloyd",
)
show_trajectories(function, arguments, one_shot=one_shot, subfig_size=subfigure_size)


# %%
# References
# ==========
#
# .. [Cha+22] Chaithya, G. R., Pierre Weiss, Guillaume Daval-Frérot,
#    Aurélien Massire, Alexandre Vignaud, and Philippe Ciuciu.
#    "Optimizing full 3d sparkling trajectories for high-resolution
#    magnetic resonance imaging."
#    IEEE Transactions on Medical Imaging 41, no. 8 (2022): 2105-2117.
