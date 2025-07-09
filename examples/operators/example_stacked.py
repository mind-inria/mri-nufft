"""
======================
Stacked NUFFT operator
======================

An example to show how to setup a stacked NUFFT operator.

This example shows how to use the stacked NUFFT operator to reconstruct data
when the sampling pattern in k-space is a stack of 2D non-Cartesian trajectories.
Hereafter a stack of 2D spirals is used for demonstration.

"""

import matplotlib.pyplot as plt
import numpy as np

from mrinufft import display_2D_trajectory

plt.rcParams["image.cmap"] = "gray"


# %%
# Data preparation
# ================
#
# Image loading
# -------------
#
# For realistic 3D images we will use the BrainWeb dataset,
# installable using ``pip install brainweb-dl``.

from brainweb_dl import get_mri

mri_data = get_mri(0, "T1")
mri_data = np.flip(mri_data, axis=(0, 1, 2))

# %%

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].imshow(mri_data[90, :, :])
ax[1].imshow(mri_data[:, 108, :])
ax[2].imshow(mri_data[:, :, 90])
plt.show()


# %%
# Trajectory generation
# ---------------------
#
# Only the 2D pattern needs to be initialized, along with
# its density to improve the adjoint NUFFT operation and
# the location of the different slices.
#

from mrinufft import initialize_2D_spiral
from mrinufft.density import voronoi

samples = initialize_2D_spiral(Nc=16, Ns=500, nb_revolutions=10)
density = voronoi(samples)
kz_slices = np.arange(mri_data.shape[-1])  # Specify locations for the stacks.

# %%

display_2D_trajectory(samples)
plt.show()


# %%
# Operator setup
# ==============

from mrinufft.operators.stacked import MRIStackedNUFFT

stacked_nufft = MRIStackedNUFFT(
    samples=2 * np.pi * samples,  # normalize for finufft
    shape=mri_data.shape,
    z_index=kz_slices,
    backend="finufft",
    n_coils=1,
    smaps=None,
    density=density,
)

kspace_stack = stacked_nufft.op(mri_data)
print(f"K-space shape: {kspace_stack.shape}")

mri_data_adj = stacked_nufft.adj_op(kspace_stack)
mri_data_adj = np.squeeze(abs(mri_data_adj))
print(f"Volume shape: {mri_data_adj.shape}")

# %%

fig2, ax2 = plt.subplots(1, 3, figsize=(10, 3))
ax2[0].imshow(mri_data_adj[90, :, :])
ax2[1].imshow(mri_data_adj[:, 108, :])
ax2[2].imshow(mri_data_adj[:, :, 90])
plt.show()
