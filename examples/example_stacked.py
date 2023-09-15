"""
======================
Stacked NUFFT Operator
======================

Example of Stacked NUFFT trajectory operator.

This examples show how to use the Stacked NUFFT operator to acquire and reconstruct data
in kspace where the sampling of pattern is a stack of non cartesian trajectory.
Here a stack of spiral is used as a demonstration.

"""

import matplotlib.pyplot as plt
import numpy as np
from mrinufft import display_2D_trajectory

plt.rcParams["image.cmap"] = "gray"

# %%
# Data Generation
# ===============
# For realistic 3D images we will use the brainweb dataset.
# installable using ``pip install brainweb-dl``

from brainweb_dl import get_mri

mri_data = get_mri(0, "T1")
mri_data = mri_data[::-1, ...]
fig, ax = plt.subplots(1, 3)
ax[0].imshow(mri_data[90, :, :])
ax[1].imshow(mri_data[:, 108, :])
ax[2].imshow(mri_data[:, :, 90])

# %%
# Generate a Spiral trajectory
# ----------------------------

from mrinufft import initialize_2D_spiral
from mrinufft.trajectories.density import voronoi

samples = initialize_2D_spiral(Nc=16, Ns=500, nb_revolutions=10)
density = voronoi(samples)

display_2D_trajectory(samples)
# specify locations for the stack of trajectories.
kz_slices = np.arange(mri_data.shape[-1])

# %%
# Setup the Operator
# ==================

from mrinufft.operators.stacked import MRIStackedNUFFT

stacked_nufft = MRIStackedNUFFT(
    samples=samples,
    shape=mri_data.shape,
    z_index=kz_slices,
    backend="finufft",
    n_coils=1,
    smaps=None,
    density=density,
)

kspace_stack = stacked_nufft.op(mri_data)
print(kspace_stack.shape)

mri_data_adj = stacked_nufft.adj_op(kspace_stack)
mri_data_adj = np.squeeze(abs(mri_data_adj))
print(mri_data_adj.shape)

fig2, ax2 = plt.subplots(1, 3)
ax2[0].imshow(mri_data_adj[90, :, :])
ax2[1].imshow(mri_data_adj[:, 108, :])
ax2[2].imshow(mri_data_adj[:, :, 90])

plt.show()
