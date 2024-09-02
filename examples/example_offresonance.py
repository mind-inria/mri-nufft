"""
======================
Off-resonance Corrected NUFFT Operator
======================

Example of Off-resonance Corrected NUFFT trajectory operator.

This examples show how to use the Off-resonance Corrected NUFFT operator to acquire 
and reconstruct data in presence of field inhomogeneities.
Here a spiral trajectory is used as a demonstration.

"""

import matplotlib.pyplot as plt
import numpy as np

from mrinufft import display_2D_trajectory

plt.rcParams["image.cmap"] = "gray"

# %%
# Data Generation
# ===============
# For realistic 2D image we will use a slice from the brainweb dataset.
# installable using ``pip install brainweb-dl``

from brainweb_dl import get_mri

mri_data = get_mri(0, "T1")
mri_data = mri_data[::-1, ...][90]
plt.imshow(mri_data), plt.axis("off"), plt.title("ground truth")

# %%
# Masking
# ===============
# Here, we generate a binary mask to exclude the background.
# We perform a simple binary threshold; in real-world application,
# it is advised to use other tools (e.g., FSL-BET).

brain_mask = mri_data > 0.1 * mri_data.max()
plt.imshow(brain_mask), plt.axis("off"), plt.title("brain mask")

# %%
# Field Generation
# ===============
# Here, we generate a radial B0 field with the same shape of
# the input Shepp-Logan phantom

from mrinufft.extras import make_b0map

# generate field
b0map, _ = make_b0map(mri_data.shape, b0range=(-200, 200), mask=brain_mask)
plt.imshow(brain_mask * b0map, cmap="bwr", vmin=-200, vmax=200), plt.axis(
    "off"
), plt.colorbar(), plt.title("B0 map [Hz]")

# %%
# Generate a Spiral trajectory
# ----------------------------

from mrinufft import initialize_2D_spiral
from mrinufft.density import voronoi
from mrinufft.trajectories.utils import DEFAULT_RASTER_TIME

samples = initialize_2D_spiral(Nc=48, Ns=600, nb_revolutions=10)
t_read = np.arange(samples.shape[1]) * DEFAULT_RASTER_TIME * 1e-3
t_read = np.repeat(t_read[None, ...], samples.shape[0], axis=0)
density = voronoi(samples)

display_2D_trajectory(samples)

# %%
# Setup the Operator
# ==================

from mrinufft import get_operator
from mrinufft.operators.off_resonance import MRIFourierCorrected

# Generate standard NUFFT operator
NufftOperator = get_operator("finufft")
nufft = NufftOperator(
    samples=samples,
    shape=mri_data.shape,
    density=density,
)

# Generate Fourier Corrected operator
mfi_nufft = MRIFourierCorrected(
    nufft, fieldmap=b0map, readout_time=t_read, mask=brain_mask
)

# Generate K-Space
kspace = mfi_nufft.op(mri_data)

# Reconstruct without field correction
mri_data_adj = nufft.adj_op(kspace)
mri_data_adj = np.squeeze(abs(mri_data_adj))

# Reconstruct with field correction
mri_data_adj_mfi = mfi_nufft.adj_op(kspace)
mri_data_adj_mfi = np.squeeze(abs(mri_data_adj_mfi))

fig2, ax2 = plt.subplots(1, 2)
ax2[0].imshow(mri_data_adj), ax2[0].axis("off"), ax2[0].set_title("w/o correction")
ax2[1].imshow(mri_data_adj_mfi), ax2[1].axis("off"), ax2[1].set_title("with correction")

plt.show()

# %%
# The blurring is significantly reduced using the Off-resonance Corrected
# operator (right)
