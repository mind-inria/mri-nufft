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

mri_data = get_mri(44, "T1")
mri_data = mri_data[::-1, ...][90]
plt.imshow(mri_data), plt.axis("off"), plt.title("ground truth")

# %%
# Field Generation
# ===============
# Here, we generate a radial B0 field with the same shape of
# the input Shepp-Logan phantom


def make_b0map(obj, b0range=(-300, 300)):
    """Make radial B0 field.

    Parameters
    ----------
    obj : np.ndarray
        Input object of shape (ny, nx).
    b0range : tuple, optional
        B0 field range in [Hz]. The default is (-300, 300).

    Returns
    -------
    b0map : np.ndarray
        Field inhomogeneities map of shape (ny, nx)
    """
    # calculate grid
    ny, nx = obj.shape
    yy, xx = np.mgrid[:ny, :nx]
    yy, xx = yy - ny // 2, xx - nx // 2
    yy, xx = yy / ny, xx / nx

    # radius
    rr = (xx**2 + yy**2) ** 0.5

    # mask
    mask = (obj != 0).astype(np.float32)
    b0map = mask * rr

    # rescale
    b0map = (b0range[1] - b0range[0]) * b0map / b0map.max() + b0range[0]  # Hz

    return b0map * mask


# generate field
b0map = make_b0map(mri_data)

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
    nufft, fieldmap=b0map, readout_time=t_read, mask=mri_data != 0
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
