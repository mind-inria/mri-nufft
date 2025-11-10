"""
Sensitivity maps estimation
===========================

This example demonstrates how to estimate coil sensitivity maps from
non-Cartesian k-space data using different methods provided in the
:mrinufft:`mrinufft.extras.smaps` module.
We will simulate k-space data from a known MRI image and coil sensitivity
maps, and then estimate the sensitivity maps using the ESPIRiT method [espirit]_ and
a low-frequency calibration method [sense]_.
We will visualize the estimated sensitivity maps and compare them to the
actual sensitivity maps used in the simulation.
"""

# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install mri-nufft[gpunufft] cufinufft sigpy scikit-image

# %%
# Imports
# -------
import matplotlib.pyplot as plt
import numpy as np
from brainweb_dl import get_mri
from sigpy.mri.sim import birdcage_maps

from mrinufft import get_operator
from mrinufft.extras.smaps import get_smaps
from mrinufft.trajectories import initialize_3D_floret
import cupy as cp
import os


# %%
# Function to display imgs
def show_maps(imgs):
    """Display 4D sensitivity maps in a 4x3 figure layout."""
    n_coils, nx, ny, nz = imgs.shape
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    imgs = np.abs(imgs)
    for i in range(n_coils):
        axes[i, 0].imshow(imgs[i, nx // 2, :, :], vmax=imgs.max(), vmin=imgs.min())
        axes[i, 1].imshow(imgs[i, :, ny // 2, :], vmax=imgs.max(), vmin=imgs.min())
        axes[i, 2].imshow(imgs[i, :, :, nz // 2], vmax=imgs.max(), vmin=imgs.min())
        axes[i, 0].set_title(f"Coil {i+1} - YZ plane")
        axes[i, 1].set_title(f"Coil {i+1} - XZ plane")
        axes[i, 2].set_title(f"Coil {i+1} - XY plane")
    plt.show()
    return fig


BACKEND = os.environ.get("MRINUFFT_BACKEND", "cufinufft")

# %%
# Get MRI data, 3D FLORET trajectory, and simulate k-space data
samples_loc = initialize_3D_floret(Nc=16 * 16, Ns=256)
mri = get_mri(0)[::2, ::2, ::2][::-1, ::-1]  # Load and downsample MRI data for speed
n_coils = 4
actual_smaps = birdcage_maps(
    (n_coils, *mri.shape), dtype=np.complex64
)  # Generate birdcage sensitivity maps

# %%
# Show the sensitivity maps
show_maps(actual_smaps)

# %%
# Show the per channel images

per_ch_mri = mri[None, ...] * actual_smaps  # Generate per-coil MRI data
show_maps(per_ch_mri)

# %%
# Simulate k-space data
forward_op = get_operator(BACKEND)(
    samples_loc, shape=mri.shape, n_coils=n_coils, density=True
)
kspace_data = forward_op.op(per_ch_mri)  # Simulate k-space data

if str.lower(BACKEND) in ["gpunufft", "cufinufft"]:
    # GPU exists, run on GPU
    import cupy as cp

    kspace_data = cp.asarray(kspace_data, dtype=cp.complex64)


# %%
# Estimate sensitivity maps using ESPIRiT
Smaps = get_smaps("espirit")(
    samples_loc,
    mri.shape,
    kspace_data=kspace_data,
    density=forward_op.density,
    backend=BACKEND,
    decim=4,
)
show_maps(Smaps.get())

# %%
# Estimate the sensitivity map using low-frequency calibration
Smaps = get_smaps("low_frequency")(
    samples_loc,
    mri.shape,
    kspace_data=kspace_data,
    density=forward_op.density,
    backend=BACKEND,
)
show_maps(Smaps.get())

# %%
# References
# ==========
#
# .. [sense] Loubna El Gueddari, C. Lazarus, H Carrié, A. Vignaud, Philippe Ciuciu.
#           Self-calibrating nonlinear reconstruction algorithms for variable density
#           sampling and parallel reception MRI. 10th IEEE Sensor Array and Multichannel
#           Signal Processing workshop, Jul 2018, Sheffield, United Kingdom. ⟨hal-01782428v1⟩
# .. [espirit] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS,
#               Lustig M. ESPIRiT--an eigenvalue approach to autocalibrating parallel
#               MRI: where SENSE meets GRAPPA. Magn Reson Med. 2014 Mar;71(3):990-1001.
#               doi: 10.1002/mrm.24751. PMID: 23649942; PMCID: PMC4142121.
