"""
Sensitivity maps estimation
===========================

An example to show how to perform a simple NUFFT.
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


BACKEND = os.environ.get("MRINUFFT_BACKEND", "gpunufft")

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

# Estimate sensitivity maps from k-space data using different method
smaps_methods = ["espirit", "low_frequency"]
if BACKEND == "gpunufft":
    # GPU exists, run on GPU
    import cupy as cp
    kspace_data = cp.asarray(kspace_data, dtype=cp.complex64)
for method in smaps_methods:
    extra_kwargs = {}
    if method == "espirit":
        extra_kwargs["decim"] = 4
    Smaps = get_smaps(method)(
        samples_loc,
        mri.shape,
        kspace_data=kspace_data,
        density=forward_op.density,
        backend=BACKEND,
        **extra_kwargs,
    )
    show_maps(Smaps.get())
