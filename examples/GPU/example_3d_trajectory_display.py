"""
======================
3D Trajectory Display
======================

In this example, we show some tools available through the `mrinufft` package to display 3D trajectories.
This is useful to understand the sampling pattern of the k-space data, and to visualize the trajectory, see sampling time, gradient strength, slew rates etc.
Also, another key useful feature is that it enables us to see the density of sampling pattern in the k-space, and help analyze k-space holes, which can help debug artifacts in reconstructions.
"""

# %%
# Imports
from mrinufft.trajectories.display3D import get_gridded_trajectory
import mrinufft.trajectories.trajectory3D as mtt
from mrinufft.trajectories.utils import Gammas
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np


# %%
# Utility function to plot mid-plane slices for 3D volumes
def plot_slices(axs, volume, title=""):
    def set_labels(ax, axis_num=None):
        ax.set_xticks([0, 32, 64])
        ax.set_yticks([0, 32, 64])
        ax.set_xticklabels([r"$-\pi$", 0, r"$\pi$"])
        ax.set_yticklabels([r"$-\pi$", 0, r"$\pi$"])
        if axis_num is not None:
            ax.set_xlabel(r"$k_" + "zxy"[axis_num] + r"$")
            ax.set_ylabel(r"$k_" + "yzx"[axis_num] + r"$")

    for i in range(3):
        volume = np.rollaxis(volume, i, 0)
        axs[i].imshow(volume[volume.shape[0] // 2])
        axs[i].set_title(
            ((title + f"\n") if i == 0 else "") + r"$k_{" + "xyz"[i] + r"}=0$"
        )
        set_labels(axs[i], i)


def create_grid(grid_type, title="", **kwargs):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i, (name, traj) in enumerate(trajectories.items()):
        grid = get_gridded_trajectory(
            traj, (64, 64, 64), grid_type=grid_type, traj_params=traj_params, **kwargs
        )
        plot_slices(axs[:, i], grid, title=name)
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()


# %%
# Create a bunch of sample trajectories
trajectories = {
    "Radial": mtt.initialize_3D_phyllotaxis_radial(64 * 8, 64),
    "FLORETs": mtt.initialize_3D_floret(64 * 8, 64, nb_revolutions=2),
    "Yarn Ball": mtt.initialize_3D_seiffert_spiral(64 * 8, 64),
}
traj_params = {
    "FOV": (0.23, 0.23, 0.23),
    "img_size": (64, 64, 64),
    "gamma": Gammas.HYDROGEN,
}

# %%
# Display the density of the trajectories, along the 3 mid-planes. For this, make `grid_type="density"`.
create_grid("density", "Sampling Density")


# %%
# Display the sample time of the trajectories, along the 3 mid-planes. For this, make `grid_type="time"`.
# This helps in obtaining relative sampling times of the k-space sampling pattern, which helps debug off-resonance issues
create_grid("time", "Sampling Time")

# %%
# Display the inversion time of the trajectories, along the 3 mid-planes. For this, make `grid_type="inversion"`.
# This helps in obtaining the inversion time when particular region of k-space is sampled, assuming the trajectories are time ordered.
# This helps understand any issues for imaging involving inversion recovery.
# The argument `turbo_factor` can be used to tell what is the number of echoes between 2 inversion pulses.
create_grid("inversion", "Inversion Time", turbo_factor=64)

# %%
# Display the k-space holes in the trajectories, along the 3 mid-planes. For this, make `grid_type="holes"`.
# This helps in understanding the k-space holes, and can help debug artifacts in reconstructions.
create_grid("holes", "K-space Holes", threshold=1e-2)

# %%
# Display the gradient strength of the trajectories, along the 3 mid-planes. For this, make `grid_type="gradients"`.
# This helps in understanding the gradient strength applied at specific k-space region.
# This can also be used as a surrogate to k-space "velocity", i.e. how fast does trajectory pass through a given region in k-space
create_grid("gradients", "Gradient Strength")

# %%
# Display the slew rates of the trajectories, along the 3 mid-planes. For this, make `grid_type="slew"`.
# This helps in understanding the slew rates applied at specific k-space region.
# This can also be used as a surrogate to k-space "acceleration", i.e. how fast does trajectory change in a given region in k-space
create_grid("slew", "Slew Rates")
