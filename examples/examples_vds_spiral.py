"""Variable density Spiral, based on the MATLAB implementation of Brian Hargreaves."""

# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: sim
#     language: python
#     name: sim
# ---

# %%
from mrinufft.trajectories.trajectory2D import initialize_2D_vds_spiral
from mrinufft.trajectories.utils import Gammas

# %%
samples = initialize_2D_vds_spiral(
    Nc=1,
    Fcoeff=[100, -24],
    raster_time=0.000004,
    res=1,
    oversamp=4,
    gmax=4,
    smax=15000,
    gamma=4258,
    in_out=True,
)

# %%
samples.shape

# %%
from mrinufft.trajectories.display import display_2D_trajectory

# %%
display_2D_trajectory(samples, linewidth=0.01)

# %%
from mrinufft.trajectories.display import display_gradients

# %%
display_gradients(samples, show_constraints=True)

# %%
max(samples.flatten())

# %%
len(samples[0])
import matplotlib.pyplot as plt

# %%
plt.plot(samples[0, :, 0], samples[0, :, 1])

# %%
len(samples[0])

# %%
len(samples[0]) * 4e-6 * 3

# %%
import numpy as np

np.sqrt(-1)

# %%
