# %%
"""
==================================
Create a GRE Sequence using Pulseq
==================================

Example how to create sequences using PyPulseq.
"""
import numpy as np
from mrinufft.io.pulseq import gre_3D
from mrinufft.trajectories.display import display_3D_trajectory
from mrinufft.trajectories import initialize_2D_spiral, stack

import matplotlib.pyplot as plt

# %%
# Some Constant values

TR = 100 # ms
TE = 50 # ms
FA = 10 # degrees
FOV = np.array([0.192, 0.192, 0.128])  # Field of View in meters
img_size = np.array([64, 64, 48])  # Image size in pixels


# %%
# Create a stack of spiral for our trajectory

traj = stack(initialize_2D_spiral(Nc=1, Ns=2000, nb_revolutions=6, in_out=True),nb_stacks=img_size[-1])

display_3D_trajectory(traj[:2])

# %%

# Creat a 3D GRE sequence with the trajectory:

seq = gre_3D(traj, fov=FOV, img_size=img_size, TR=TR, TE=TE, FA=FA)

# Let's show the sequence.
plt.rcParams["figure.figsize"]=(20,10)
seq.plot(show_blocks=True, grad_disp="mT/m")

# %%
from mrinufft.io.utils import prepare_trajectory_for_seq
full_grads, skip_start, skip_end = prepare_trajectory_for_seq(traj, fov=FOV, img_size=img_size)

# %%
plt.plot(full_grads[0]*1e3)

# %%
full_grads.shape[1]-skip_start -skip_end

# %%
