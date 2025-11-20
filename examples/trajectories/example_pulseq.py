# %%
"""
==================================
Create a GRE Sequence using Pulseq
==================================

Example how to create sequences using PyPulseq.
"""
import numpy as np
from mrinufft.trajectories.display import display_3D_trajectory
from mrinufft.trajectories import initialize_2D_spiral, stack

from mrinufft.io.pulseq import pulseq_gre, read_pulseq_traj
import matplotlib.pyplot as plt

# %%
# Defining the sequence parameters like repetition time (TR), echo time (TE), flip angle (FA), field-of-view (FOV) and image matrix size.

TR = 100  # ms
TE = 50  # ms
FA = 10  # degrees
FOV = np.array([0.192, 0.192, 0.128])  # Field of View in meters
img_size = np.array([64, 64, 48])  # Image size in pixels


# %%
# Create a stack of spiral for our trajectory

traj = stack(
    initialize_2D_spiral(Nc=1, Ns=4096, nb_revolutions=12, in_out=True)[:, ::-1, :],
    nb_stacks=img_size[-1],
)

display_3D_trajectory(traj)

# %%

# Create a 3D GRE sequence with the trajectory:

seq = pulseq_gre(traj[:3], acq=None, TR=TR, TE=TE, FA=FA)


# %%

# Let's show the sequence.
plt.rcParams["figure.figsize"] = (20, 10)
seq.plot(show_blocks=True, grad_disp="mT/m")

# %%

# %%
read_kspace = read_pulseq_traj(seq)

# %%
KMAX = 0.5

# %%
kspace_adc, _, t_exc, t_refocus, t_adc = seq.calculate_kspace()

# split t_adc with t_exc and t_refocus, the index are then used to split kspace_adc
FOV = seq.get_definition("FOV")
t_splits = np.sort(np.concatenate([t_exc, t_refocus]))
idx_prev = 0
kspace_shots = []
for t in t_splits:
    idx_next = np.searchsorted(t_adc, t, side="left")
    if idx_next == idx_prev:
        continue
    kspace_shots.append(kspace_adc[:, idx_prev:idx_next].T)
    if idx_next == kspace_adc.shape[1] and t > t_adc[-1]:  # last useful point
        break
    idx_prev = idx_next
if idx_next < kspace_adc.shape[1]:
    kspace_shots.append(kspace_adc[:, idx_next:].T)  # add remaining gradients.
# convert to KMAX standard.
kspace_shots = np.ascontiguousarray(kspace_shots) * KMAX * 2 * np.asarray(FOV)


# %%
# and we can display the k-space trajectory, loaded from the sequence file.
display_3D_trajectory(kspace_shots)
