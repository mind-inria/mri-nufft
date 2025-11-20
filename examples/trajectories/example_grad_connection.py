# %%
"""
===========================================
Prephasors, Spoilers and arbitrary Waveforms
============================================


This example showcases how to create gradient waveform under a set of constraints. 


When designing MRI sequences It's often necessary to have some way of going from
a point A to point B in the kspace. For instance:

- At the beginning of an
acquisition we may want to go from the center of the k-space to the starting
point of the trajectory. This is usually call the "prephasor" or "prewinder"
gradient.
- At the end of an acquisition we may want to go from the end point of
the trajectory back to the center, or to crush any residual magnetization by
going to the edge of the k-space. This is usually called a "rewind" or "spoiler"
gradient.

However, these gradient waveforms needs to be designed under the hardware system constraints:
The maximum gradient strength :math:`g_\max`, and slew rate :math:`s_\max`, and the raster time :math:`\Delta t`.

Once the constraints are defined, the gradient waveforms can be determined using different methods
such as linear programming or quadratic programming.

This example shows how to create such gradient waveforms using MRI-NUFFT.
"""

# %%
import numpy as np

from mrinufft import initialize_2D_cones
from mrinufft.trajectories.utils import (
    Acquisition,
    convert_gradients_to_trajectory,
    convert_trajectory_to_gradients,
)
from mrinufft.trajectories.gradients import (
    connect_gradient,
    get_prephasors_and_spoilers,
)

# %%
# We are going to rely on the :py:obj:`Acquisition` configuration to define the constraints

acq = Acquisition.default

# %%
# Create a demo radial trajectory

traj = initialize_2D_cones(Nc=32, Ns=512, tilt="uniform", in_out=True)

traj_grad, init_points = convert_trajectory_to_gradients(traj, acq)


# Create prephasor and spoiler gradients
# =======================================

prephasors = {}
spoilers = {}
full_grads = {}
full_traj = {}
for method in ["lp", "lp-minslew", "osqp"]:
    print(f"Creating prephasor and spoiler gradients using method: {method}")
    p, s = get_prephasors_and_spoilers(
        traj, acq=acq, method=method, spoil_loc=(1, 0, 0)
    )
    g = np.concatenate([p, traj_grad, s], axis=1)

    prephasors[method] = p
    spoilers[method] = s
    # Connect the prephasor and spoiler gradients to the trajectory gradients
    full_grads[method] = g
    full_traj[method] = convert_gradients_to_trajectory(
        g, np.zeros_like(init_points), acq=acq
    )


# %%
# Show the results
# ==================

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(21, 7))
gs0 = fig.add_gridspec(2, 1, hspace=0.3)

gsgrad = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[0])
gstraj = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[1])


grad_ax = gsgrad.subplots(sharex=True)
axs = gstraj.subplots(sharex=True, sharey=True)
time = np.arange(full_grads["lp"].shape[1]) * acq.raster_time  # in ms
# Plot gradients

for method in ["lp", "lp-minslew", "osqp"]:
    grad_ax[0].plot(
        np.arange(full_grads[method].shape[1]) * acq.raster_time,
        full_grads[method][0, :, 0],  # show x gradient
        label=f"{method}",
    )
    grad_ax[1].plot(
        np.arange(full_grads[method].shape[1]) * acq.raster_time,
        full_grads[method][0, :, 1],  # show x gradient
        label=f"{method}",
    )
grad_ax[0].set_title("Full gradient waveforms with prephasor and spoiler")
grad_ax[1].set_xlabel("Time (ms)")
grad_ax[0].set_ylabel("Gx (T/m)")
grad_ax[1].set_ylabel("Gy (T/m)")
grad_ax[0].axvline(acq.raster_time * prephasors[method].shape[1], ls="--", c="gray")
grad_ax[0].axvline(
    acq.raster_time * (full_grads[method].shape[1] - spoilers[method].shape[1]),
    ls="--",
    c="gray",
)
grad_ax[1].axvline(acq.raster_time * prephasors[method].shape[1], ls="--", c="gray")
grad_ax[1].axvline(
    acq.raster_time * (full_grads[method].shape[1] - spoilers[method].shape[1]),
    ls="--",
    c="gray",
)

grad_ax[0].legend(loc="upper center")


for i, method in enumerate(["lp", "lp-minslew", "osqp"]):
    t = full_traj[method]
    t_pre = t[:, : prephasors[method].shape[1], :]
    t_post = t[:, -spoilers[method].shape[1] :]
    t_core = t[:, prephasors[method].shape[1] : -spoilers[method].shape[1], :]

    axs[i].scatter(
        t_core.reshape(-1, 2)[:, 0],
        t_core.reshape(-1, 2)[:, 1],
        c="k",
        s=0.5,
    )

    axs[i].scatter(
        t_pre.reshape(-1, 2)[:, 0],
        t_pre.reshape(-1, 2)[:, 1],
        c="tab:blue",
        s=0.5,
    )
    axs[i].scatter(
        t_post.reshape(-1, 2)[:, 0],
        t_post.reshape(-1, 2)[:, 1],
        c="tab:green",
        s=0.5,
    )
    axs[i].set_title(f"'{method}' prephasor/spoiler")
    axs[i].grid()
plt.legend()
plt.show()

# %%
