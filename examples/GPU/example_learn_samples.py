# %%
r"""
======================
Learn Sampling pattern
======================

A small pytorch example to showcase learning k-space sampling patterns.
This example showcases the auto-diff capabilities of the NUFFT operator
wrt to k-space trajectory in mri-nufft.

In this example, we solve the following optimization problem:

.. math::

    \mathbf{\hat{K}} =  \mathrm{arg} \min_{\mathbf{K}} ||  \mathcal{F}_\mathbf{K}^* D_\mathbf{K} \mathcal{F}_\mathbf{K} \mathbf{x} - \mathbf{x} ||_2^2

where :math:`\mathcal{F}_\mathbf{K}` is the forward NUFFT operator and :math:`D_\mathbf{K}` is the density compensators for trajectory :math:`\mathbf{K}`,  :math:`\mathbf{x}` is the MR image which is also the target image to be reconstructed.

.. warning::
    This example only showcases the autodiff capabilities, the learned sampling pattern is not scanner compliant as the scanner gradients required to implement it violate the hardware constraints. In practice, a projection :math:`\Pi_\mathcal{Q}(\mathbf{K})` into the scanner constraints set :math:`\mathcal{Q}` is recommended (see [Proj]_). This is implemented in the proprietary SPARKLING package [Sparks]_. Users are encouraged to contact the authors if they want to use it.
"""


import os

import brainweb_dl as bwdl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch

from mrinufft import get_operator
from mrinufft.trajectories import initialize_2D_radial

# %%
# Setup a simple class to learn trajectory
# ----------------------------------------
# .. note::
#     While we are only learning the NUFFT operator, we still need the gradient ``wrt_data=True`` to be setup in ``get_operator`` to have all the gradients computed correctly.
#     See [Projector]_ for more details.

BACKEND = os.environ.get("MRINUFFT_BACKEND", "gpunufft")

plt.rcParams["animation.embed_limit"] = 2**30  # 1GiB is very large.


class Model(torch.nn.Module):
    def __init__(self, inital_trajectory):
        super(Model, self).__init__()
        self.trajectory = torch.nn.Parameter(
            data=torch.Tensor(inital_trajectory),
            requires_grad=True,
        )
        self.operator = get_operator(BACKEND, wrt_data=True, wrt_traj=True)(
            self.trajectory.detach().cpu().numpy(),
            shape=(256, 256),
            density=True,
            squeeze_dims=False,
        )

    def forward(self, x):
        # Update the trajectory in the NUFFT operator.
        # Note that the re-computation of density compensation happens internally.
        self.operator.samples = self.trajectory.clone()

        # A simple acquisition model simulated with a forward NUFFT operator
        kspace = self.operator.op(x)

        # A simple density compensated adjoint operator
        adjoint = self.operator.adj_op(kspace)
        return adjoint / torch.linalg.norm(adjoint)


# %%
# Setup Data and Model
# --------------------
#
num_epochs = 100

mri_2D = torch.Tensor(np.flipud(bwdl.get_mri(4, "T1")[80, ...]).astype(np.complex64))
mri_2D = mri_2D[None, ...] / torch.linalg.norm(mri_2D)

init_traj = initialize_2D_radial(16, 512).reshape(-1, 2).astype(np.float32)
model = Model(init_traj)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
schedulder = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1, end_factor=0.1, total_iters=num_epochs
)


model.eval()


# %%
# Training and plotting
# ---------------------

recon = model(mri_2D)

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("Training Starting")
axs = axs.flatten()

axs[0].imshow(np.abs(mri_2D[0]), cmap="gray")
axs[0].axis("off")
axs[0].set_title("MR Image")

traj_scat = axs[1].scatter(*init_traj.T, s=0.5)
axs[1].set_title("Trajectory")

recon_im = axs[2].imshow(np.abs(recon.squeeze().detach().cpu().numpy()), cmap="gray")
axs[2].axis("off")
axs[2].set_title("Reconstruction")
(loss_curve,) = axs[3].plot([], [])
axs[3].grid()
axs[3].set_xlabel("epochs")
axs[3].set_ylabel("loss")

fig.tight_layout()


def train():
    """Train loop."""
    losses = []
    for i in range(num_epochs):
        out = model(mri_2D)
        loss = torch.norm(out - mri_2D[None])  # Compute loss

        optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        with torch.no_grad():
            # clamp the value of trajectory between [-0.5, 0.5]
            for param in model.parameters():
                param.clamp_(-0.5, 0.5)
        schedulder.step()
        losses.append(loss.item())
        yield (
            out.detach().cpu().numpy().squeeze(),
            model.trajectory.detach().cpu().numpy(),
            losses,
        )


def plot_epoch(data):
    img, traj, losses = data

    cur_epoch = len(losses)
    recon_im.set_data(abs(img))
    loss_curve.set_xdata(np.arange(cur_epoch))
    loss_curve.set_ydata(losses)
    traj_scat.set_offsets(traj)

    axs[3].set_xlim(0, cur_epoch)
    axs[3].set_ylim(0, 1.1 * max(losses))
    axs[2].set_title(f"Reconstruction, frame {cur_epoch}/{num_epochs}")
    axs[1].set_title(f"Trajectory, frame {cur_epoch}/{num_epochs}")

    if cur_epoch < num_epochs:
        fig.suptitle("Training in progress " + "." * (1 + cur_epoch % 3))
    else:
        fig.suptitle("Training complete !")


ani = animation.FuncAnimation(
    fig, plot_epoch, train, save_count=num_epochs, repeat=False
)
plt.show()


# %%
# References
# ==========
#
# .. [Proj] N. Chauffert, P. Weiss, J. Kahn and P. Ciuciu, "A Projection Algorithm for
#           Gradient Waveforms Design in Magnetic Resonance Imaging," in
#           IEEE Transactions on Medical Imaging, vol. 35, no. 9, pp. 2026-2039, Sept. 2016,
#           doi: 10.1109/TMI.2016.2544251.
# .. [Sparks] Chaithya GR, P. Weiss, G. Daval-Frérot, A. Massire, A. Vignaud and P. Ciuciu,
#           "Optimizing Full 3D SPARKLING Trajectories for High-Resolution Magnetic
#           Resonance Imaging," in IEEE Transactions on Medical Imaging, vol. 41, no. 8,
#           pp. 2105-2117, Aug. 2022, doi: 10.1109/TMI.2022.3157269.
# .. [Projector] Chaithya GR, and Philippe Ciuciu. 2023. "Jointly Learning Non-Cartesian
#           k-Space Trajectories and Reconstruction Networks for 2D and 3D MR Imaging
#           through Projection" Bioengineering 10, no. 2: 158.
#           https://doi.org/10.3390/bioengineering10020158
