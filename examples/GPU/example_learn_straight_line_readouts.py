# %%
r"""
===================================
Learn Straight line readout pattern
===================================

A small pytorch example to showcase learning k-space sampling patterns.
In this example we learn the 2D sampling pattern for a 3D MRI image, assuming
straight line readouts. This example showcases the auto-diff capabilities of the NUFFT operator
The image resolution is kept small to reduce computation time.

.. warning:: This example only showcases the autodiff capabilities, the learned
    sampling pattern is not scanner compliant as the scanner gradients required
    to implement it violate the hardware constraints. In practice, a projection
    :math:`\Pi_\mathcal{Q}(\mathbf{K})` into the scanner constraints set
    :math:`\mathcal{Q}` is recommended (see [Proj]_). This is implemented in the
    proprietary SPARKLING package [Sparks]_. Users are encouraged to contact the
    authors if they want to use it.

"""

# %%
# Imports
# -------
import os

import brainweb_dl as bwdl
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.animation as animation

from mrinufft import get_operator

BACKEND = os.environ.get("MRINUFFT_BACKEND", "cufinufft")

plt.rcParams["animation.embed_limit"] = 2**30  # 1GiB is very large.

# %%
# Setup a simple class to learn trajectory
# ----------------------------------------
# .. note::
#     While we are only learning the NUFFT operator, we still need the gradient `wrt_data=True` to have all the gradients computed correctly.
#     See [Projector]_ for more details.
#
# .. note::
#     Since we are training a 2D non-Cartesian pattern for a 3D volume, we could use the "stacked" version of
#     the operator, which uses a FFT instead of a NUFFT. However, this is not supported yet, so we use the full 3D implementation !


class Model(torch.nn.Module):
    def __init__(self, num_shots, img_size, factor_cartesian=0.1):
        super().__init__()
        self.num_samples_per_shot = 128
        cart_del = 1 / img_size[0]
        num_cart_points = np.round(np.sqrt(factor_cartesian * num_shots)).astype(int)
        edge_center = cart_del * num_cart_points / 2

        self.central_points = torch.nn.Parameter(
            data=torch.stack(
                torch.meshgrid(
                    torch.linspace(
                        -edge_center, edge_center, num_cart_points, dtype=torch.float32
                    ),
                    torch.linspace(
                        -edge_center, edge_center, num_cart_points, dtype=torch.float32
                    ),
                    indexing="ij",
                ),
                axis=-1,
            ).reshape(-1, 2),
            requires_grad=False,
        )
        self.non_center_points = torch.nn.Parameter(
            data=torch.Tensor(
                np.random.random((num_shots - self.central_points.shape[0], 2)).astype(
                    np.float32
                )
                - 0.5
            ),
            requires_grad=True,
        )
        self.operator = get_operator(BACKEND, wrt_data=True, wrt_traj=True)(
            np.random.random(
                (self.get_2D_points().shape[0] * self.num_samples_per_shot, 3)
            ).astype(np.float32)
            - 0.5,
            shape=img_size,
            density=True,
            squeeze_dims=False,
        )

    def get_trajectory(self, get_as_shot=False):
        samples = self._get_3D_points(self.get_2D_points())
        if not get_as_shot:
            return samples
        return samples.reshape(-1, self.num_samples_per_shot, 3)

    def get_2D_points(self):
        return torch.vstack([self.central_points, self.non_center_points])

    def _get_3D_points(self, samples2D):
        line = torch.linspace(
            -0.5,
            0.5,
            self.num_samples_per_shot,
            device=samples2D.device,
            dtype=samples2D.dtype,
        )
        return torch.stack(
            [
                line.repeat(samples2D.shape[0], 1),
                samples2D[:, 0].repeat(self.num_samples_per_shot, 1).T,
                samples2D[:, 1].repeat(self.num_samples_per_shot, 1).T,
            ],
            dim=-1,
        ).reshape(-1, 3)

    def forward(self, x):
        self.operator.samples = self.get_trajectory()
        kspace = self.operator.op(x)
        adjoint = self.operator.adj_op(kspace).abs()
        return adjoint / torch.mean(adjoint)


# %%
# Setup model and optimizer
# -------------------------

num_epochs = 100

cart_data = np.flipud(bwdl.get_mri(4, "T1")).T[::8, ::8, ::8].astype(np.complex64)
model = Model(253, cart_data.shape)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
schedulder = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1,
    end_factor=0.01,
    total_iters=num_epochs,
)
# %%
# Setup data
# ----------

mri_3D = torch.Tensor(cart_data)[None]
mri_3D = mri_3D / torch.mean(mri_3D)
model.eval()
recon = model(mri_3D)
# %%
#
#
# Start training loop
# -------------------
# Red points in the graph show the original locations, and the blue ones the new updated trajectory.
# As training goes, they will deviate more and more.


fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs = axs.ravel()
axs[0].imshow(np.abs(mri_3D.squeeze())[..., 11], cmap="gray")
axs[0].axis("off")
axs[0].set_title("Ground truth")
axs[1].remove()
axs[1] = fig.add_subplot(222, projection="3d", azim=0, elev=0)
traj_scat = axs[1].scatter(
    *model.get_trajectory(True).detach().cpu().numpy()[:, 0, :].T, s=1, c="tab:blue"
)
traj_scat2 = axs[1].scatter(
    *model.get_trajectory(True).detach().cpu().numpy()[:, 0, :].T, s=1, c="tab:red"
)
axs[1].set_xlim(-0.5, 0.5)
axs[1].set_ylim(-0.5, 0.5)
axs[1].set_zlim(-0.5, 0.5)
# traj_scat, = axs[1].plot(*model.get_trajectory(True).detach().cpu().numpy()[:,0,:].T, linestyle="", marker="o")
axs[1].set_title("Trajectory")

recon_im = axs[2].imshow(
    np.abs(recon.squeeze()[..., 11].detach().cpu().numpy()), cmap="gray"
)
axs[2].set_title("Reconstruction")
axs[2].axis("off")

axs[3].grid()
(loss_curve,) = axs[3].plot([], [])
axs[3].set_ylabel("Loss")
axs[3].set_xlabel("epoch")
fig.suptitle("Starting Training")
fig.tight_layout()


def train():
    """Train loop."""
    losses = []
    old_traj = None
    for i in range(num_epochs):
        out = model(mri_3D)
        loss = torch.norm(out - mri_3D[None])  # Compute loss

        optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        with torch.no_grad():
            # clamp the value of trajectory between [-0.5, 0.5]
            for param in model.parameters():
                param.clamp_(-0.5, 0.5)
        schedulder.step()
        losses.append(loss.item())
        new_traj = model.get_trajectory(True).detach().cpu().numpy()
        yield (
            out.detach().cpu().numpy().squeeze()[..., 11],
            new_traj,
            old_traj,
            losses,
        )
        old_traj = new_traj


def plot_epoch(data):
    img, new_traj, old_traj, losses = data

    cur_epoch = len(losses)
    recon_im.set_data(abs(img))
    loss_curve.set_xdata(np.arange(cur_epoch))
    loss_curve.set_ydata(losses)
    mov3d = 70
    if cur_epoch > mov3d:
        #        traj_scat2.set_offsets([[np.nan, np.nan]])
        trajf = new_traj.reshape(-1, 3)
        traj_scat.set_offsets(trajf[:, :2])
        traj_scat.set_3d_properties(trajf[:, 2], "z", True)
        # traj_scat.set_xdata(traj[:, :, 0].ravel())
        # traj_scat.set_ydata(traj[:, :, 1].ravel())
        # traj_scat.set_3d_properties(traj[:, :, 2].ravel())
        axs[1].view_init(azim=(cur_epoch - mov3d), elev=(cur_epoch - mov3d))
    else:
        traj_scat.set_offsets(new_traj[:, 0, :2])

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
# .. [Sparks] G. R. Chaithya, P. Weiss, G. Daval-Frérot, A. Massire, A. Vignaud and P. Ciuciu,
#           "Optimizing Full 3D SPARKLING Trajectories for High-Resolution Magnetic
#           Resonance Imaging," in IEEE Transactions on Medical Imaging, vol. 41, no. 8,
#           pp. 2105-2117, Aug. 2022, doi: 10.1109/TMI.2022.3157269.
# .. [Projector] Chaithya GR, and Philippe Ciuciu. 2023. "Jointly Learning Non-Cartesian
#           k-Space Trajectories and Reconstruction Networks for 2D and 3D MR Imaging
#           through Projection" Bioengineering 10, no. 2: 158.
#           https://doi.org/10.3390/bioengineering10020158
