# %%
r"""
==============================================
Supervised vs Self-supervised UNet for MRI.
==============================================

This example compares two UNet training strategies for non-Cartesian
MRI reconstruction:

**Supervised** (left): the standard approach from [fastmri]_, which
requires a ground truth image for training.

.. math::

    \mathcal{L}_{\text{sup}} = ||\mathcal{U}_\theta(\mathbf{y}) - \mathbf{x}||_1

**Self-supervised SSDU** (right): the self-supervised approach from
[ssdu]_ and [ddss]_, which requires only the acquired k-space.

.. math::

    \mathcal{L}_{\text{SSDU}} = \frac{||\mathbf{y}_\Lambda -
    \mathcal{A}_\Lambda(\mathcal{U}_\theta(\mathbf{y}_\Theta))||_1}
    {||\mathbf{y}_\Lambda||_1}

The k-space :math:`\mathbf{y}` is split within each cone so that both
:math:`\Theta` and :math:`\Lambda` span all cones [ddss]_.

Both models use the same trajectory, image, UNet architecture, and
optimizer, only the loss and training input differ.

.. warning::
    We train on a single image here. In practice, this should be done
    on a database like fastMRI [fastmri]_.
"""

# %%
# Imports
import os
import brainweb_dl as bwdl
import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.animation as animation
from fastmri.models import Unet
from mrinufft import get_operator
from mrinufft.trajectories import initialize_2D_cones

# %%
# Setup a simple class for the U-Net model
BACKEND = os.environ.get("MRINUFFT_BACKEND", "cufinufft")
N_CONES = 32
N_SAMPLES = 256
SPLIT_RATIO = 0.6  # 60% of points per cone → Theta, 40% → Lambda
num_epochs = 200

plt.rcParams["animation.embed_limit"] = 2**30  # 1GiB is very large.


# Model is identical for both supervised and self-supervised.
# The only difference is which k-space is passed to forward().
class Model(torch.nn.Module):
    """Model for MRI reconstruction using a U-Net."""

    def __init__(self, initial_trajectory):
        super().__init__()
        self.operator = get_operator(BACKEND, wrt_data=True)(
            initial_trajectory,
            shape=(256, 256),
            density=True,
            squeeze_dims=False,
        )
        self.unet = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4)

    def forward(self, kspace):
        """Forward pass of the model."""
        image = self.operator.adj_op(kspace)
        recon = self.unet(image.float()).abs()
        recon /= torch.mean(recon)
        return recon


# %%
# Utility function to plot the state of the model
def plot_state(axs, mri_2D, traj, recon, loss=None, save_name=None):
    """Image plotting function.

    Plot the original MRI image, the trajectory, the reconstructed image,
    and the loss curve (if provided). Saves the plot if a filename is provided.

    Parameters
    ----------
    axs (numpy array): Array of matplotlib axes to plot on.
    mri_2D (torch.Tensor): Original MRI image.
    traj : Trajectory.
    recon (torch.Tensor): Reconstructed image after training.
    loss (list, optional): List of loss values to plot. Defaults to None.
    save_name (str, optional): Filename to save the plot. Defaults to None.
    """
    axs = axs.flatten()
    axs[0].imshow(np.abs(mri_2D[0]), cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("MR Image")
    axs[1].scatter(*traj.T, s=0.5)
    axs[1].set_title("Trajectory")
    axs[2].imshow(np.abs(recon[0][0].detach().cpu().numpy()), cmap="gray")
    axs[2].axis("off")
    axs[2].set_title("Reconstruction")
    if loss is not None:
        axs[3].plot(loss)
        axs[3].grid("on")
        axs[3].set_title("Loss")
    if save_name is not None:
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# Load BrainWeb image, same for both models.
# Ground truth is used for the supervised loss and visualization only.
mri_2D = torch.Tensor(np.flipud(bwdl.get_mri(4, "T1")[80, ...]).astype(np.complex64))[
    None
]
mri_2D = mri_2D / torch.mean(mri_2D)

# %%
# Build trajectory and simulate full k-space Omega.
init_traj = initialize_2D_cones(N_CONES, N_SAMPLES).reshape(-1, 2).astype(np.float32)
model = Model(init_traj)
model.eval()

op_omega = get_operator(BACKEND, wrt_data=True)(
    init_traj,
    shape=(256, 256),
    density=True,
    squeeze_dims=False,
)
kspace_omega = op_omega.op(mri_2D)  # shape: (1, 1, N_CONES * N_SAMPLES)

# Before training, here is the simple reconstruction we have using a
# density compensated adjoint.
dc_adjoint = model.operator.adj_op(kspace_omega)


# %%
# Split Omega into Theta and Lambda only used by the self-supervised model.
#
# Split at the point level within each cone so that both subsets
# span all cones with complementary subsets of points per cone.
rng = np.random.default_rng(seed=42)
n_theta_per_cone = int(SPLIT_RATIO * N_SAMPLES)

idx_theta, idx_lambda = [], []
for s in range(N_CONES):
    # Get the indices of the points in the current cone and randomly permute them.
    cone_pts = np.arange(s * N_SAMPLES, (s + 1) * N_SAMPLES)
    perm = rng.permutation(N_SAMPLES)
    idx_theta.append(cone_pts[perm[:n_theta_per_cone]])
    idx_lambda.append(cone_pts[perm[n_theta_per_cone:]])

# Concatenate the indices from all cones to get the final Theta and Lambda indices.
idx_theta = np.concatenate(idx_theta)
idx_lambda = np.concatenate(idx_lambda)

# The self-supervised model will only see the k-space points in Theta during training, and the loss will be computed on the points in Lambda.
traj_theta = init_traj[idx_theta]
traj_lambda = init_traj[idx_lambda]

# For visualization, we can also get the k-space data corresponding to Theta and Lambda.
kspace_omega_cpu = kspace_omega.cpu()
kspace_theta = kspace_omega_cpu[:, :, idx_theta]  # (1, 1, n_theta)
kspace_lambda = kspace_omega_cpu[:, :, idx_lambda]  # (1, 1, n_lambda)


# %%
# NUFFT operator for Lambda — used only in the SSDU loss.
op_lambda = get_operator(BACKEND, wrt_data=True)(
    traj_lambda,
    shape=(256, 256),
    density=True,
    squeeze_dims=False,
)


# %%
# Setup both models, same architecture, different input trajectories.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Supervised model uses full Omega trajectory
model_sup = Model(init_traj).to(device)
optimizer_sup = torch.optim.RAdam(model_sup.parameters(), lr=1e-3)

# Self-supervised model uses Theta trajectory only
model_ss = Model(traj_theta).to(device)
optimizer_ss = torch.optim.RAdam(model_ss.parameters(), lr=1e-3)

kspace_omega_gpu = kspace_omega.to(device)
kspace_theta_gpu = kspace_theta.to(device)
kspace_lambda_gpu = kspace_lambda.to(device)


# %%
# Visualization setup — 2 columns (supervised left, self-supervised right).
fig, axs = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle("Training Starting")

# Supervised column (left)
axs[0, 0].imshow(np.abs(mri_2D[0]), cmap="gray")
axs[0, 0].axis("off")
axs[0, 0].set_title("MR Image")

axs[0, 1].scatter(*init_traj.T, s=0.5)
axs[0, 1].set_title("Trajectory (Omega)")

dc_sup = model_sup.operator.adj_op(kspace_omega_gpu)
recon_sup = axs[1, 0].imshow(np.abs(dc_sup[0][0].detach().cpu().numpy()), cmap="gray")
axs[1, 0].axis("off")
axs[1, 0].set_title("Supervised reconstruction")

(loss_curve_sup,) = axs[1, 1].plot([], [], color="steelblue", label="supervised")
axs[1, 1].grid()
axs[1, 1].set_xlabel("epochs")
axs[1, 1].set_ylabel("loss")
axs[1, 1].set_title("Supervised loss (L1 vs ground truth)")

# Self-supervised column (right)
axs[0, 2].imshow(np.abs(mri_2D[0]), cmap="gray")
axs[0, 2].axis("off")
axs[0, 2].set_title("MR Image")

axs[0, 3].scatter(*traj_lambda.T, s=0.1, alpha=0.3, c="gray", label="Lambda 40%")
axs[0, 3].scatter(*traj_theta.T, s=0.1, alpha=0.5, c="steelblue", label="Theta 60%")
axs[0, 3].legend(loc="upper right", fontsize=8)
axs[0, 3].set_title("Trajectory split Theta / Lambda")

dc_ss = model_ss.operator.adj_op(kspace_theta_gpu)
recon_ss = axs[1, 2].imshow(np.abs(dc_ss[0][0].detach().cpu().numpy()), cmap="gray")
axs[1, 2].axis("off")
axs[1, 2].set_title("Self-supervised reconstruction")

(loss_curve_ss,) = axs[1, 3].plot([], [], color="coral", label="self-supervised")
axs[1, 3].grid()
axs[1, 3].set_xlabel("epochs")
axs[1, 3].set_ylabel("loss")
axs[1, 3].set_title("SSDU loss (no ground truth)")

fig.tight_layout()


# %%
def train():
    """Train both models simultaneously and yield results for animation."""
    losses_sup, losses_ss = [], []
    for i in range(num_epochs):

        # --- Supervised model ---
        out_sup = model_sup(kspace_omega_gpu)
        loss_sup = torch.nn.functional.l1_loss(out_sup, mri_2D[None].to(device))
        optimizer_sup.zero_grad()
        loss_sup.backward()
        optimizer_sup.step()
        losses_sup.append(loss_sup.item())

        # --- Self-supervised model (SSDU) ---
        # out_ss is an image reconstruction
        out_ss = model_ss(kspace_theta_gpu)
        # The SSDU loss compares the predicted k-space on Lambda with the acquired k-space on Lambda.
        recon_complex = out_ss.squeeze(1).to(torch.complex64)
        kspace_pred = op_lambda.op(recon_complex)
        eps = 1e-8
        loss_ss = torch.norm(kspace_lambda_gpu - kspace_pred, p=1) / (
            torch.norm(kspace_lambda_gpu, p=1) + eps
        )
        optimizer_ss.zero_grad()
        loss_ss.backward()
        optimizer_ss.step()
        losses_ss.append(loss_ss.item())

        yield (
            out_sup.detach().cpu().numpy().squeeze(),
            out_ss.detach().cpu().numpy().squeeze(),
            losses_sup,
            losses_ss,
        )


def plot_epoch(data):
    """Update both columns at each epoch."""
    img_sup, img_ss, losses_sup, losses_ss = data
    cur_epoch = len(losses_sup)

    # Update supervised column
    recon_sup.set_data(abs(img_sup))
    loss_curve_sup.set_xdata(np.arange(cur_epoch))
    loss_curve_sup.set_ydata(losses_sup)
    axs[1, 1].set_xlim(0, cur_epoch)
    axs[1, 1].set_ylim(0, 1.1 * max(losses_sup))
    axs[1, 0].set_title(f"Supervised, frame {cur_epoch}/{num_epochs}")

    # Update self-supervised column
    recon_ss.set_data(abs(img_ss))
    loss_curve_ss.set_xdata(np.arange(cur_epoch))
    loss_curve_ss.set_ydata(losses_ss)
    axs[1, 3].set_xlim(0, cur_epoch)
    axs[1, 3].set_ylim(0, 1.1 * max(losses_ss))
    axs[1, 2].set_title(f"Self-supervised, frame {cur_epoch}/{num_epochs}")

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
# .. [fastmri] O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
#           for biomedical image segmentation. In International Conference on Medical
#           image computing and computer-assisted intervention, pages 234–241.
#           Springer, 2015.
#           https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models/unet.py
#
# .. [ssdu] B. Yaman, S. A. H. Hosseini, S. Moeller, J. Ellermann, K. Ugurbil,
#           and M. Akcakaya. Self-supervised learning of physics-guided
#           reconstruction neural networks without fully sampled reference data.
#           Magnetic Resonance in Medicine, 84(6):3172-3191, 2020.
#           https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.28378
#
# .. [ddss] B. Zhou, J. Schlemper, N. Dey, S. S. M. Salehi, K. Sheth, C. Liu,
#           J. S. Duncan, and M. Sofka. Dual-domain self-supervised learning for
#           accelerated non-Cartesian MRI reconstruction.
#           Medical Image Analysis, 81:102538, 2022.
#           https://doi.org/10.1016/j.media.2022.102538
