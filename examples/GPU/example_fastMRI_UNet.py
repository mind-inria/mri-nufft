# %%
r"""
==============================================
Supervised vs Self-supervised UNet for MRI.
==============================================

This example compares two UNet training strategies for non-Cartesian
MRI reconstruction. It works in 2D only, reconstructing a single brain
slice at a time.

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
:math:`\Theta` and :math:`\Lambda` span all cones [ddss]_. Both models
share the same UNet architecture: the complex zero-filled image is
split into 2 real channels (real, imaginary) before the UNet, and the
2-channel output is recombined into a complex image afterward, so no
phase information is discarded.

.. warning::
    We train on a single image here. In practice, this should be done
    on a database like fastMRI [fastmri]_.
"""

# %%
# Imports
import os

import brainweb_dl as bwdl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastmri.models import Unet

from mrinufft import get_operator
from mrinufft.trajectories import initialize_2D_cones

# %%
# Configuration
BACKEND = os.environ.get("MRINUFFT_BACKEND", "cufinufft")
IMG_SHAPE = (256, 256)
N_CONES = 32
N_SAMPLES = 256
SPLIT_RATIO = 0.6
N_EPOCHS = 300
LR = 1e-2

plt.rcParams["animation.embed_limit"] = 2**30
device = "cuda" if torch.cuda.is_available() else "cpu"


# %%
# Model — shared architecture for both supervised and self-supervised.
class Model(torch.nn.Module):
    """UNet model reconstructing a complex image from k-space.

    Parameters
    ----------
    operator : MRINufftOperator
        NUFFT operator used to compute the adjoint (zero-filled) image.
    """

    def __init__(self, operator):
        super().__init__()
        self.operator = operator
        self.unet = Unet(in_chans=2, out_chans=2, chans=32, num_pool_layers=4)

    def forward(self, kspace):
        """Reconstruct a 2-channel real/imaginary image from k-space."""
        image_zf = self.operator.adj_op(kspace)  # (1, 1, H, W) complex
        image_2ch = torch.view_as_real(image_zf)  # (1, 1, H, W, 2)
        image_2ch = image_2ch.squeeze(1).permute(0, 3, 1, 2)  # (1, 2, H, W)
        return self.unet(image_2ch)


# %%
# Split utility : works for any branch-based trajectory (radial, cones, etc.)
#  ordered as n_cones consecutive blocks of n_samples points.
def split_kspace(traj_omega, kspace_omega, split_ratio, n_cones, n_samples):
    """Split a trajectory into Theta and Lambda subsets for SSDU.

    Splits at the point level within each cone so both subsets span
    all cones. Theta U Lambda = Omega, Theta ∩ Lambda = ∅.

    Parameters
    ----------
    traj_omega : np.ndarray
        Full trajectory, shape (n_cones * n_samples, 2).
    kspace_omega : torch.Tensor
        Full k-space measurements, shape (1, 1, n_cones * n_samples).
    split_ratio : float
        Fraction of points per cone assigned to Theta.
    n_cones : int
        Number of cones in the trajectory.
    n_samples : int
        Number of samples per cone.

    Returns
    -------
    tuple
        (traj_theta, traj_lambda, kspace_theta, kspace_lambda)
    """
    rng = np.random.default_rng(seed=42)
    n_theta_per_cone = int(split_ratio * n_samples)

    idx_theta_list, idx_lambda_list = [], []
    for s in range(n_cones):
        cone_pts = np.arange(s * n_samples, (s + 1) * n_samples)
        perm = rng.permutation(n_samples)
        idx_theta_list.append(cone_pts[perm[:n_theta_per_cone]])
        idx_lambda_list.append(cone_pts[perm[n_theta_per_cone:]])

    idx_theta = np.concatenate(idx_theta_list)
    idx_lambda = np.concatenate(idx_lambda_list)

    traj_theta = traj_omega[idx_theta]
    traj_lambda = traj_omega[idx_lambda]
    kspace_theta = kspace_omega[:, :, idx_theta]
    kspace_lambda = kspace_omega[:, :, idx_lambda]

    return traj_theta, traj_lambda, kspace_theta, kspace_lambda


# %%
# Load BrainWeb image — ground truth used for the supervised loss and
# for visualization only (never seen by the self-supervised model).
mri_2d = torch.tensor(np.flipud(bwdl.get_mri(4, "T1")[80, ...]).astype(np.complex64))[
    None
].to(
    device
)  # (1, 256, 256)
mri_2d = mri_2d / torch.mean(mri_2d)


# %%
# Build the full cones trajectory (Omega) and simulate k-space.
traj_omega = (
    initialize_2D_cones(N_CONES, N_SAMPLES, nb_zigzags=10)
    .reshape(-1, 2)
    .astype(np.float32)
)

op_omega = get_operator(BACKEND, wrt_data=True)(
    traj_omega,
    shape=IMG_SHAPE,
    density=True,
    squeeze_dims=False,
)
kspace_omega = op_omega.op(mri_2d)  # (1, 1, N_CONES * N_SAMPLES)


# %%
# Split Omega into Theta and Lambda, used only by the self-supervised
# model. Theta is the network input, Lambda is used for the SSDU loss.
traj_theta, traj_lambda, kspace_theta, kspace_lambda = split_kspace(
    traj_omega, kspace_omega, SPLIT_RATIO, N_CONES, N_SAMPLES
)

op_theta = get_operator(BACKEND, wrt_data=True)(
    traj_theta,
    shape=IMG_SHAPE,
    density=True,
    squeeze_dims=False,
)
op_lambda = get_operator(BACKEND, wrt_data=True)(
    traj_lambda,
    shape=IMG_SHAPE,
    density=True,
    squeeze_dims=False,
)


# %%
# SSDU loss predicts k-space at Lambda from the reconstruction and
# compares it with the true measurements in the k-space. No ground truth needed.
def ssdu_loss(recon, kspace_lambda, op_lambda):
    """Compute the SSDU self-supervised loss.

    Parameters
    ----------
    recon : torch.Tensor
        Reconstructed real/imaginary image, shape (1, 2, H, W).
    kspace_lambda : torch.Tensor
        True k-space at Lambda locations, shape (1, 1, n_lambda).
    op_lambda : MRINufftOperator
        NUFFT operator for the Lambda trajectory.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    recon_complex = torch.complex(recon[:, 0, :, :], recon[:, 1, :, :])
    kspace_pred = op_lambda.op(recon_complex)

    eps = 1e-8
    return torch.norm(kspace_lambda - kspace_pred, p=1) / (
        torch.norm(kspace_lambda, p=1) + eps
    )


# %%
# Setup both models, same architecture, different input trajectories.
model_sup = Model(op_omega).to(device)
optimizer_sup = torch.optim.RAdam(model_sup.parameters(), lr=LR)

model_ss = Model(op_theta).to(device)
optimizer_ss = torch.optim.RAdam(model_ss.parameters(), lr=LR)


# %%
# Visualization setup — 2 columns: supervised (left), self-supervised
# (right).
fig, axs = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle("Training Starting")

# Supervised column
axs[0, 0].imshow(np.abs(mri_2d[0].cpu().numpy()), cmap="gray")
axs[0, 0].axis("off")
axs[0, 0].set_title("MR Image")

axs[0, 1].scatter(*traj_omega.T, s=0.5)
axs[0, 1].set_title("Trajectory (Omega)")

dc_sup = model_sup.operator.adj_op(kspace_omega)
recon_sup = axs[1, 0].imshow(np.abs(dc_sup[0, 0].detach().cpu().numpy()), cmap="gray")
axs[1, 0].axis("off")
axs[1, 0].set_title("Supervised reconstruction")

(loss_curve_sup,) = axs[1, 1].plot([], [], color="steelblue")
axs[1, 1].grid()
axs[1, 1].set_xlabel("epochs")
axs[1, 1].set_ylabel("loss")
axs[1, 1].set_title("Supervised loss (L1 vs ground truth)")

# Self-supervised column
axs[0, 2].imshow(np.abs(mri_2d[0].cpu().numpy()), cmap="gray")
axs[0, 2].axis("off")
axs[0, 2].set_title("MR Image (not used in training)")

axs[0, 3].scatter(*traj_theta.T, s=0.1, alpha=0.5, c="steelblue", label="Theta 60%")
axs[0, 3].scatter(*traj_lambda.T, s=0.1, alpha=0.3, c="coral", label="Lambda 40%")
axs[0, 3].legend(loc="upper right", fontsize=8)
axs[0, 3].set_title("Trajectory split Theta / Lambda")

dc_ss = model_ss.operator.adj_op(kspace_theta)
recon_ss = axs[1, 2].imshow(np.abs(dc_ss[0, 0].detach().cpu().numpy()), cmap="gray")
axs[1, 2].axis("off")
axs[1, 2].set_title("Self-supervised reconstruction")

(loss_curve_ss,) = axs[1, 3].plot([], [], color="coral")
axs[1, 3].grid()
axs[1, 3].set_xlabel("epochs")
axs[1, 3].set_ylabel("loss")
axs[1, 3].set_title("SSDU loss (no ground truth)")

fig.tight_layout()


# %%
# Training loop, both models trained simultaneously, one step each per
# epoch, so they can be compared under the same animation.
def train():
    """Train both models simultaneously and yield results for animation.

    Yields
    ------
    tuple
        (sup recon, ss recon, sup losses, ss losses), each recon a
        (2, H, W) numpy array of real/imaginary channels.
    """
    losses_sup, losses_ss = [], []
    target = torch.view_as_real(mri_2d[None])
    target = target.squeeze(1).permute(0, 3, 1, 2)  # (1, 2, H, W)

    for _ in range(N_EPOCHS):
        # --- Supervised model ---
        out_sup = model_sup(kspace_omega)
        loss_sup = torch.nn.functional.l1_loss(out_sup, target)
        optimizer_sup.zero_grad()
        loss_sup.backward()
        optimizer_sup.step()
        losses_sup.append(loss_sup.item())

        # --- Self-supervised model (SSDU) ---
        out_ss = model_ss(kspace_theta)
        loss_ss = ssdu_loss(out_ss, kspace_lambda, op_lambda)

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
    """Update both columns of the figure at each epoch."""
    img_sup, img_ss, losses_sup, losses_ss = data
    cur_epoch = len(losses_sup)

    recon_sup.set_data(np.abs(img_sup[0] + 1j * img_sup[1]))
    loss_curve_sup.set_xdata(np.arange(cur_epoch))
    loss_curve_sup.set_ydata(losses_sup)
    axs[1, 1].set_xlim(0, cur_epoch)
    axs[1, 1].set_ylim(0, 1.1 * max(losses_sup))
    axs[1, 0].set_title(f"Supervised, frame {cur_epoch}/{N_EPOCHS}")

    recon_ss.set_data(np.abs(img_ss[0] + 1j * img_ss[1]))
    loss_curve_ss.set_xdata(np.arange(cur_epoch))
    loss_curve_ss.set_ydata(losses_ss)
    axs[1, 3].set_xlim(0, cur_epoch)
    axs[1, 3].set_ylim(0, 1.1 * max(losses_ss))
    axs[1, 2].set_title(f"Self-supervised, frame {cur_epoch}/{N_EPOCHS}")

    if cur_epoch < N_EPOCHS:
        fig.suptitle("Training in progress " + "." * (1 + cur_epoch % 3))
    else:
        fig.suptitle("Training complete !")


ani = animation.FuncAnimation(fig, plot_epoch, train, save_count=N_EPOCHS, repeat=False)
plt.show()


# %%
# References
# ==========
#
# .. [fastmri] M. J. Muckley, B. Riemenschneider, A. Radmanesh, et al.
#          Results of the 2020 fastMRI challenge for machine learning
#         MR image reconstruction. IEEE Transactions on Medical
#         Imaging, 40(9):2306-2317, 2021.
#          https://doi.org/10.1109/TMI.2021.3075856
#
# .. [ssdu] B. Yaman et al. Self-supervised learning of physics-guided
#           reconstruction neural networks without fully sampled
#           reference data. MRM, 84(6):3172-3191, 2020.
#           https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.28378
#
# .. [ddss] B. Zhou et al. Dual-domain self-supervised learning for
#           accelerated non-Cartesian MRI reconstruction.
#           Medical Image Analysis, 81:102538, 2022.
#           https://doi.org/10.1016/j.media.2022.102538
