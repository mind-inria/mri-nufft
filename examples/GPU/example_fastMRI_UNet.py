# %%
r"""
==================
Simple UNet model.
==================

This model is a simplified version of the U-Net architecture,
which is widely used for image segmentation tasks.
This is implemented in the proprietary FASTMRI package [fastmri]_.

The U-Net model consists of an encoder (downsampling path) and
a decoder (upsampling path) with skip connections between corresponding
layers in the encoder and decoder.
These skip connections help in retaining spatial information
that is lost during the downsampling process.

The primary purpose of this model is to perform image reconstruction tasks,
specifically for MRI images.
It takes an input MRI image and reconstructs it to improve the image quality
or to recover missing parts of the image.

This implementation of the UNet model was pulled from the FastMRI Facebook
repository, which is a collaborative research project aimed at advancing
the field of medical imaging using machine learning techniques.

.. math::

    \mathbf{\hat{x}} = \mathrm{arg} \min_{\mathbf{x}} || \mathcal{U}_\mathbf{\theta}(\mathbf{y}) - \mathbf{x} ||_2^2

where :math:`\mathbf{\hat{x}}` is the reconstructed MRI image, :math:`\mathbf{x}` is the ground truth image,
:math:`\mathbf{y}` is the input MRI image (e.g., k-space data), and :math:`\mathcal{U}_\mathbf{\theta}` is the U-Net model parameterized by :math:`\theta`.

.. warning::
    We train on a single image here. In practice, this should be done on a database like fastMRI [fastmri]_.
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

plt.rcParams["animation.embed_limit"] = 2**30  # 1GiB is very large.


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


# %%
# Setup Inputs (models, trajectory and image)
init_traj = initialize_2D_cones(32, 256).reshape(-1, 2).astype(np.float32)
model = Model(init_traj)
model.eval()

# %%
# Get the image on which we will train our U-Net Model
mri_2D = torch.Tensor(np.flipud(bwdl.get_mri(4, "T1")[80, ...]).astype(np.complex64))[
    None
]
mri_2D = mri_2D / torch.mean(mri_2D)
kspace_mri_2D = model.operator.op(mri_2D)

# Before training, here is the simple reconstruction we have using a
# density compensated adjoint.
dc_adjoint = model.operator.adj_op(kspace_mri_2D)


# %%
num_epochs = 100
optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3)
model.train()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("Training Starting")
axs = axs.flatten()

axs[0].imshow(np.abs(mri_2D[0]), cmap="gray")
axs[0].axis("off")
axs[0].set_title("MR Image")

axs[1].scatter(*init_traj.T, s=0.5)
axs[1].set_title("Trajectory")

recon_im = axs[2].imshow(
    np.abs(dc_adjoint[0][0].detach().cpu().numpy()),
    cmap="gray",
)
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
        out = model(kspace_mri_2D)  # Forward pass

        loss = torch.nn.functional.l1_loss(out, mri_2D[None])  # Compute loss

        optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        losses.append(loss.item())
        yield out.detach().cpu().numpy().squeeze(), losses


def plot_epoch(data):
    img, losses = data
    cur_epoch = len(losses)
    recon_im.set_data(abs(img))
    loss_curve.set_xdata(np.arange(cur_epoch))
    loss_curve.set_ydata(losses)
    axs[3].set_xlim(0, cur_epoch)
    axs[3].set_ylim(0, 1.1 * max(losses))
    axs[2].set_title(f"Reconstruction, frame {cur_epoch}/{num_epochs}")

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
