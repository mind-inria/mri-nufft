# %%
"""
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

# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install mri-nufft[gpunufft] scikit-image

import time
import joblib

import brainweb_dl as bwdl
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageSequence

from mrinufft import get_operator
from mrinufft.trajectories import initialize_2D_radial

# %%
# Setup a simple class to learn trajectory
# ----------------------------------------
# .. note::
#     While we are only learning the NUFFT operator, we still need the gradient ``wrt_data=True`` to be setup in ``get_operator`` to have all the gradients computed correctly.
#     See [Projector]_ for more details.


class Model(torch.nn.Module):
    def __init__(self, inital_trajectory):
        super(Model, self).__init__()
        self.trajectory = torch.nn.Parameter(
            data=torch.Tensor(inital_trajectory),
            requires_grad=True,
        )
        self.operator = get_operator("gpunufft", wrt_data=True, wrt_traj=True)(
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
# Util function to plot the state of the model
# --------------------------------------------


def plot_state(axs, mri_2D, traj, recon, loss=None, save_name=None):
    axs = axs.flatten()
    axs[0].imshow(np.abs(mri_2D[0]), cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("MR Image")
    axs[1].scatter(*traj.T, s=1)
    axs[1].set_title("Trajectory")
    axs[2].imshow(np.abs(recon[0][0].detach().cpu().numpy()), cmap="gray")
    axs[2].axis("off")
    axs[2].set_title("Reconstruction")
    if loss is not None:
        axs[3].plot(loss)
        axs[3].set_title("Loss")
        axs[3].grid("on")
    if save_name is not None:
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# %%
# Setup model and optimizer
# -------------------------
init_traj = initialize_2D_radial(16, 512).reshape(-1, 2).astype(np.float32)
model = Model(init_traj)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
schedulder = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1, end_factor=0.1, total_iters=100
)

# %%
# Setup data
# ----------

mri_2D = torch.Tensor(np.flipud(bwdl.get_mri(4, "T1")[80, ...]).astype(np.complex64))[
    None
]
mri_2D = mri_2D / torch.linalg.norm(mri_2D)
model.eval()
recon = model(mri_2D)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plot_state(axs, mri_2D, init_traj, recon)

# %%
# Start training loop
# -------------------
losses = []
image_files = []
model.train()
with tqdm(range(100), unit="steps") as tqdms:
    for i in tqdms:
        out = model(mri_2D)
        loss = torch.norm(out - mri_2D[None])
        numpy_loss = loss.detach().cpu().numpy()
        tqdms.set_postfix({"loss": numpy_loss})
        losses.append(numpy_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            # Clamp the value of trajectory between [-0.5, 0.5]
            for param in model.parameters():
                param.clamp_(-0.5, 0.5)
        schedulder.step()
        # Generate images for gif
        hashed = joblib.hash((i, "learn_traj", time.time()))
        filename = "/tmp/" + f"{hashed}.png"
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        plot_state(
            axs,
            mri_2D,
            model.trajectory.detach().cpu().numpy(),
            out,
            losses,
            save_name=filename,
        )
        image_files.append(filename)


# Make a GIF of all images.
imgs = [Image.open(img) for img in image_files]
imgs[0].save(
    "mrinufft_learn_traj.gif",
    save_all=True,
    append_images=imgs[1:],
    optimize=False,
    duration=2,
    loop=0,
)
# sphinx_gallery_start_ignore
# cleanup
import os
import shutil
from pathlib import Path

for f in image_files:
    try:
        os.remove(f)
    except OSError:
        continue
# don't raise errors from pytest. This will only be executed for the sphinx gallery stuff
try:
    final_dir = (
        Path(os.getcwd()).parent.parent
        / "docs"
        / "generated"
        / "autoexamples"
        / "GPU"
        / "images"
    )
    shutil.copyfile("mrinufft_learn_traj.gif", final_dir / "mrinufft_learn_traj.gif")
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# sphinx_gallery_thumbnail_path = 'generated/autoexamples/GPU/images/mrinufft_learn_traj.gif'

# %%
# .. image-sg:: /generated/autoexamples/GPU/images/mrinufft_learn_traj.gif
#    :alt: example learn_samples
#    :srcset: /generated/autoexamples/GPU/images/mrinufft_learn_traj.gif
#    :class: sphx-glr-single-img

# %%
# Trained trajectory
# ------------------
model.eval()
recon = model(mri_2D)
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
plot_state(axs, mri_2D, model.trajectory.detach().cpu().numpy(), recon, losses)
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
