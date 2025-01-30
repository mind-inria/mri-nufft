# %%
"""
=========================================
Learn Sampling pattern for multi-coil MRI
=========================================

A small pytorch example to showcase learning k-space sampling patterns.
This example showcases the auto-diff capabilities of the NUFFT operator
wrt to k-space trajectory in mri-nufft.

Briefly, in this example we try to learn the k-space samples :math:`\mathbf{K}` for the following cost function:

.. math::

    \mathbf{\hat{K}} =  arg \min_{\mathbf{K}} ||  \sum_{\ell=1}^LS_\ell^* \mathcal{F}_\mathbf{K}^* D_\mathbf{K} \mathcal{F}_\mathbf{K} x_\ell - \mathbf{x}_{sos} ||_2^2

where :math:`S_\ell` is the sensitivity map for the :math:`\ell`-th coil, :math:`\mathcal{F}_\mathbf{K}` is the forward NUFFT operator and :math:`D_\mathbf{K}` is the density compensators for trajectory :math:`\mathbf{K}`,  :math:`\mathbf{x}_\ell` is the image for the :math:`\ell`-th coil, and :math:`\mathbf{x}_{sos} = \sqrt{\sum_{\ell=1}^L x_\ell^2}` is the sum-of-squares image as target image to be reconstructed.

In this example, the forward NUFFT operator :math:`\mathcal{F}_\mathbf{K}` is implemented with `model.operator` while the SENSE operator :math:`model.sense_op` models the term :math:`\mathbf{A} = \sum_{\ell=1}^LS_\ell^* \mathcal{F}_\mathbf{K}^* D_\mathbf{K}`.
For our data, we use a 2D slice of a 3D MRI image from the BrainWeb dataset, and the sensitivity maps are simulated using the `birdcage_maps` function from `sigpy.mri`.

.. note::
    To showcase the features of ``mri-nufft``, we use ``
    "cufinufft"`` backend for ``model.operator`` without density compensation and ``"gpunufft"`` backend for ``model.sense_op`` with density compensation.

.. warning::
    This example only showcases the autodiff capabilities, the learned sampling pattern is not scanner compliant as the scanner gradients required to implement it violate the hardware constraints. In practice, a projection :math:`\Pi_\mathcal{Q}(\mathbf{K})` into the scanner constraints set :math:`\mathcal{Q}` is recommended (see [Proj]_). This is implemented in the proprietary SPARKLING package [Sparks]_. Users are encouraged to contact the authors if they want to use it.
"""
# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install mri-nufft[gpunufft] cufinufft sigpy scikit-image

# %%
# Imports
# -------
import time
import joblib

import brainweb_dl as bwdl
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageSequence

from mrinufft import get_operator
from mrinufft.extras import get_smaps
from mrinufft.trajectories import initialize_2D_radial
from sigpy.mri import birdcage_maps

# %%
# Setup a simple class to learn trajectory
# ----------------------------------------
# .. note::
#     While we are only learning the NUFFT operator, we still need the gradient `wrt_data=True` to have all the gradients computed correctly.
#     See [Projector]_ for more details.


class Model(torch.nn.Module):
    def __init__(self, inital_trajectory, n_coils, img_size=(256, 256)):
        super(Model, self).__init__()
        self.trajectory = torch.nn.Parameter(
            data=torch.Tensor(inital_trajectory),
            requires_grad=True,
        )
        sample_points = inital_trajectory.reshape(-1, inital_trajectory.shape[-1])
        # A simple acquisition model simulated with a forward NUFFT operator. We dont need density compensation here.
        # The trajectory is scaled by 2*pi for cufinufft backend.
        self.operator = get_operator("cufinufft", wrt_data=True, wrt_traj=True)(
            sample_points * 2 * np.pi,
            shape=img_size,
            n_coils=n_coils,
            squeeze_dims=False,
        )
        # A simple density compensated adjoint SENSE operator with sensitivity maps `smaps`.
        self.sense_op = get_operator("gpunufft", wrt_data=True, wrt_traj=True)(
            sample_points,
            shape=img_size,
            density=True,
            n_coils=n_coils,
            smaps=np.ones(
                (n_coils, *img_size)
            ),  # Dummy smaps, this is updated in forward pass
            squeeze_dims=False,
        )
        self.img_size = img_size

    def forward(self, x):
        """Forward pass of the model."""
        # Update the trajectory in the NUFFT operator.
        # The trajectory is scaled by 2*pi for cufinufft backend.
        # Note that the re-computation of density compensation happens internally.
        self.operator.samples = self.trajectory.clone() * 2 * np.pi
        self.sense_op.samples = self.trajectory.clone()

        # Simulate the acquisition process
        kspace = self.operator.op(x)

        # Recompute the sensitivity maps for the updated trajectory.
        self.sense_op.smaps, _ = get_smaps("low_frequency")(
            self.trajectory.detach().numpy(),
            self.img_size,
            kspace.detach(),
            backend="gpunufft",
            density=self.sense_op.density,
            blurr_factor=20,
        )
        # Reconstruction using the sense operator
        adjoint = self.sense_op.adj_op(kspace).abs()
        return adjoint / torch.mean(adjoint)


# %%
# Util function to plot the state of the model
# --------------------------------------------
def plot_state(axs, mri_2D, traj, recon, loss=None, save_name=None):
    axs = axs.flatten()
    axs[0].imshow(np.abs(mri_2D), cmap="gray")
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
n_coils = 6
init_traj = initialize_2D_radial(32, 256).astype(np.float32).reshape(-1, 2)
model = Model(init_traj, n_coils=n_coils, img_size=(256, 256))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
schedulder = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1,
    end_factor=0.1,
    total_iters=100,
)
# %%
# Setup data
# ----------
mri_2D = torch.from_numpy(np.flipud(bwdl.get_mri(4, "T1")[80, ...]).astype(np.float32))
mri_2D = mri_2D / torch.mean(mri_2D)
smaps_simulated = torch.from_numpy(birdcage_maps((n_coils, *mri_2D.shape)))
mcmri_2D = mri_2D[None].to(torch.complex64) * smaps_simulated
model.eval()
recon = model(mcmri_2D)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plot_state(axs, mri_2D, model.trajectory.detach().cpu().numpy(), recon)

# %%
# Start training loop
# -------------------
losses = []
image_files = []
model.train()

with tqdm(range(100), unit="steps") as tqdms:
    for i in tqdms:
        out = model(mcmri_2D)
        loss = torch.nn.functional.mse_loss(out, mri_2D[None, None])
        numpy_loss = loss.detach().cpu().numpy()
        tqdms.set_postfix({"loss": numpy_loss})
        losses.append(numpy_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedulder.step()
        with torch.no_grad():
            # Clamp the value of trajectory between [-0.5, 0.5]
            for param in model.parameters():
                param.clamp_(-0.5, 0.5)
        # Generate images for gif
        hashed = joblib.hash((i, "learn_traj", time.time()))
        filename = "/tmp/" + f"{hashed}.png"
        plt.clf()
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
    "mrinufft_learn_traj_mc.gif",
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
    shutil.copyfile(
        "mrinufft_learn_traj_mc.gif", final_dir / "mrinufft_learn_traj_mc.gif"
    )
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# sphinx_gallery_thumbnail_path = 'generated/autoexamples/GPU/images/mrinufft_learn_traj_mc.gif'

# %%
# .. image-sg:: /generated/autoexamples/GPU/images/mrinufft_learn_traj_mc.gif
#    :alt: example learn_samples
#    :srcset: /generated/autoexamples/GPU/images/mrinufft_learn_traj_mc.gif
#    :class: sphx-glr-single-img

# %%
# Trained trajectory
# ------------------
model.eval()
recon = model(mcmri_2D)
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
# .. [Sparks] G. R. Chaithya, P. Weiss, G. Daval-Frérot, A. Massire, A. Vignaud and P. Ciuciu,
#           "Optimizing Full 3D SPARKLING Trajectories for High-Resolution Magnetic
#           Resonance Imaging," in IEEE Transactions on Medical Imaging, vol. 41, no. 8,
#           pp. 2105-2117, Aug. 2022, doi: 10.1109/TMI.2022.3157269.
# .. [Projector] Chaithya GR, and Philippe Ciuciu. 2023. "Jointly Learning Non-Cartesian
#           k-Space Trajectories and Reconstruction Networks for 2D and 3D MR Imaging
#           through Projection" Bioengineering 10, no. 2: 158.
#           https://doi.org/10.3390/bioengineering10020158
