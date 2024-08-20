# %%
"""
======================
Learn Sampling pattern
======================

A small pytorch example to showcase learning k-space sampling patterns.
This example showcases the auto-diff capabilities of the NUFFT operator 
wrt to k-space trajectory in mri-nufft.

.. warning::
    This example only showcases the autodiff capabilities, the learned sampling pattern is not scanner compliant as the scanner gradients required to implement it violate the hardware constraints. In practice, a projection into the scanner constraints set is recommended. This is implemented in the proprietary SPARKLING package. Users are encouraged to contact the authors if they want to use it.
"""
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


class Model(torch.nn.Module):
    def __init__(self, inital_trajectory, n_coils, img_size=(256, 256)):
        super(Model, self).__init__()
        self.trajectory = torch.nn.Parameter(
            data=torch.Tensor(inital_trajectory),
            requires_grad=True,
        )
        sample_points = inital_trajectory.reshape(-1, inital_trajectory.shape[-1])
        self.operator = get_operator("gpunufft", wrt_data=True, wrt_traj=True)(
            sample_points,
            shape=img_size,
            n_coils=n_coils,
            squeeze_dims=False,
        )
        self.sense_op = get_operator("gpunufft", wrt_data=True, wrt_traj=True)(
            sample_points,
            shape=img_size,
            density=True,
            n_coils=n_coils,
            smaps=np.ones((n_coils, *img_size)), # Dummy smaps, this is updated in forward pass
            squeeze_dims=False,
        )
        self.img_size = img_size
         
    def forward(self, x):
        self.operator.samples = self.trajectory.clone()
        self.sense_op.samples = self.trajectory.clone()
        
        # Simulate the acquisition process
        kspace = self.operator.op(x)

        # Reconstruction using the sense operator
        self.sense_op.smaps, _ = get_smaps("low_frequency")(
            self.trajectory.detach().numpy(),
            self.img_size,
            kspace.detach(),
            backend="gpunufft",
            density=self.sense_op.density,
            blurr_factor=20,
        )
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
    optimizer, start_factor=1, end_factor=0.1, total_iters=100,
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
    shutil.copyfile("mrinufft_learn_traj_mc.gif", final_dir / "mrinufft_learn_traj_mc.gif")
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
