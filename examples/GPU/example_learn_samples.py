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
        self.operator.samples = self.trajectory.clone()
        kspace = self.operator.op(x)
        adjoint = self.operator.adj_op(kspace)
        return adjoint / torch.linalg.norm(adjoint)


# %%
# Util function to plot the state of the model
# --------------------------------------------


def plot_state(axs, mri_2D, traj, recon, loss=None, save_dir="/tmp/", save_name=None):
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
    if save_name is not None:
        plt.savefig(save_dir + save_name, bbox_inches="tight")
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
imgs = []
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
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        plot_state(
            axs,
            mri_2D,
            model.trajectory.detach().cpu().numpy(),
            out,
            losses,
            save_name=f"{i}.png",
        )
        imgs.append(Image.open(f"/tmp/{i}.png"))

# Make a GIF of all images.
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

for f in range(100):
    f = f"/tmp/{f}.png"
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
