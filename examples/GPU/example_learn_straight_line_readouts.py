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


# %%
# Setup a simple class to learn trajectory
# ----------------------------------------


class Model(torch.nn.Module):
    def __init__(self, num_shots, img_size, factor_cartesian=0.3):
        super(Model, self).__init__()
        self.num_samples_per_shot = 128
        cart_del = 1 / img_size[0]
        num_cart_points = np.round(np.sqrt(factor_cartesian * num_shots)).astype(int)
        edge_center = cart_del * num_cart_points / 2
        
        self.central_points = torch.nn.Parameter(
            data=torch.stack(
                torch.meshgrid(
                    torch.linspace(-edge_center, edge_center, num_cart_points),
                    torch.linspace(-edge_center, edge_center, num_cart_points),
                    indexing="ij",
                ),
                axis=-1,
            ).reshape(-1, 2),
            requires_grad=False,
        )
        self.non_center_points = torch.nn.Parameter(
            data=torch.Tensor(np.random.random((num_shots-self.central_points.shape[0], 2))-0.5),
            requires_grad=True,
        )
        self.operator = get_operator("gpunufft", wrt_data=True, wrt_traj=True)(
            np.random.random((self.get_2D_points().shape[0]*self.num_samples_per_shot, 3))-0.5,
            shape=img_size,
            density=True,
            squeeze_dims=False,
        )
        
    def get_trajectory(self):
        return self._get_3D_points(self.get_2D_points())
        
    def get_2D_points(self):
        return torch.vstack([self.central_points, self.non_center_points])
    
    def _get_3D_points(self, samples2D):
        line = torch.linspace(-0.5, 0.5, self.num_samples_per_shot, device=samples2D.device, dtype=samples2D.dtype)
        return torch.stack(
            [
                line.repeat(samples2D.shape[0], 1),
                samples2D[:, 0].repeat(self.num_samples_per_shot, 1).T,
                samples2D[:, 1].repeat(self.num_samples_per_shot, 1).T, 
            ], 
            dim=-1,
        ).permute(1, 0, 2).reshape(-1, 3)

    def forward(self, x):
        self.operator.samples = self.get_trajectory()
        kspace = self.operator.op(x)
        adjoint = self.operator.adj_op(kspace)
        return adjoint / torch.linalg.norm(adjoint)


# %%
# Util function to plot the state of the model
# --------------------------------------------


def plot_state(axs, mri_2D, traj, recon, loss=None, save_dir="/tmp/", save_name=None):
    axs = axs.flatten()
    axs[0].imshow(np.abs(mri_2D[0][..., 11]), cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("MR Image")
    axs[1].scatter(*traj.T, s=10)
    axs[1].set_title("Trajectory")
    axs[2].imshow(np.abs(recon[0][0][..., 11].detach().cpu().numpy()), cmap="gray")
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

cart_data = np.flipud(bwdl.get_mri(4, "T1")).T[::8, ::8, ::8].astype(np.complex64)
model = Model(253, cart_data.shape)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# %%
# Setup data
# ----------

mri_3D = torch.Tensor(cart_data)[None]
mri_3D = mri_3D / torch.linalg.norm(mri_3D)
model.eval()
recon = model(mri_3D)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plot_state(axs, mri_3D, model.get_2D_points().detach().cpu().numpy(), recon)
# %%
# Start training loop
# -------------------
losses = []
imgs = []
model.train()
with tqdm(range(100), unit="steps") as tqdms:
    for i in tqdms:
        out = model(mri_3D)
        loss = torch.norm(out - mri_3D[None])
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
        # Generate images for gif
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        plot_state(
            axs,
            mri_3D,
            model.get_2D_points().detach().cpu().numpy(), 
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
recon = model(mri_3D)
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
plot_state(axs, mri_3D, model.trajectory.detach().cpu().numpy(), recon, losses)
plt.show()
