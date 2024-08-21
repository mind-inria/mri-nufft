# %%
"""
===============================================
Learn Sampling pattern with multi-resolution
===============================================

A small pytorch example to showcase learning k-space sampling patterns.
This example showcases the auto-diff capabilities of the NUFFT operator 
wrt to k-space trajectory in mri-nufft.
This example can run on a binder instance as it is purely CPU based backend (finufft).

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
from mrinufft.trajectories import initialize_2D_radial

# %%
# Setup a simple class to learn trajectory
# ----------------------------------------


class Model(torch.nn.Module):
    def __init__(
        self,
        inital_trajectory,
        img_size=(256, 256),
        start_decim=8,
        interpolation_mode="linear",
    ):
        super(Model, self).__init__()
        self.control = torch.nn.Parameter(
            data=torch.Tensor(inital_trajectory[:, ::start_decim]),
            requires_grad=True,
        )
        self.current_decim = start_decim
        self.interpolation_mode = interpolation_mode
        sample_points = inital_trajectory.reshape(-1, inital_trajectory.shape[-1])
        self.operator = get_operator("gpunufft", wrt_data=True, wrt_traj=True)(
            sample_points,
            shape=img_size,
            density=True,
            squeeze_dims=False,
        )
        self.img_size = img_size

    def _interpolate(self, traj, factor=2):
        """Torch interpolate function to upsample the trajectory"""
        return torch.nn.functional.interpolate(
            traj.moveaxis(1, -1),
            scale_factor=factor,
            mode=self.interpolation_mode,
            align_corners=True,
        ).moveaxis(-1, 1)

    def get_trajectory(self):
        traj = self.control.clone()
        for i in range(np.log2(self.current_decim).astype(int)):
            traj = self._interpolate(traj)

        return traj.reshape(-1, traj.shape[-1])

    def upscale(self, factor=2):
        """Upscaling the model."""
        self.control = torch.nn.Parameter(
            data=self._interpolate(self.control),
            requires_grad=True,
        )
        self.current_decim /= factor

    def forward(self, x):
        traj = self.get_trajectory()
        self.operator.samples = traj

        # Simulate the acquisition process
        kspace = self.operator.op(x)

        adjoint = self.operator.adj_op(kspace).abs()
        return adjoint / torch.mean(adjoint)


# %%
# Util function to plot the state of the model
# --------------------------------------------


def plot_state(axs, mri_2D, traj, control_points, recon, loss=None, save_name=None):
    axs = axs.flatten()
    axs[0].imshow(np.abs(mri_2D[0]), cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("MR Image")
    axs[1].scatter(*traj.T, s=0.5)
    axs[1].scatter(*control_points.T, s=1, color="r")
    axs[1].legend(["Trajectory", "Control Points"])
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


def upsample_optimizer(optimizer, new_optimizer, factor=2):
    """Upsample the optimizer."""
    for old_group, new_group in zip(optimizer.param_groups, new_optimizer.param_groups):
        for old_param, new_param in zip(old_group["params"], new_group["params"]):
            # Interpolate the weights
            old_weight = old_param.data
            # Interpolate optimizer states
            if old_param in optimizer.state:
                for key in optimizer.state[old_param].keys():
                    if isinstance(optimizer.state[old_param][key], torch.Tensor):
                        old_state = optimizer.state[old_param][key]
                        if old_state.ndim == 0:
                            new_state = old_state
                        else:
                            new_state = torch.nn.functional.interpolate(
                                old_state.moveaxis(1, -1), scale_factor=2, mode="linear"
                            ).moveaxis(-1, 1)
                        new_optimizer.state[new_param][key] = new_state
                    else:
                        new_optimizer.state[new_param][key] = optimizer.state[
                            old_param
                        ][key]
    return new_optimizer


# %%
# Setup model and optimizer
# -------------------------
init_traj = initialize_2D_radial(32, 256).astype(np.float32)
model = Model(init_traj, img_size=(256, 256))

# %%
# Setup data
# ----------
mri_2D = torch.from_numpy(np.flipud(bwdl.get_mri(4, "T1")[80, ...]).astype(np.float32))[
    None
]
mri_2D = mri_2D / torch.mean(mri_2D)
model.eval()
recon = model(mri_2D)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plot_state(axs, mri_2D, init_traj, model.control.detach().cpu().numpy(), recon)

# %%
# Start training loop
# -------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
image_files = []
model.train()
while model.current_decim >= 1:
    with tqdm(range(30), unit="steps") as tqdms:
        for i in tqdms:
            out = model(mri_2D)
            loss = torch.nn.functional.mse_loss(out, mri_2D[None, None])
            numpy_loss = (loss.detach().cpu().numpy(),)

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
            hashed = joblib.hash((i, "learn_traj", time.time()))
            filename = "/tmp/" + f"{hashed}.png"
            plt.clf()
            fig, axs = plt.subplots(2, 2, figsize=(10, 10), num=1)
            plot_state(
                axs,
                mri_2D,
                model.get_trajectory().detach().cpu().numpy(),
                model.control.detach().cpu().numpy(),
                out,
                losses,
                save_name=filename,
            )
            image_files.append(filename)
        if model.current_decim == 1:
            break
        else:
            model.upscale()
            optimizer = upsample_optimizer(
                optimizer, torch.optim.Adam(model.parameters(), lr=1e-3)
            )


# Make a GIF of all images.
imgs = [Image.open(img) for img in image_files]
imgs[0].save(
    "mrinufft_learn_traj_multires.gif",
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
        / "images"
    )
    shutil.copyfile(
        "mrinufft_learn_traj_multires.gif",
        final_dir / "mrinufft_learn_traj_multires.gif",
    )
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# sphinx_gallery_thumbnail_path = 'generated/autoexamples/images/mrinufft_learn_traj_multires.gif'

# %%
# .. image-sg:: /generated/autoexamples/images/mrinufft_learn_traj_multires.gif
#    :alt: example learn_samples
#    :srcset: /generated/autoexamples/images/mrinufft_learn_traj_multires.gif
#    :class: sphx-glr-single-img

# %%
# Trained trajectory
# ------------------
model.eval()
recon = model(mri_2D)
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
plot_state(
    axs,
    mri_2D,
    model.get_trajectory().detach().cpu().numpy(),
    model.control.detach().cpu().numpy(),
    recon,
    losses,
)
plt.show()
