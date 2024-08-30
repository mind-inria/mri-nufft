# %%
"""
===============================================
Learn Sampling pattern with multi-resolution
===============================================

A small pytorch example to showcase learning k-space sampling patterns.
This example showcases the auto-diff capabilities of the NUFFT operator 
wrt to k-space trajectory in mri-nufft.

In this example we learn the k-space samples :math:`\mathbf{K}` for the following cost function:

.. math::
    \mathbf{\hat{K}} =  arg \min_{\mathbf{K}} ||  \mathcal{F}_\mathbf{K}^* D_\mathbf{K} \mathcal{F}_\mathbf{K} \mathbf{x} - \mathbf{x} ||_2^2
    
where :math:`\mathcal{F}_\mathbf{K}` is the forward NUFFT operator and :math:`D_\mathbf{K}` is the density compensators for trajectory :math:`\mathbf{K}`,  :math:`\mathbf{x}` is the MR image which is also the target image to be reconstructed.

Additionally, in-order to converge faster, we also learn the trajectory in a multi-resolution fashion. This is done by first optimizing a 8 times decimated trajectory locations, called control points. After a fixed number of iterations (5 in this example), these control points are upscaled by a factor of 2. However, note that the NUFFT operator always holds linearly interpolated version of the control points as k-space sampling trajectory.

.. note::
    This example can run on a binder instance as it is purely CPU based backend (finufft), and is restricted to a 2D single coil toy case.

.. warning::
    This example only showcases the autodiff capabilities, the learned sampling pattern is not scanner compliant as the scanner gradients required to implement it violate the hardware constraints. In practice, a projection :math:`\Pi_\mathcal{Q}(\mathbf{K})` into the scanner constraints set :math:`\mathcal{Q}` is recommended (see [Proj]_). This is implemented in the proprietary SPARKLING package [Sparks]_. Users are encouraged to contact the authors if they want to use it.
"""
# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install mri-nufft[finufft]

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
from mrinufft.trajectories import initialize_2D_radial

# %%
# Setup a simple class to learn trajectory
# ----------------------------------------
# .. note::
#     While we are only learning the NUFFT operator, we still need the gradient `wrt_data=True` to have all the gradients computed correctly.
#     See [Projector]_ for more details.


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
        self.operator = get_operator("finufft", wrt_data=True, wrt_traj=True)(
            sample_points,
            shape=img_size,
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
        """Function to get trajectory, which is interpolated version of control points."""
        traj = self.control.clone()
        for i in range(np.log2(self.current_decim).astype(int)):
            traj = self._interpolate(traj)

        return traj.reshape(-1, traj.shape[-1])

    def upscale(self, factor=2):
        """Upscaling the model.
        In this step, the number of control points are doubled and interpolated.
        """
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


def plot_state(
    axs, mri_2D, traj, recon, control_points=None, loss=None, save_name=None
):
    axs = axs.flatten()
    axs[0].imshow(np.abs(mri_2D[0]), cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("MR Image")
    axs[1].scatter(*traj.T, s=0.5)
    if control_points is not None:
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
            # Interpolate optimizer states
            if old_param in optimizer.state:
                for key in optimizer.state[old_param].keys():
                    if isinstance(optimizer.state[old_param][key], torch.Tensor):
                        old_state = optimizer.state[old_param][key]
                        if old_state.ndim == 0:
                            new_state = old_state
                        else:
                            new_state = torch.nn.functional.interpolate(
                                old_state.moveaxis(1, -1),
                                scale_factor=factor,
                                mode="linear",
                            ).moveaxis(-1, 1)
                        new_optimizer.state[new_param][key] = new_state
                    else:
                        new_optimizer.state[new_param][key] = optimizer.state[
                            old_param
                        ][key]
    return new_optimizer


# %%
# Setup Inputs (models, trajectory and image)
# -------------------------------------------
# First we create the model with a simple radial trajectory (32 shots of 256 points)

init_traj = initialize_2D_radial(32, 256).astype(np.float32)
model = Model(init_traj, img_size=(256, 256))
model.eval()

# %%
# The image on which we are going to train.
# .. note ::
#    In practice we would use instead a dataset (e.g. fastMRI)
#

mri_2D = torch.from_numpy(np.flipud(bwdl.get_mri(4, "T1")[80, ...]).astype(np.float32))[
    None
]
mri_2D = mri_2D / torch.mean(mri_2D)


# Initialisation
# --------------
# Before training, here is the simple reconstruction we have using a
# density compensated adjoint.

recon = model(mri_2D)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plot_state(axs, mri_2D, init_traj, recon, model.control.detach().cpu().numpy())

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
                out,
                model.control.detach().cpu().numpy(),
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
        Path(os.getcwd()).parent / "docs" / "generated" / "autoexamples" / "images"
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
    recon=recon,
    control_points=None,
    loss=losses,
)
plt.show()

# %%
# .. note::
#   The above learned trajectory is not that good because:
#    - The trajectory is trained only for 5 iterations per decimation level, resulting in a suboptimal trajectory.
#    - In order to make the example CPU compliant, we had to resort to preventing density compensation, hence the reconstructor is not good.
#
# Users are requested to checkout :ref:`sphx_glr_generated_autoexamples_GPU_example_learn_samples.py` for example with density compensation.

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
