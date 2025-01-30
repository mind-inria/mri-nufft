# %%
"""
=========================================
Learning sampling pattern with decimation
=========================================

An example using PyTorch to showcase learning k-space sampling patterns with decimation.

This example showcases the auto-differentiation capabilities of the NUFFT operator
with respect to the k-space trajectory in MRI-nufft.

Hereafter we learn the k-space sample locations :math:`\mathbf{K}` using the following cost function:

.. math::
    \mathbf{\hat{K}} =  arg \min_{\mathbf{K}} ||  \mathcal{F}_\mathbf{K}^* D_\mathbf{K} \mathcal{F}_\mathbf{K} \mathbf{x} - \mathbf{x} ||_2^2

where :math:`\mathcal{F}_\mathbf{K}` is the forward NUFFT operator,
:math:`D_\mathbf{K}` is the density compensator for trajectory :math:`\mathbf{K}`,
and :math:`\mathbf{x}` is the MR image which is also the target image to be reconstructed.

Additionally, in order to converge faster, we also learn the trajectory in a multi-resolution fashion.
This is done by first optimizing x8 times decimated trajectory locations, called control points.
After a fixed number of iterations (5 in this example), these control points are upscaled by a factor of 2.
Note that the NUFFT operator always holds linearly interpolated version of the control points as k-space sampling trajectory.

.. note::
    This example can run on a binder instance as it is purely CPU based backend (finufft), and is restricted to a 2D single coil toy case.

.. warning::
    This example only showcases the auto-differentiation capabilities, the learned sampling pattern
    is not scanner compliant as the gradients required to implement it violate the hardware constraints.
    In practice, a projection :math:`\Pi_\mathcal{Q}(\mathbf{K})` onto the scanner constraints set :math:`\mathcal{Q}` is recommended
    (see [Cha+16]_). This is implemented in the proprietary SPARKLING package [Cha+22]_.
    Users are encouraged to contact the authors if they want to use it.
"""
# %%
# .. colab-link::
#    :needs_gpu: 0
#
#    !pip install mri-nufft[finufft]

import time

import brainweb_dl as bwdl
import joblib
import matplotlib.pyplot as plt
import numpy as np
import tempfile as tmp
import torch
from PIL import Image, ImageSequence
from tqdm import tqdm

from mrinufft import get_operator
from mrinufft.trajectories import initialize_2D_radial

# %%
# Utils
# =====
#
# Model class
# -----------
# .. note::
#     While we are only learning the NUFFT operator, we still need the gradient `wrt_data=True` to have all the gradients computed correctly.
#     See [GRC23]_ for more details.


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
# State plotting
# --------------


def plot_state(axs, image, traj, recon, control_points=None, loss=None, save_name=None):
    axs = axs.flatten()
    # Upper left reference image
    axs[0].imshow(np.abs(image[0]), cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("MR Image")

    # Upper right trajectory
    axs[1].scatter(*traj.T, s=0.5)
    if control_points is not None:
        axs[1].scatter(*control_points.T, s=1, color="r")
        axs[1].legend(
            ["Trajectory", "Control points"], loc="right", bbox_to_anchor=(2, 0.6)
        )
    axs[1].grid(True)
    axs[1].set_title("Trajectory")
    axs[1].set_xlim(-0.5, 0.5)
    axs[1].set_ylim(-0.5, 0.5)
    axs[1].set_aspect("equal")

    # Down left reconstructed image
    axs[2].imshow(np.abs(recon[0][0].detach().cpu().numpy()), cmap="gray")
    axs[2].axis("off")
    axs[2].set_title("Reconstruction")

    # Down right loss evolution
    if loss is not None:
        axs[3].plot(loss)
        axs[3].set_ylim(0, None)
        axs[3].grid("on")
        axs[3].set_title("Loss")
        plt.subplots_adjust(hspace=0.3)

    # Save & close
    if save_name is not None:
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# %%
# Optimizer upscaling
# -------------------


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
# Data preparation
# ================
#
# A single image to train the model over. Note that in practice
# we would use a whole dataset instead (e.g. fastMRI).
#

volume = np.flip(bwdl.get_mri(4, "T1"), axis=(0, 1, 2))
image = torch.from_numpy(volume[-80, ...].astype(np.float32))[None]
image = image / torch.mean(image)

# %%
# A basic radial trajectory with an acceleration factor of 8.

AF = 8
initial_traj = initialize_2D_radial(image.shape[1] // AF, image.shape[2]).astype(
    np.float32
)


# %%
# Trajectory learning
# ===================
#
# Initialisation
# --------------

model = Model(initial_traj, img_size=image.shape[1:])
model = model.eval()

# %%
# The image obtained before learning the sampling pattern
# is highly degraded because of the acceleration factor and simplicity
# of the trajectory.

initial_recons = model(image)

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
plot_state(axs, image, initial_traj, initial_recons)


# %%
# Training loop
# -------------

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()

losses = []
image_files = []
while model.current_decim >= 1:
    with tqdm(range(30), unit="steps") as tqdms:
        for i in tqdms:
            out = model(image)
            loss = torch.nn.functional.mse_loss(out, image[None, None])
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
            filename = f"{tmp.NamedTemporaryFile().name}.png"
            plt.clf()
            fig, axs = plt.subplots(2, 2, figsize=(10, 10), num=1)
            plot_state(
                axs,
                image,
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

# %%

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

# %%
# .. image-sg:: /generated/autoexamples/images/mrinufft_learn_traj_multires.gif
#    :alt: example learn_samples
#    :srcset: /generated/autoexamples/images/mrinufft_learn_traj_multires.gif
#    :class: sphx-glr-single-img

# %%
# Results
# -------

model.eval()
final_recons = model(image)
final_traj = model.get_trajectory().detach().cpu().numpy()

# %%

fig, axs = plt.subplots(1, 3, figsize=(9, 3))
plot_state(axs, image, final_traj, final_recons)
plt.show()

# %%
#
# The learned trajectory above improves the reconstruction quality as compared to
# the initial trajectory shown above. Note of course that the reconstructed
# image is far from perfect because of the documentation rendering constraints.
# In order to improve the results one can start by training it for more than
# just 5 iterations per decimation level. Also density compensation should be used,
# even though it was avoided here for CPU compliance. Check out
# :ref:`sphx_glr_generated_autoexamples_GPU_example_learn_samples.py` to know more.


# %%
# References
# ==========
#
# .. [Cha+16] N. Chauffert, P. Weiss, J. Kahn and P. Ciuciu, "A Projection Algorithm for
#           Gradient Waveforms Design in Magnetic Resonance Imaging," in
#           IEEE Transactions on Medical Imaging, vol. 35, no. 9, pp. 2026-2039, Sept. 2016,
#           doi: 10.1109/TMI.2016.2544251.
# .. [Cha+22] G. R. Chaithya, P. Weiss, G. Daval-Frérot, A. Massire, A. Vignaud and P. Ciuciu,
#           "Optimizing Full 3D SPARKLING Trajectories for High-Resolution Magnetic
#           Resonance Imaging," in IEEE Transactions on Medical Imaging, vol. 41, no. 8,
#           pp. 2105-2117, Aug. 2022, doi: 10.1109/TMI.2022.3157269.
# .. [GRC23] Chaithya GR, and Philippe Ciuciu. 2023. "Jointly Learning Non-Cartesian
#           k-Space Trajectories and Reconstruction Networks for 2D and 3D MR Imaging
#           through Projection" Bioengineering 10, no. 2: 158.
#           https://doi.org/10.3390/bioengineering10020158
