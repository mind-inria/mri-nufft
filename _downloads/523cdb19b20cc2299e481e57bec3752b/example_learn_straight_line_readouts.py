# %%
"""
===================================
Learn Straight line readout pattern
===================================

A small pytorch example to showcase learning k-space sampling patterns.
In this example we learn the 2D sampling pattern for a 3D MRI image, assuming
straight line readouts. This example showcases the auto-diff capabilities of the NUFFT operator
The image resolution is kept small to reduce computation time.

.. warning::
    This example only showcases the autodiff capabilities, the learned sampling pattern is not scanner compliant as the scanner gradients required to implement it violate the hardware constraints. In practice, a projection :math:`\Pi_\mathcal{Q}(\mathbf{K})` into the scanner constraints set :math:`\mathcal{Q}` is recommended (see [Proj]_). This is implemented in the proprietary SPARKLING package [Sparks]_. Users are encouraged to contact the authors if they want to use it.
"""

# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install mri-nufft[gpunufft]

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


# %%
# Setup a simple class to learn trajectory
# ----------------------------------------
# .. note::
#     While we are only learning the NUFFT operator, we still need the gradient `wrt_data=True` to have all the gradients computed correctly.
#     See [Projector]_ for more details.


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
            data=torch.Tensor(
                np.random.random((num_shots - self.central_points.shape[0], 2)) - 0.5
            ),
            requires_grad=True,
        )
        self.operator = get_operator("gpunufft", wrt_data=True, wrt_traj=True)(
            np.random.random(
                (self.get_2D_points().shape[0] * self.num_samples_per_shot, 3)
            )
            - 0.5,
            shape=img_size,
            density=True,
            squeeze_dims=False,
        )

    def get_trajectory(self, get_as_shot=False):
        samples = self._get_3D_points(self.get_2D_points())
        if not get_as_shot:
            return samples
        return samples.reshape(-1, self.num_samples_per_shot, 3)

    def get_2D_points(self):
        return torch.vstack([self.central_points, self.non_center_points])

    def _get_3D_points(self, samples2D):
        line = torch.linspace(
            -0.5,
            0.5,
            self.num_samples_per_shot,
            device=samples2D.device,
            dtype=samples2D.dtype,
        )
        return torch.stack(
            [
                line.repeat(samples2D.shape[0], 1),
                samples2D[:, 0].repeat(self.num_samples_per_shot, 1).T,
                samples2D[:, 1].repeat(self.num_samples_per_shot, 1).T,
            ],
            dim=-1,
        ).reshape(-1, 3)

    def forward(self, x):
        self.operator.samples = self.get_trajectory()
        kspace = self.operator.op(x)
        adjoint = self.operator.adj_op(kspace).abs()
        return adjoint / torch.mean(adjoint)


# %%
# Util function to plot the state of the model
# --------------------------------------------


def plot_state(mri_2D, traj, recon, loss=None, save_name=None, i=None):
    fig_grid = (2, 2)
    if loss is None:
        fig_grid = (1, 3)
    fig, axs = plt.subplots(*fig_grid, figsize=tuple(i * 5 for i in fig_grid[::-1]))
    axs = axs.flatten()
    axs[0].imshow(np.abs(mri_2D[0][..., 11]), cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("MR Image")
    if traj.shape[-1] == 3:
        if i is not None and i > 20:
            axs[1].scatter(*traj.T[1:3, 0], s=10, color="blue")
        else:
            fig_kwargs = {}
            plt_kwargs = {"s": 1, "alpha": 0.2}
            if i is not None:
                fig_kwargs["azim"], fig_kwargs["elev"] = (
                    i / 25 * 60 - 60,
                    30 - i / 25 * 30,
                )
                plt_kwargs["alpha"] = 0.2 + 0.8 * i / 20
                plt_kwargs["s"] = 1 + 9 * i / 20
            axs[1].remove()
            axs[1] = fig.add_subplot(*fig_grid, 2, projection="3d", **fig_kwargs)
            for shot in traj:
                axs[1].scatter(*shot.T, color="blue", **plt_kwargs)
    else:
        axs[1].scatter(*traj.T, s=10)
    axs[1].set_title("Trajectory")
    axs[2].imshow(np.abs(recon[0][0][..., 11].detach().cpu().numpy()), cmap="gray")
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
# Setup model and optimizer
# -------------------------

cart_data = np.flipud(bwdl.get_mri(4, "T1")).T[::8, ::8, ::8].astype(np.complex64)
model = Model(253, cart_data.shape)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
# %%
# Setup data
# ----------

mri_3D = torch.Tensor(cart_data)[None]
mri_3D = mri_3D / torch.mean(mri_3D)
model.eval()
recon = model(mri_3D)
plot_state(mri_3D, model.get_trajectory(True).detach().cpu().numpy(), recon)
# %%
# Start training loop
# -------------------
losses = []
image_files = []
model.train()
with tqdm(range(40), unit="steps") as tqdms:
    for i in tqdms:
        out = model(mri_3D)
        loss = torch.nn.functional.mse_loss(out, mri_3D[None])
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
        hashed = joblib.hash((i, "learn_line", time.time()))
        filename = "/tmp/" + f"{hashed}.png"
        plot_state(
            mri_3D,
            model.get_trajectory(True).detach().cpu().numpy(),
            out,
            losses,
            save_name=filename,
            i=i,
        )
        image_files.append(filename)

# Make a GIF of all images.
imgs = [Image.open(img) for img in image_files]
imgs[0].save(
    "mrinufft_learn_2d_sampling_pattern.gif",
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
        "mrinufft_learn_2d_sampling_pattern.gif",
        final_dir / "mrinufft_learn_2d_sampling_pattern.gif",
    )
except FileNotFoundError:
    pass

# sphinx_gallery_end_ignore

# sphinx_gallery_thumbnail_path = 'generated/autoexamples/GPU/images/mrinufft_learn_2d_sampling_pattern.gif'

# %%
# .. image-sg:: /generated/autoexamples/GPU/images/mrinufft_learn_2d_sampling_pattern.gif
#    :alt: example learn_samples
#    :srcset: /generated/autoexamples/GPU/images/mrinufft_learn_2d_sampling_pattern.gif
#    :class: sphx-glr-single-img

# %%
# Trained trajectory
# ------------------
model.eval()
recon = model(mri_3D)
plot_state(mri_3D, model.get_trajectory(True).detach().cpu().numpy(), recon, losses)
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
