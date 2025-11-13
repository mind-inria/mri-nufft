# %%
r"""
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

import os
import brainweb_dl as bwdl
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import torch

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


BACKEND = os.environ.get("MRINUFFT_BACKEND", "finufft")


plt.rcParams["animation.embed_limit"] = 2**30  # 1GiB is very large.


class Model(torch.nn.Module):
    def __init__(
        self,
        inital_trajectory,
        img_size=(256, 256),
        start_decim=8,
        interpolation_mode="linear",
    ):
        super().__init__()
        self.control = torch.nn.Parameter(
            data=torch.Tensor(inital_trajectory[:, ::start_decim]),
            requires_grad=True,
        )
        self.current_decim = start_decim
        self.interpolation_mode = interpolation_mode
        sample_points = inital_trajectory.reshape(-1, inital_trajectory.shape[-1])
        self.operator = get_operator(BACKEND, wrt_data=True, wrt_traj=True)(
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
# Optimizer upscaling
# -------------------
#
# The multi-resolution training requires us to update the optimizer as well. The optimization weights will also be
# linearly interpolated.


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

N_upscale = 4

model = Model(initial_traj, img_size=image.shape[1:], start_decim=2 ** (N_upscale - 1))
model = model.eval()

# %%
# The image obtained before learning the sampling pattern
# is highly degraded because of the acceleration factor and simplicity
# of the trajectory.

initial_recons = model(image)


# %%
# Training loop
# -------------


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
num_epochs = 30


# setup plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("Training Starting")
axs = axs.flatten()

axs[0].imshow(np.abs(image.detach().cpu().numpy().squeeze()), cmap="gray")
axs[0].axis("off")
axs[0].set_title("MR Image")

traj_scat = axs[1].scatter(
    *model.get_trajectory().detach().cpu().numpy().T, s=0.5, c="tab:blue"
)
traj_scat2 = axs[1].scatter(*model.control.detach().cpu().numpy().T, s=2, c="tab:red")

axs[1].legend(["Trajectory", "Control Points"], loc="upper right")
axs[1].set_title("Trajectory")

recon_im = axs[2].imshow(
    np.abs(initial_recons.squeeze().detach().cpu().numpy()), cmap="gray"
)
axs[2].axis("off")
axs[2].set_title("Reconstruction")
(loss_curve,) = axs[3].plot([], [])
axs[3].grid()
axs[3].set_xlim(0, 1)
axs[3].set_xlabel("epochs")
axs[3].set_ylabel("loss")
# add line marking the decimation steps
[
    axs[3].axvline(num_epochs * i, c="tab:red", linestyle="dashed")
    for i in range(N_upscale)
]
fig.tight_layout()


def train():
    global optimizer
    losses = []
    while model.current_decim >= 1:
        for _ in range(num_epochs):
            out = model(image)
            loss = torch.nn.functional.mse_loss(out, image[None, None])
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            with torch.no_grad():
                # Clamp the value of trajectory between [-0.5, 0.5]
                for param in model.parameters():
                    param.clamp_(-0.5, 0.5)

            yield (
                out.detach().cpu().numpy(),
                model.get_trajectory().detach().cpu().numpy(),
                model.control.detach().cpu().numpy(),
                losses,
                model.current_decim,
            )

        if model.current_decim == 1:
            break
        else:
            model.upscale()
            optimizer = upsample_optimizer(
                optimizer, torch.optim.Adam(model.parameters(), lr=1e-3)
            )


def plot_epoch(data):
    recon, traj, control, losses, decim = data
    cur_epoch = len(losses)
    recon_im.set_data(abs(recon).squeeze())
    loss_curve.set_xdata(np.arange(cur_epoch))
    loss_curve.set_ydata(losses)
    traj_scat.set_offsets(traj)

    axs[3].set_xlim(0, cur_epoch)
    axs[3].set_ylim(0, 1.1 * max(losses))
    axs[2].set_title(f"Reconstruction, frame {cur_epoch}/{num_epochs*N_upscale}")
    axs[1].set_title(
        f"Trajectory, step {cur_epoch}/{num_epochs * N_upscale}, decim = {decim}"
    )

    traj_scat.set_offsets(traj.reshape(-1, 2))
    traj_scat2.set_offsets(control.reshape(-1, 2))

    if cur_epoch < num_epochs * N_upscale:
        fig.suptitle("Training in progress " + "." * (1 + cur_epoch % 3))
    else:
        fig.suptitle("Training complete !")


ani = animation.FuncAnimation(
    fig, plot_epoch, train, repeat=False, save_count=num_epochs, interval=50
)
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
