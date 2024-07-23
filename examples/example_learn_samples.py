# %%
"""
====================================
A simple example to learn trajectory
====================================

A small pytorch example to showcase learning k-space sampling patterns.
This example showcases the auto-diff capabilities of the NUFFT operator 
wrt to k-space trajectory in mri-nufft
"""
import brainweb_dl as bwdl
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

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
        

def plot_state(axs, mri_2D, traj, recon, loss=None):
    axs = axs.flatten()
    axs[0].imshow(np.abs(mri_2D[0]), cmap='gray')
    axs[0].axis('off')
    axs[0].set_title("MR Image")
    axs[1].scatter(*traj.T, s=1)
    axs[1].set_title("Trajectory")
    axs[2].imshow(np.abs(recon[0][0].detach().cpu().numpy()), cmap='gray')
    axs[2].axis('off')
    axs[2].set_title("Reconstruction")
    if loss is not None:
        axs[3].plot(loss)
        axs[3].set_title("Loss")
    plt.show()

# %%
# Setup model and optimizer
# -------------------------
init_traj = initialize_2D_radial(32, 512).reshape(-1, 2).astype(np.float32)
model = Model(init_traj)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
schedulder = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1,
    end_factor=0.01,
    total_iters=1000
)
# %%
# Setup data
# ----------

mri_2D = torch.Tensor(np.flipud(
    bwdl.get_mri(4, "T1")[80, ...]
).astype(np.complex64))[None]
mri_2D = mri_2D / torch.linalg.norm(mri_2D)
model.eval()
recon = model(mri_2D)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plot_state(axs, mri_2D, init_traj, recon)

# %%
# Start training loop
# -------------------
losses = []
model.train()
with tqdm(range(1000), unit='steps') as tqdms:
    for i in tqdms:
        out = model(mri_2D)
        loss = torch.norm(out - mri_2D[None])
        numpy_loss = loss.detach().cpu().numpy()
        tqdms.set_postfix({"loss": numpy_loss})
        losses.append(numpy_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedulder.step()


# %%
# Trained trajectory
# ------------------
model.eval()
recon = model(mri_2D)
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
plot_state(axs, mri_2D, model.trajectory.detach().cpu().numpy(), recon, losses)
plt.show()