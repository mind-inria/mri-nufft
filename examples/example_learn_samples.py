# %%
"""
=============================
Density Compensation Routines
=============================

Examples of differents density compensation methods.

Density compensation depends on the sampling trajectory,and is apply before the
adjoint operation to act as preconditioner, and should make the lipschitz constant
of the operator roughly equal to 1.

"""
import brainweb_dl as bwdl
import matplotlib.pyplot as plt
import numpy as np
import torch

from mrinufft import check_backend, get_density, get_operator
from mrinufft.trajectories import initialize_2D_radial
from mrinufft.trajectories.display import display_2D_trajectory

# %%
# Create sample data
# ------------------

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.trajectory = torch.nn.Parameter(
            data=torch.Tensor(initialize_2D_radial(192, 192).astype(np.float32)),
            requires_grad=True,
        ) 
        self.operator = get_operator("gpunufft", wrt_data=True, wrt_traj=True)(
            self.trajectory.detach().cpu().numpy(),
            shape=(192, 192),
            density=True,
        )

    def forward(self, x):
        kspace = self.operator.op(x)
        adjoint = self.operator.adj_op(kspace)
        return adjoint


# %%
# Create sample data
# ------------------

mri_2D = torch.Tensor(bwdl.get_mri(4, "T1")[80, ...].astype(np.complex64))

print(mri_2D.shape)

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


for i in range(10):
    out = model(mri_2D)
    loss = torch.norm(out - mri_2D)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
    

# %%
# As you can see, the radial sampling pattern as a strong concentration of sampling point in the center, resulting in a  low-frequency biased adjoint reconstruction.

# %%
# Geometry based methods
# ======================
#
# Voronoi
# -------
#
# Voronoi Parcellation attribute a weights to each k-space coordinate, inversely
# proportional to its voronoi cell area.


# .. warning::
#    The current implementation of voronoi parcellation is CPU only, and is thus
#    **very** slow in 3D ( > 1h).

# %%
voronoi_weights = get_density("voronoi", traj)

nufft_voronoi = get_operator("finufft")(
    traj, shape=mri_2D.shape, density=voronoi_weights
)
adjoint_voronoi = nufft_voronoi.adj_op(kspace)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(abs(mri_2D))
axs[0].set_title("Ground Truth")
axs[1].imshow(abs(adjoint))
axs[1].set_title("no density compensation")
axs[2].imshow(abs(adjoint_voronoi))
axs[2].set_title("Voronoi density compensation")


# %%
# Cell Counting
# -------------
#
# Cell Counting attributes weights based on the number of trajectory point lying in a same k-space nyquist voxel.
# This can be viewed as an approximation to the voronoi neth

# .. note::
#    Cell counting is faster than voronoi (especially in 3D), but is less precise.

# The size of the niquist voxel can be tweak by using the osf parameter. Typically as the NUFFT (and by default in MRI-NUFFT) is performed at an OSF of 2


# %%
cell_count_weights = get_density("cell_count", traj, shape=mri_2D.shape, osf=2.0)

nufft_cell_count = get_operator("finufft")(
    traj, shape=mri_2D.shape, density=cell_count_weights, upsampfac=2.0
)
adjoint_cell_count = nufft_cell_count.adj_op(kspace)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(abs(mri_2D))
axs[0].set_title("Ground Truth")
axs[1].imshow(abs(adjoint))
axs[1].set_title("no density compensation")
axs[2].imshow(abs(adjoint_cell_count))
axs[2].set_title("cell_count density compensation")

# %%
# Manual Density Estimation
# -------------------------
#
# For some analytical trajectory it is also possible to determine the density compensation vector directly.
# In radial trajectory for instance, a sample's weight can be determined from its distance to the center.


# %%
flat_traj = traj.reshape(-1, 2)
weights = np.sqrt(np.sum(flat_traj**2, axis=1))
nufft = get_operator("finufft")(traj, shape=mri_2D.shape, density=weights)
adjoint_manual = nufft.adj_op(kspace)
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(abs(mri_2D))
axs[0].set_title("Ground Truth")
axs[1].imshow(abs(adjoint))
axs[1].set_title("no density compensation")
axs[2].imshow(abs(adjoint_manual))
axs[2].set_title("manual density compensation")

# %%
# Operator-based method
# =====================
#
# Pipe's Method
# -------------
# Pipe's method is an iterative scheme, that use the interpolation and spreading kernel operator for computing the density compensation.
#
# .. warning::
#    If this method is widely used in the literature, there exists no convergence guarantees for it.
#
# .. note::
#    The Pipe method is currently only implemented for gpuNUFFT.

# %%
if check_backend("gpunufft"):
    flat_traj = traj.reshape(-1, 2)
    nufft = get_operator("gpunufft")(
        traj, shape=mri_2D.shape, density={"name": "pipe", "osf": 2}
    )
    adjoint_manual = nufft.adj_op(kspace)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(abs(mri_2D))
    axs[0].set_title("Ground Truth")
    axs[1].imshow(abs(adjoint))
    axs[1].set_title("no density compensation")
    axs[2].imshow(abs(adjoint_manual))
    axs[2].set_title("Pipe density compensation")

    print(nufft.density)
