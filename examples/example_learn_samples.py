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
        self.operator = get_operator("gpunufft", wrt_data=True, wrt_traj=True)(
            initialize_2D_radial(10, 192).reshape(-1, 2).astype(np.float32),
            shape=(256, 256),
            density=True,
            squeeze_dims=False,
        )
        self.trajectory = self.operator.samples_torch


    def forward(self, x):
        kspace = self.operator.op(x)
        adjoint = self.operator.adj_op(kspace)
        return adjoint


# %%
# Create sample data
# ------------------

mri_2D = torch.Tensor(bwdl.get_mri(4, "T1")[80, ...].astype(np.complex64))[None]

print(mri_2D.shape)

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


for i in range(100):
    out = model(mri_2D)
    loss = torch.norm(out - mri_2D[None])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
    traj = model.trajectory.detach().cpu().numpy()
    plt.cla()
    plt.scatter(*traj.T, s=1)
    plt.pause(0.1)
