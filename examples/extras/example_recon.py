# %%
"""
Model-based iterative reconstruction
====================================

This example demonstrates how to reconstruct image from 
non-Cartesian k-space data with a regularization prior, using deepinv. 

"""

# %%

# %%
# Imports
# -------
import numpy as np
import matplotlib.pyplot as plt
from brainweb_dl import get_mri
from deepinv.optim.prior import WaveletPrior
from deepinv.optim.prior import TVPrior
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder

from mrinufft import get_operator
from mrinufft.trajectories import initialize_3D_cones
import torch
import os


BACKEND = os.environ.get("MRINUFFT_BACKEND", "cufinufft")

# %%
# Get MRI data, 3D FLORET trajectory, and simulate k-space data
samples_loc = initialize_3D_cones(32 * 64, Ns=256, nb_zigzags=100)
# Load and downsample MRI data for speed
mri = (
    torch.Tensor(np.ascontiguousarray(get_mri(0)[::2, ::2, ::2][::-1, ::-1]))
    .to(torch.complex64)
    .to("cuda")
)

# %%
# Simulate k-space data
fourier_op = get_operator(BACKEND)(
    samples_loc,
    shape=mri.shape,
    density="pipe",
)
y = fourier_op.op(mri)  # Simulate k-space data
noise_level = y.abs().max().item() * 0.001
y_noisy = y + 0.01 * torch.randn_like(y) + 0.01j * torch.randn_like(y)

physics = fourier_op.make_deepinv_phy(wrt_data=True)
autograd = fourier_op.make_autograd(wrt_data=True)

# %%
physics.__dict__

# %%

# %%

x_dagger = physics.A_dagger(y)
wavelet = WaveletPrior(
    wv="sym8",
    wvdim=3,
    level=3,
    is_complex=True,
)

data_fidelity = L2()
# Algorithm parameters
lamb = 0.1
stepsize = 0.8 * float(1 / fourier_op.get_lipschitz_cst().get())
params_algo = {"stepsize": stepsize, "lambda": lamb}
max_iter = 100
early_stop = True

# Instantiate the algorithm class to solve the problem.
model = optim_builder(
    iteration="PGD",
    prior=wavelet,
    data_fidelity=data_fidelity,
    early_stop=early_stop,
    max_iter=max_iter,
    params_algo=params_algo,
)

x_model, metrics = model(y, physics, x_gt=torch.abs(mri), compute_metrics=True)


x_model
