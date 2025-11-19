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
samples_loc = initialize_3D_cones(32 * 32, Ns=256, nb_zigzags=16, width=3)
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
noise_level = y.abs().max().item() * 0.0002
y += noise_level * (torch.randn_like(y) + 1j * torch.randn_like(y))


# %%
# Setup the physics and prior
physics = fourier_op.make_deepinv_phy()
wavelet = WaveletPrior(
    wv="sym8",
    wvdim=3,
    level=3,
    is_complex=True,
)

# %%
# Initial reconstruction with adjoint
x_dagger = physics.A_dagger(y)

# %%
# Setup and run the reconstruction algorithm
# Data fidelity term
data_fidelity = L2()
# Algorithm parameters
lamb = 1e1
stepsize = 0.8 * float(1 / fourier_op.get_lipschitz_cst().get())
params_algo = {"stepsize": stepsize, "lambda": lamb, "a": 3}
max_iter = 100
early_stop = True

# %%
# Instantiate the algorithm class to solve the problem.
wavelet_recon = optim_builder(
    iteration="FISTA",
    prior=wavelet,
    data_fidelity=data_fidelity,
    early_stop=early_stop,
    max_iter=max_iter,
    params_algo=params_algo,
)
x_wavelet = wavelet_recon(y, physics)


# %%
# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(torch.abs(mri[..., mri.shape[2] // 2 - 5]).cpu(), cmap="gray")
plt.title("Ground truth")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(
    torch.abs(x_dagger[0, 0, ..., x_dagger.shape[2] // 2 - 5]).cpu(), cmap="gray"
)
plt.title("Adjoint reconstruction")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(
    torch.abs(x_wavelet[0, 0, ..., x_wavelet.shape[2] // 2 - 5]).cpu(), cmap="gray"
)
plt.title("Reconstruction with wavelet prior")
plt.axis("off")
plt.show()
