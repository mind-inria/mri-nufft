# %%
"""
Model-based iterative reconstruction
====================================

This example demonstrates how to reconstruct a 3D MRI image from undersampled
non-Cartesian k-space measurements using MRI-NUFFT and DeepInverse.

We first simulate non-Cartesian acquisitions from a reference MRI volume using
a 3D cones trajectory. We then compare several reconstruction approaches:

1. Adjoint reconstruction, providing a fast baseline with sampling artifacts.
2. Wavelet-regularized reconstruction solved with FISTA.
3. Total Variation reconstruction solved with the DeepInverse PDCP optimizer.

The goal of this example is to illustrate how MRI-NUFFT physics operators can
be coupled with DeepInverse optimization tools to solve model-based MRI inverse
problems and compare different regularization priors, supporting both
complex-valued and real-valued (applied on the real and imaginary part)
regularizations.

"""

# %%
# Imports
# -------
import numpy as np
import matplotlib.pyplot as plt
from brainweb_dl import get_mri
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.prior import WaveletPrior, TVPrior
from mrinufft import get_operator
from mrinufft.trajectories import initialize_3D_cones
from mrinufft.operators import kspace_as_real
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
    squeeze_dims=False,
)
y = fourier_op.op(mri)  # Simulate k-space data
noise_level = y.abs().max().item() * 0.0002
y += noise_level * (torch.randn_like(y) + 1j * torch.randn_like(y))

# %%
# real-valued physics
physics = fourier_op.make_deepinv_phy(viewed_as_real=True)

y_real = kspace_as_real(y).float()

wavelet = WaveletPrior(
    wv="sym8",
    wvdim=3,
    level=3,
    is_complex=False,
)

# %%
# Initial reconstruction with adjoint
x_dagger = physics.A_dagger(y_real)

# %%
# Wavelet reconstruction with FISTA
# ---------------------------------
#
# The adjoint reconstruction is fast, but it contains artifacts due to
# undersampling. We therefore solve a regularized inverse problem using a
# wavelet sparsity prior.
#
# The reconstruction minimizes a data-fidelity term together with a wavelet
# regularization term:
#
# .. math::
#
#    \min_x \frac{1}{2}\|Ax - y\|_2^2 + \lambda \|Wx\|_1
#
# where A is the MRI forward operator, y is the measured k-space data, and W is
# a wavelet transform. The L2 data-fidelity term enforces consistency with the
# acquired measurements, while the wavelet prior promotes sparse image
# representations.
#
# We use FISTA, an accelerated proximal-gradient algorithm, to solve this
# optimization problem.


# %%
# Setup and run the reconstruction algorithm
# Data fidelity term
data_fidelity = L2()
# Algorithm parameters
lamb = 1e1
L = fourier_op.get_lipschitz_cst()
stepsize = 0.8 / float(L)

params_algo = {"stepsize": stepsize, "lambda": lamb, "a": 3}
max_iter = 100
early_stop = True

# %%
# Instantiate the algorithm class to solve the problem.
wavelet_model = optim_builder(
    iteration="FISTA",
    prior=wavelet,
    data_fidelity=data_fidelity,
    early_stop=early_stop,
    max_iter=max_iter,
    params_algo=params_algo,
)
x_wavelet = wavelet_model(y_real, physics)

# %%
# Total variation reconstruction with PDCP
# ----------------------------------------------------
#
# We reconstruct the image using a Total Variation (TV) prior solved with
# the Chambolle-Pock primal-dual algorithm (PDCP). TV promotes piecewise-
# smooth images while preserving sharp edges.
#
# We solve:
#
# .. math::
#
#    \min_x \frac{1}{2}\|Ax - y\|_2^2 + \lambda \operatorname{TV}(x)

from deepinv.optim import PDCP
from deepinv.optim.data_fidelity import L2Distance


def pdcp_cost_fn(x, data_fidelity, prior, cur_params, y, physics):
    return data_fidelity(cur_params["K"](x), y) + cur_params["lambda"] * prior(x)


lamb_tv = 50
tv = TVPrior(n_it_max=20)
stepsize_pdcp = 1.0 / float(L)

pdcp_model = PDCP(
    K=physics.A,
    K_adjoint=physics.A_adjoint,
    data_fidelity=L2Distance(),
    prior=tv,
    lambda_reg=lamb_tv,
    stepsize=stepsize_pdcp,
    stepsize_dual=1.0,
    max_iter=20,
    g_first=False,
    cost_fn=pdcp_cost_fn,
)

x_pdcp_real = pdcp_model(y_real, physics)


# %%
# Quantitative evaluation
# -----------------------
#
# We evaluate the reconstructions with PSNR and SSIM. PSNR measures the
# reconstruction fidelity with respect to the reference image: higher PSNR
# means lower pixel-wise error. SSIM measures structural similarity and is often
# more informative for images because it compares local contrast and structure.
#
# Metrics are computed on magnitude images, since the reconstructions are
# complex-valued.

# %%
from deepinv.loss.metric import PSNR, SSIM

psnr = PSNR(max_pixel=None)
ssim = SSIM()

x_ref = torch.abs(mri).unsqueeze(0).unsqueeze(0)

from mrinufft.operators.autodiff import image_as_cpx


def to_magnitude(x):
    return torch.abs(image_as_cpx(x))


x_adjoint_mag = to_magnitude(x_dagger)
x_wavelet_mag = to_magnitude(x_wavelet)
x_pdcp_mag = to_magnitude(x_pdcp_real)

print(f"Adjoint PSNR: {psnr(x_adjoint_mag, x_ref).item():.2f}")
print(f"Wavelet PSNR: {psnr(x_wavelet_mag, x_ref).item():.2f}")
print(f"TV-PDCP PSNR: {psnr(x_pdcp_mag, x_ref).item():.2f}")

print(f"Adjoint SSIM: {ssim(x_adjoint_mag, x_ref).item():.4f}")
print(f"Wavelet SSIM: {ssim(x_wavelet_mag, x_ref).item():.4f}")
print(f"TV-PDCP SSIM: {ssim(x_pdcp_mag, x_ref).item():.4f}")

# %%
# Visualize the reconstructions
# -----------------------------
#
# We compare the ground-truth image, the adjoint reconstruction, and the
# wavelet reconstruction.

slice_idx = mri.shape[-1] // 2 - 5

fig, axes = plt.subplots(1, 4, figsize=(20, 6))

images = [
    (torch.abs(mri[..., slice_idx]).detach().cpu(), "Ground truth"),
    (torch.abs(x_dagger[0, 0, ..., slice_idx]).detach().cpu(), "Adjoint"),
    (torch.abs(x_wavelet[0, 0, ..., slice_idx]).detach().cpu(), "Wavelet"),
    (torch.abs(x_pdcp_mag[0, 0, ..., slice_idx]).detach().cpu(), "TV-PDCP"),
]

for ax, (image, title) in zip(axes, images):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
