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
problems and compare different regularization priors.

"""

# %%
# Imports
# -------
import numpy as np
import matplotlib.pyplot as plt
from brainweb_dl import get_mri
from deepinv.optim import PDCP
from deepinv.optim.data_fidelity import L2, L2Distance
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.prior import WaveletPrior, TVPrior
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
    squeeze_dims=False,
)
y = fourier_op.op(mri)  # Simulate k-space data
noise_level = y.abs().max().item() * 0.0002
y += noise_level * (torch.randn_like(y) + 1j * torch.randn_like(y))


# %%
# Setup the physics and prior
# With ``viewed_as_real=True``, the MRI-NUFFT DeepInverse interface exposes
# complex-valued MRI tensors through real-valued tensor representations
# compatible with DeepInverse optimization methods.
#
# The wrapper automatically converts between:
#
# - image-space tensors:
#   ``(B, 2C, D, H, W)`` real-packed
#   ↔
#   ``(B, C, D, H, W)`` complex
#
# - k-space tensors:
#   ``(B, C, N, 2)``
#   ↔
#   ``(B, C, N)`` complex
physics = fourier_op.make_deepinv_phy(viewed_as_real=True)

# Complex-valued physics for methods that need complex tensors directly
physics_complex = fourier_op.make_deepinv_phy()

y_real = torch.view_as_real(y.contiguous())

wavelet = WaveletPrior(
    wv="sym8",
    wvdim=3,
    level=3,
    is_complex=True,
)

# %%
# Initial reconstruction with adjoint
x_dagger = physics_complex.A_dagger(y)

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
x_wavelet = wavelet_model(y, physics_complex)


# %%
# Total variation reconstruction with DeepInverse PDCP
# ----------------------------------------------------
#
# As an additional model-based reconstruction baseline, we reconstruct the
# image using a Total Variation (TV) prior. TV regularization promotes
# piecewise-smooth images while preserving sharp edges, which is useful in MRI
# because anatomical structures often contain smooth regions separated by clear
# boundaries.
#
# We solve the following variational problem:
#
# .. math::
#
#    \min_x \frac{1}{2}\|Ax - y\|_2^2 + \lambda \operatorname{TV}(x)
#
# where A is the non-Cartesian MRI forward operator, y is the measured k-space
# data, and x is the reconstructed image.
#
# ``TVPrior`` operates on real-valued image tensors. With
# ``viewed_as_real=True``, complex MRI images are exposed through a real-valued
# channel representation:
#
# .. math::
#
#    (B, C, D, H, W)_{\mathbb{C}}
#    \longrightarrow
#    (B, 2C, D, H, W)_{\mathbb{R}}
#
# This keeps the tensor compatible with DeepInverse priors expecting 4D or 5D
# real-valued image tensors, while internally preserving the complex-valued MRI
# representation.

lamb_tv = 50
tv = TVPrior(n_it_max=20)


# %%
# We now solve the TV-regularized MRI reconstruction problem using the
# official DeepInverse PDCP optimizer. PDCP implements a Chambolle-Pock
# primal-dual splitting method for objectives of the form F(Kx) + lambda G(x).
#
# In this formulation, K is the MRI forward operator A. Therefore, F acts on
# the predicted k-space data A(x), not directly on the image x.
#
# We use ``L2Distance`` as the data-fidelity term. Since ``L2Distance`` directly
# compares its two inputs, the monitored objective must be evaluated as:
#
# .. math::
#
#    \frac{1}{2}\|A(x) - y\|_2^2 + \lambda \operatorname{TV}(x)
#
# We therefore define a custom cost function to compare ``A(x)`` with the
# measured k-space data ``y``. Without this custom cost function, the default
# objective would incorrectly compare the image tensor ``x`` with the k-space
# measurements ``y``.
#
# The optimization problem remains:
#
# .. math::
#
#    \min_x \frac{1}{2}\|Ax - y\|_2^2 + \lambda \operatorname{TV}(x)
#
# Both arguments of the data-fidelity term are k-space tensors..

def pdcp_cost_fn(x, data_fidelity, prior, cur_params, y, physics):
    return data_fidelity(cur_params["K"](x), y) + cur_params["lambda"] * prior(x)


pdcp_model = PDCP(
    K=physics.A,
    K_adjoint=physics.A_adjoint,
    data_fidelity=L2Distance(),
    prior=tv,
    lambda_reg=lamb_tv,
    stepsize=stepsize,
    stepsize_dual=1.0,
    max_iter=20,
    g_first=False,
    cost_fn=pdcp_cost_fn,
)



x_pdcp_real = pdcp_model(y_real, physics)


# Convert from 5D real-packed to complex for visualization
b, c2, *spatial = x_pdcp_real.shape
c = c2 // 2
x_pdcp = torch.view_as_complex(
    x_pdcp_real.reshape(b, c, 2, *spatial).movedim(2, -1).contiguous()
)


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

psnr = PSNR()
ssim = SSIM()

x_ref = torch.abs(mri).unsqueeze(0).unsqueeze(0)
x_adjoint_mag = torch.abs(x_dagger)
x_wavelet_mag = torch.abs(x_wavelet)
x_pdcp_mag = torch.abs(x_pdcp)

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
# We compare the ground-truth image, the adjoint reconstruction, the wavelet
# reconstruction, and the TV-PDCP reconstruction.

slice_idx = mri.shape[-1] // 2 - 5

fig, axes = plt.subplots(1, 4, figsize=(20, 6))

images = [
    (torch.abs(mri[..., slice_idx]).detach().cpu(), "Ground truth"),
    (torch.abs(x_dagger[0, 0, ..., slice_idx]).detach().cpu(), "Adjoint"),
    (torch.abs(x_wavelet[0, 0, ..., slice_idx]).detach().cpu(), "Wavelet"),
    (torch.abs(x_pdcp[0, 0, ..., slice_idx]).detach().cpu(), "TV-PDCP"),
]

for ax, (image, title) in zip(axes, images):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
