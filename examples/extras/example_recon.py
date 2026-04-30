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
3. Total Variation (TV)-regularized reconstruction solved with proximal
   gradient descent.

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
from deepinv.optim.data_fidelity import L2
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
if hasattr(L, "get"):
    L = L.get()

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
x_wavelet = wavelet_model(y, physics)


# %%
# Total variation reconstruction with proximal gradient descent
# ------------------------------------------------------------
#
# As an additional model-based reconstruction baseline, we reconstruct the
# image using a Total Variation (TV) prior. TV regularization promotes images
# that are piecewise smooth while preserving sharp edges, which is useful in
# MRI where anatomical structures often have smooth regions separated by
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
# Since the MRI image is complex-valued, and TVPrior is applied to real-valued
# tensors here, we apply the TV proximal operator separately to the real and
# imaginary parts before recombining them.


class ComplexTVPrior(TVPrior):
    
    """TV prior for complex-valued images.

    TVPrior is designed for real-valued tensors. Since MRI reconstructions are
    complex-valued, we apply the TV prior independently to the real and
    imaginary parts.
    """
    def prox(self, x, *args, gamma=None, **kwargs):
        """Apply TV proximal operator separately to real and imaginary parts."""

        if torch.is_complex(x):
            x_real = super().prox(x.real, *args, gamma=gamma, **kwargs)
            x_imag = super().prox(x.imag, *args, gamma=gamma, **kwargs)
            return x_real + 1j * x_imag

        return super().prox(x, *args, gamma=gamma, **kwargs)

    def forward(self, x, *args, **kwargs):
        """Compute TV cost for complex tensors."""
        if torch.is_complex(x):
            return super().forward(x.real, *args, **kwargs) + super().forward(
                x.imag, *args, **kwargs
            )

        return super().forward(x, *args, **kwargs)


tv = ComplexTVPrior(n_it_max=20)

# The TV regularization weight was selected separately using Optuna.
lamb_tv = 0.05789015101052105

tv_model = optim_builder(
    iteration="PGD",
    prior=tv,
    data_fidelity=data_fidelity,
    max_iter=20,
    params_algo={
        "stepsize": stepsize,
        "lambda": lamb_tv,
    },
)

x_tv = tv_model(y, physics)

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
x_tv_mag = torch.abs(x_tv)

print(f"Adjoint PSNR: {psnr(x_adjoint_mag, x_ref).item():.2f}")
print(f"Wavelet PSNR: {psnr(x_wavelet_mag, x_ref).item():.2f}")
print(f"TV PSNR: {psnr(x_tv_mag, x_ref).item():.2f}")

print(f"Adjoint SSIM: {ssim(x_adjoint_mag, x_ref).item():.4f}")
print(f"Wavelet SSIM: {ssim(x_wavelet_mag, x_ref).item():.4f}")
print(f"TV SSIM: {ssim(x_tv_mag, x_ref).item():.4f}")


# %%
# Visualize the reconstructions
# -----------------------------
#
# We compare the ground-truth image, the adjoint reconstruction, the wavelet
# reconstruction, and the TV reconstruction on the same slice.

slice_idx = mri.shape[-1] // 2 - 5

fig, axes = plt.subplots(1, 4, figsize=(16, 6))

images = [
    (torch.abs(mri[..., slice_idx]).detach().cpu(), "Ground truth"),
    (torch.abs(x_dagger[0, 0, ..., slice_idx]).detach().cpu(), "Adjoint"),
    (torch.abs(x_wavelet[0, 0, ..., slice_idx]).detach().cpu(), "Wavelet"),
    (torch.abs(x_tv[0, 0, ..., slice_idx]).detach().cpu(), "TV"),
]

for ax, (image, title) in zip(axes, images):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
