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
4. Total Variation reconstruction solved with a primal-dual hybrid gradient
   algorithm inspired by Chambolle-Pock.   

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
import torch.nn as nn
from deepinv.models import TVDenoiser
from tqdm import tqdm
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
physics_complex = fourier_op.make_deepinv_phy()

# With viewed_as_real=True, complex tensors are represented as
# real-valued tensors with a final dimension of size 2:
# (..., 2) = (real part, imaginary part).
physics_real = fourier_op.make_deepinv_phy(viewed_as_real=True)

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
x_wavelet = wavelet_model(y, physics_complex)


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
# TVPrior is designed for real-valued tensors. With the new
# `viewed_as_real=True` option, complex MRI data are represented as
# real-valued tensors with a final dimension of size 2:
# (..., 2) = (real part, imaginary part).
# For 3D MRI data, this representation produces 6D tensors of shape
# (B, C, D, H, W, 2), while DeepInverse TVPrior expects 4D or 5D inputs.
# We therefore pack the final real/imaginary dimension into the channel
# dimension before applying TVPrior, and unpack it afterwards.
# This avoids manually splitting complex tensors into separate real and
# imaginary TV proximal operations.
# The TV regularization weight was selected separately using Optuna.

lamb_tv = 0.05789015101052105


class RealViewTVPrior(TVPrior):
    """Adapt TVPrior to tensors represented with torch.view_as_real.

    This helper packs the final real/imaginary dimension into the channel
    dimension before applying TV regularization, then restores the original
    layout afterwards.
    """

    def _pack(self, x):
        if x.shape[-1] != 2:
            return x, None

        original_shape = x.shape
        b, c, *spatial, two = original_shape
        x = x.movedim(-1, 2)          # (B, C, 2, D, H, W)
        x = x.reshape(b, c * 2, *spatial)  # (B, 2*C, D, H, W)
        return x, original_shape

    def _unpack(self, x, original_shape):
        if original_shape is None:
            return x

        b, c, *spatial, two = original_shape
        x = x.reshape(b, c, 2, *spatial)
        x = x.movedim(2, -1)          # back to (B, C, D, H, W, 2)
        return x

    def prox(self, x, *args, gamma=None, **kwargs):
        x_packed, original_shape = self._pack(x)
        out = super().prox(x_packed, *args, gamma=gamma, **kwargs)
        return self._unpack(out, original_shape)

    def forward(self, x, *args, **kwargs):
        x_packed, _ = self._pack(x)
        return super().forward(x_packed, *args, **kwargs)


class RealViewPDHGTV(nn.Module):
    """Primal-dual TV reconstruction for real-view complex MRI tensors.

    The input image is represented as (..., 2), where the last dimension stores
    the real and imaginary parts. Since TVDenoiser expects 4D or 5D tensors,
    we temporarily pack the real/imaginary dimension into the channel dimension.
    """

    def __init__(
        self,
        lambda_reg,
        max_iter,
        lipschitz,
        data_fidelity,
        stopping_criterion=1e-5,
        relaxation_param=1.0,
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.data_fidelity = data_fidelity
        self.stopping_criterion = stopping_criterion
        self.rho = relaxation_param

        # Primal step size for the image update.
        self.tau = 1.0 / lipschitz

        # Dual step size. The value 12 is a conservative bound for the squared
        # norm of the 3D finite-difference gradient operator.
        self.sigma = 0.9 / (self.tau * 12)

    @staticmethod
    def _pack_real_view(x):
        """Convert (B, C, D, H, W, 2) into (B, 2*C, D, H, W)."""
        if x.shape[-1] != 2:
            return x, None

        original_shape = x.shape
        b, c, *spatial, two = original_shape
        x = x.movedim(-1, 2)
        x = x.reshape(b, c * 2, *spatial)
        return x, original_shape

    @staticmethod
    def _unpack_real_view(x, original_shape):
        """Convert (B, 2*C, D, H, W) back into (B, C, D, H, W, 2)."""
        if original_shape is None:
            return x

        b, c, *spatial, two = original_shape
        x = x.reshape(b, c, 2, *spatial)
        x = x.movedim(2, -1)
        return x

    @staticmethod
    def _project_l2_ball_pointwise(p, radius):
        """Project the dual TV variable onto an L2 ball pointwise."""
        norm = torch.linalg.norm(p, dim=-1, keepdim=True)
        scale = torch.clamp(norm / radius, min=1.0)
        return p / scale

    def forward(self, y, physics, init):
        """Run primal-dual TV reconstruction."""
        x = init

        # TVDenoiser.nabla expects a 4D or 5D real tensor.
        x_packed, original_shape = self._pack_real_view(x)
        p = torch.zeros_like(TVDenoiser.nabla(x_packed))

        for _ in tqdm(range(self.max_iter)):
            x_old = x.clone()
            p_old = p.clone()

            # Pack x before applying the TV gradient.
            x_packed, original_shape = self._pack_real_view(x)

            # Dual update: update the TV dual variable.
            p = p + self.sigma * TVDenoiser.nabla(x_packed)
            p = self._project_l2_ball_pointwise(p, self.lambda_reg)

            # TV adjoint gradient, then unpack back to real-view MRI format.
            tv_grad_packed = TVDenoiser.nabla_adjoint(2.0 * p - p_old)
            tv_grad = self._unpack_real_view(tv_grad_packed, original_shape)

            # Primal update: data-fidelity gradient + TV contribution.
            data_grad = self.data_fidelity.grad(x, y, physics)
            x = x - self.tau * (data_grad + tv_grad)

            # Optional relaxation.
            x = x_old + self.rho * (x - x_old)
            p = p_old + self.rho * (p - p_old)

            rel_err = torch.linalg.norm(
                x_old.flatten() - x.flatten()
            ) / (torch.linalg.norm(x.flatten()) + 1e-12)

            if rel_err < self.stopping_criterion:
                break

        return x


tv = RealViewTVPrior(n_it_max=20)

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

x_tv_real = tv_model(y_real, physics_real)
x_tv = torch.view_as_complex(x_tv_real.contiguous())


# %%
# Total variation reconstruction with primal-dual hybrid gradient
# ---------------------------------------------------------------
#
# We now solve the same TV-regularized MRI reconstruction problem using the
# primal-dual hybrid gradient algorithm inspired by Chambolle-Pock
#
# Compared with proximal gradient descent, this primal-dual approach introduces
# an additional dual variable associated with the TV regularization term.
# This often leads tofaster convergence and better handling of non-smooth priors such as Total
# Variation.
#
# The optimization problem remains:
#
# .. math::
#
#    \min_x \frac{1}{2}\|Ax - y\|_2^2 + \lambda \operatorname{TV}(x)
#
# but the optimization is performed using alternating primal and dual updates.

pdhg_tv_model = RealViewPDHGTV(
    lambda_reg=lamb_tv,
    max_iter=20,
    lipschitz=float(L),
    data_fidelity=data_fidelity,
)

x0_real = torch.view_as_real(x_dagger.contiguous())

x_pdhg_real = pdhg_tv_model(
    y_real,
    physics_real,
    init=x0_real,
)

x_pdhg = torch.view_as_complex(x_pdhg_real.contiguous())

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
x_pdhg_mag = torch.abs(x_pdhg)

print(f"Adjoint PSNR: {psnr(x_adjoint_mag, x_ref).item():.2f}")
print(f"Wavelet PSNR: {psnr(x_wavelet_mag, x_ref).item():.2f}")
print(f"TV-PGD PSNR: {psnr(x_tv_mag, x_ref).item():.2f}")
print(f"TV-PDHG PSNR: {psnr(x_pdhg_mag, x_ref).item():.2f}")

print(f"Adjoint SSIM: {ssim(x_adjoint_mag, x_ref).item():.4f}")
print(f"Wavelet SSIM: {ssim(x_wavelet_mag, x_ref).item():.4f}")
print(f"TV-PGD SSIM: {ssim(x_tv_mag, x_ref).item():.4f}")
print(f"TV-PDHG SSIM: {ssim(x_pdhg_mag, x_ref).item():.4f}")

# %%
# Visualize the reconstructions
# -----------------------------
#
# We compare the ground-truth image, the adjoint reconstruction, the wavelet
# reconstruction, the TV-PGD reconstruction, and the TV-PDHG reconstruction.

slice_idx = mri.shape[-1] // 2 - 5

fig, axes = plt.subplots(1, 5, figsize=(20, 6))

images = [
    (torch.abs(mri[..., slice_idx]).detach().cpu(), "Ground truth"),
    (torch.abs(x_dagger[0, 0, ..., slice_idx]).detach().cpu(), "Adjoint"),
    (torch.abs(x_wavelet[0, 0, ..., slice_idx]).detach().cpu(), "Wavelet"),
    (torch.abs(x_tv[0, 0, ..., slice_idx]).detach().cpu(), "TV-PGD"),
    (torch.abs(x_pdhg[0, 0, ..., slice_idx]).detach().cpu(), "TV-PDHG"),
]

for ax, (image, title) in zip(axes, images):
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
