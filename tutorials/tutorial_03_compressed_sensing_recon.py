# %%
"""
.. _tutorial_compressed_sensing_recon:

=======================================================
Beyond the adjoint: compressed-sensing reconstruction
=======================================================

:ref:`tutorial_nufft_basics` and :ref:`tutorial_density_compensation` both
reconstruct images with (a corrected variant of) the adjoint operator. This
tutorial explains why that family of methods plateaus under heavy
undersampling, and how solving an actual optimization problem -- with a
sparsity prior -- goes further.

This example uses `DeepInverse <https://deepinv.github.io/deepinv/>`_ to
solve the reconstruction problem, and requires the ``autodiff`` extra
(``pip install mri-nufft[autodiff]``).

"""

# %%
# Where adjoint-based reconstruction plateaus
# ==============================================
#
# Both the naive and the density-compensated adjoint are *direct*
# (non-iterative) approximations of the inverse of the forward model
# :math:`A` (the NUFFT operator): they apply :math:`A^H`, optionally
# re-weighted, and stop there. Neither uses everything we know: that the
# underlying image is not arbitrary noise, but a natural image with
# structure (piecewise-smooth regions, sparse edges, ...).
#
# Under heavy undersampling (few samples relative to image size), the
# inverse problem :math:`Ax = y` is under-determined -- many images explain
# the same measurements -- and no amount of density re-weighting can recover
# information that was simply never measured.

# %%
# Formulating a regularized least-squares problem
# ==================================================
#
# Instead of directly inverting :math:`A`, we look for the image that best
# explains the measurements while also being "plausible" according to some
# prior. This is written as:
#
# .. math::
#
#    \min_x \frac{1}{2}\|Ax - y\|_2^2 + \lambda \|Wx\|_1
#
# - the data-fidelity term :math:`\|Ax - y\|_2^2` comes directly from the
#   forward model and enforces consistency with the acquired k-space data;
# - the regularization term :math:`\|Wx\|_1` encodes a prior: natural images
#   are *compressible* in a wavelet basis :math:`W` (most wavelet
#   coefficients are close to zero), so penalizing the :math:`\ell_1` norm of
#   the wavelet coefficients favors images that are sparse in that domain --
#   this is what makes the problem well-posed again even when
#   under-determined.

# %%
# Imports
# -------
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import brainweb_dl as bwdl
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.prior import WaveletPrior
from deepinv.loss.metric import PSNR

from mrinufft import get_operator
from mrinufft.trajectories import initialize_2D_radial
from mrinufft.operators import kspace_as_real
from mrinufft.operators.autodiff import image_as_cpx

BACKEND = os.environ.get("MRINUFFT_BACKEND", "finufft")
# Kept on CPU regardless of GPU availability: this problem is small enough to
# stay fast there, and it keeps the tutorial consistent with a CPU-only
# `finufft` backend instead of silently round-tripping tensors to the GPU.
DEVICE = "cpu"

# %%
# Setup: same ground truth and trajectory as the previous tutorials
# --------------------------------------------------------------------

mri_2D = np.flipud(bwdl.get_mri(4, "T1")[80, ...]).astype(np.float32)
mri_2D /= np.sqrt(np.mean(np.abs(mri_2D) ** 2))
mri_2D = torch.from_numpy(mri_2D).to(torch.complex64).to(DEVICE)

samples = initialize_2D_radial(Nc=32, Ns=256).astype(np.float32)  # heavily undersampled

fourier_op = get_operator(BACKEND)(
    samples, shape=mri_2D.shape, density="pipe", squeeze_dims=False
)

y = fourier_op.op(mri_2D)
noise_level = y.abs().max().item() * 0.0002
y += noise_level * (torch.randn_like(y) + 1j * torch.randn_like(y))

# %%
# Recap: the two adjoint-based baselines
# -----------------------------------------

physics = fourier_op.make_deepinv_phy(viewed_as_real=True)
y_real = kspace_as_real(y).float()

x_adjoint = physics.A_dagger(y_real)  # density-compensated adjoint (pipe)

# %%
# Solving it with FISTA, and why lambda matters
# =================================================
#
# We solve the wavelet-regularized problem with FISTA, an accelerated
# proximal-gradient algorithm well suited to this kind of non-smooth
# (:math:`\ell_1`) regularization.
#
# The regularization strength ``lambda`` controls how much weight the
# sparsity prior gets relative to the data-fidelity term: small values stay
# close to a plain (regularized-least-squares) fit of the data, while large
# values push harder for a sparse-in-wavelets solution, at the risk of
# discarding genuine image detail along with noise/artifacts. We show this
# directly by sweeping a few values rather than asserting a single "correct"
# one.

wavelet = WaveletPrior(wv="sym8", wvdim=2, level=3, is_complex=False)
data_fidelity = L2()
stepsize = 0.8 / float(fourier_op.get_lipschitz_cst())

x_ref = torch.abs(mri_2D).unsqueeze(0).unsqueeze(0)
psnr_metric = PSNR(max_pixel=None)


def to_magnitude(x):
    """Convert a deepinv real-valued tensor back to a magnitude image."""
    return torch.abs(image_as_cpx(x))


results = {}
for lamb in (0.1, 1.0, 1e1):
    model = optim_builder(
        iteration="FISTA",
        prior=wavelet,
        data_fidelity=data_fidelity,
        early_stop=True,
        max_iter=100,
        params_algo={"stepsize": stepsize, "lambda": lamb, "a": 3},
    )
    x_wavelet = model(y_real, physics)
    results[lamb] = to_magnitude(x_wavelet)

# %%
# Closing comparison
# =====================
#
# We compare the ground truth, the density-compensated adjoint from
# :ref:`tutorial_density_compensation`, and the wavelet-regularized
# reconstructions for each ``lambda``.

images = [
    (torch.abs(mri_2D).detach().cpu(), "Ground truth"),
    (to_magnitude(x_adjoint)[0, 0].detach().cpu(), "Density-compensated adjoint"),
] + [
    (results[lamb][0, 0].detach().cpu(), f"Wavelet-CS ($\\lambda$={lamb:g})")
    for lamb in results
]

fig, axs = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
for ax, (image, title) in zip(axs, images):
    ax.imshow(image, cmap="gray")
    ax.axis("off")
    if title != "Ground truth":
        p = psnr_metric(image.unsqueeze(0).unsqueeze(0), x_ref.cpu()).item()
        title = f"{title}\nPSNR: {p:.2f} dB"
    ax.set_title(title)
fig.tight_layout()
plt.show()

# %%
# The wavelet-regularized reconstruction recovers structure that neither the
# naive adjoint (:ref:`tutorial_nufft_basics`) nor the density-compensated
# adjoint (:ref:`tutorial_density_compensation`) could -- at the cost of
# solving an iterative optimization problem instead of a single linear
# operation, and of choosing a regularization strength.
