# %%
"""
.. _tutorial_nufft_basics:

====================================
From k-space to image: NUFFT basics
====================================

Non-Cartesian MRI does not sample k-space on a regular grid, so the FFT
cannot be used directly to reconstruct an image. This tutorial builds the
intuition for *why* we would want to sample this way in the first place,
*why* a dedicated Non-Uniform FFT (NUFFT) operator is then needed, and *why*
the most naive way of using it (the adjoint) is not enough on its own.

"""

# %%
# Why go non-Cartesian at all ?
# ================================
#
# A standard Cartesian MRI acquisition samples k-space on a regular grid,
# one line at a time -- simple to reconstruct (a single FFT), but not
# particularly efficient to *acquire*: each line only covers a thin strip of
# k-space, and the center of k-space (which carries most of the image's
# energy/contrast) is visited only once.
#
# Non-Cartesian trajectories -- radial spokes, spirals, ... -- are designed
# instead to cover k-space efficiently with the scanner's gradient hardware,
# which brings several concrete benefits:
#
# - **Faster acquisitions**: a well-designed non-Cartesian readout can cover
#   more of k-space per unit time / per RF excitation than a Cartesian line,
#   directly reducing scan time.
# - **Built-in redundancy at the center of k-space**: e.g. every radial spoke
#   passes through the k-space center, so the low-frequency content is
#   sampled many times over. This makes these trajectories more robust to
#   motion (each readout is a partial, self-consistent snapshot) and gives
#   them a natural way to detect/correct it.
# - **Favorable undersampling behavior**: when k-space must be undersampled
#   to go even faster, Cartesian undersampling produces coherent, structured
#   aliasing (ghosting), while non-Cartesian undersampling tends to produce
#   incoherent, noise-like aliasing -- which is exactly what compressed-sensing
#   reconstructions (:ref:`tutorial_compressed_sensing_recon`) are good at
#   removing.
#
# The price to pay for these advantages is that the samples no longer lie on
# a regular grid, so we give up the ability to reconstruct with a plain FFT --
# which is precisely the problem the rest of this tutorial addresses.

# %%
# The cost of a non-uniform sampling grid
# ==========================================
#
# The (fast) Fourier Transform assumes that the k-space samples lie on a
# regular Cartesian grid. Non-Cartesian trajectories (radial, spiral, ...)
# instead sample k-space at arbitrary, non-uniformly spaced locations
# :math:`\mathbf{k}_i`, so the FFT's grid assumption is violated and it
# cannot be applied as-is.
#
# What we actually want to compute is the (adjoint) Non-Uniform Discrete
# Fourier Transform:
#
# .. math::
#
#    \hat{x}(\mathbf{r}) = \sum_{i=1}^{M} y_i \, e^{2 \pi i \mathbf{k}_i \cdot
#    \mathbf{r}}
#
# where :math:`y_i` are the measured k-space samples and :math:`\mathbf{r}`
# are image-space locations. This sum has to be evaluated for every voxel and
# every sample, i.e. a direct (non-uniform) discrete Fourier transform.
#
# See :ref:`NUFFT` for the full derivation, including the forward NUDFT and
# how the interpolation/gridding kernels make this tractable at scale.

# %%
# From a direct sum to a fast operator
# =======================================
#
# Evaluating the sum above directly costs :math:`O(NM)` operations (N voxels,
# M samples), which is intractable for realistic image sizes. NUFFT
# libraries (`finufft`, `cufinufft`, `gpuNUFFT`, ...) instead interpolate
# (grid) the non-uniform samples onto an oversampled regular grid, run a
# regular FFT, and correct for the interpolation kernel — bringing the cost
# down to roughly :math:`O(N \log N + M)`.
#
# MRI-NUFFT provides a single entry point, :func:`~mrinufft.get_operator`,
# that wraps all these backends behind the same interface, so we do not need
# to implement any of this ourselves.

# %%
# Imports
# -------
import os

import brainweb_dl as bwdl
import numpy as np
import matplotlib.pyplot as plt

from mrinufft import get_operator
from mrinufft.trajectories import initialize_2D_radial
from mrinufft.display import display_2D_trajectory

BACKEND = os.environ.get("MRINUFFT_BACKEND", "finufft")

# %%
# What do we get if we just take the adjoint ?
# ===============================================
#
# We simulate k-space data from a reference image using the forward NUFFT
# operator (``.op``), then reconstruct with the adjoint (``.adj_op``) — the
# simplest possible "reconstruction", obtained by applying the conjugate
# transpose of the forward operator.

mri_2D = np.flipud(bwdl.get_mri(4, "T1")[80, ...]).astype(np.float32)
mri_2D /= np.sqrt(np.mean(np.abs(mri_2D) ** 2))

samples = initialize_2D_radial(Nc=64, Ns=256).astype(np.float32)

nufft = get_operator(BACKEND)(samples, shape=mri_2D.shape, squeeze_dims=True)

kspace_data = nufft.op(mri_2D)
adjoint = nufft.adj_op(kspace_data)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(abs(mri_2D), cmap="gray")
axs[0].set_title("Ground truth")
display_2D_trajectory(samples, subfigure=axs[1])
axs[1].set_title("Radial trajectory")
axs[2].imshow(abs(adjoint), cmap="gray")
axs[2].set_title("Naive adjoint reconstruction")
for ax in (axs[0], axs[2]):
    ax.axis("off")
fig.tight_layout()
plt.show()

# %%
# Where the naive adjoint falls short
# ======================================
#
# The adjoint reconstruction above is blurrier and dimmer than the ground
# truth. Two distinct reasons compound here:
#
# - **Uneven sampling density**: a radial trajectory over-samples the center
#   of k-space and under-samples its periphery, so the adjoint over-weights
#   low frequencies. :ref:`tutorial_density_compensation` addresses this
#   directly.
# - **The adjoint is not a true inverse**: even with perfect density
#   weighting, the adjoint is only a good *approximation* of the inverse
#   problem :math:`\arg\min_x \|Ax - y\|_2^2` — it ignores everything we
#   might know about the image beyond the acquired samples.
#   :ref:`tutorial_compressed_sensing_recon` builds a real reconstruction
#   that goes beyond this approximation.
