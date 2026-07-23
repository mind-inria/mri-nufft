# %%
"""
.. _tutorial_density_compensation:

===============================================
Why and how to compensate for sampling density
===============================================

In :ref:`tutorial_nufft_basics` we saw that the naive adjoint reconstruction
of a radial acquisition is biased towards low frequencies. This tutorial
explains why that happens, and how density compensation corrects for it --
as well as why it is only a partial fix.

"""

# %%
# The naive adjoint over-weights the k-space center
# =====================================================
#
# The adjoint operator sums the contribution of every acquired sample without
# accounting for how densely that region of k-space was sampled. Most
# non-Cartesian trajectories (radial, spiral, ...) sample the center of
# k-space much more densely than its periphery -- e.g. every radial spoke
# passes through the center, so it is visited by every readout, while the
# outer k-space is only touched once per spoke.
#
# The adjoint of an operator that over-represents certain k-space locations
# therefore over-weights their contribution to the reconstructed image: dense
# regions (the center, i.e. low spatial frequencies) dominate, giving a
# blurred, low-frequency-biased image -- exactly what we observed in
# :ref:`tutorial_nufft_basics`.

# %%
# Imports
# -------
import os

import brainweb_dl as bwdl
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

from mrinufft import get_operator
from mrinufft.density import get_density
from mrinufft.trajectories import initialize_2D_radial

BACKEND = os.environ.get("MRINUFFT_BACKEND", "finufft")

mri_2D = np.flipud(bwdl.get_mri(4, "T1")[80, ...]).astype(np.float32)
mri_2D /= np.sqrt(np.mean(np.abs(mri_2D) ** 2))

samples = initialize_2D_radial(Nc=64, Ns=256).astype(np.float32)

nufft = get_operator(BACKEND)(samples, shape=mri_2D.shape, squeeze_dims=True)
kspace_data = nufft.op(mri_2D)
adjoint = nufft.adj_op(kspace_data)


def show(images, titles):
    """Display a row of images with matching titles."""
    fig, axs = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    for ax, image, title in zip(axs, images, titles):
        ax.imshow(abs(image), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def rescale(image, reference):
    """Best-fit scalar rescaling of `image` onto `reference`'s intensity range.

    The (adjoint of the) NUFFT operator is not normalized to a physical
    intensity scale, so raw adjoint outputs live on an arbitrary scale that
    also changes with the density weighting used. Rescaling by the
    least-squares optimal scalar isolates the *shape* difference we actually
    care about here from this unrelated scale factor.
    """
    image = np.abs(image)
    alpha = np.sum(image * reference) / np.sum(image**2)
    return alpha * image


# %%
# How do we fix it ?
# ====================
#
# Density compensation re-weights each sample by (approximately) the inverse
# of the local sampling density, before the adjoint is applied, so that
# over-sampled regions stop dominating the reconstruction.
#
# For a radial trajectory, the density is analytically known: it grows
# linearly with distance to the k-space center, so we can compute the
# weights directly.

flat_samples = samples.reshape(-1, 2)
radial_weights = np.sqrt(np.sum(flat_samples**2, axis=1))
nufft_radial = get_operator(BACKEND)(
    samples, shape=mri_2D.shape, density=radial_weights, squeeze_dims=True
)
adjoint_radial = nufft_radial.adj_op(kspace_data)

# %%
# More generally, MRI-NUFFT exposes several density estimators through
# :func:`~mrinufft.density.get_density`, that work for arbitrary
# trajectories, not just radial ones:
#
# - ``"voronoi"``: weight inversely proportional to the area of each
#   sample's Voronoi cell -- geometrically exact, but slow, especially in 3D.
# - ``"pipe"``: an iterative estimate computed directly from the
#   interpolation/spreading kernels of the NUFFT operator.

voronoi_weights = get_density("voronoi", samples)
nufft_voronoi = get_operator(BACKEND)(
    samples, shape=mri_2D.shape, density=voronoi_weights, squeeze_dims=True
)
adjoint_voronoi = nufft_voronoi.adj_op(kspace_data)

adjoint_r = rescale(adjoint, mri_2D)
adjoint_radial_r = rescale(adjoint_radial, mri_2D)
adjoint_voronoi_r = rescale(adjoint_voronoi, mri_2D)

show(
    [mri_2D, adjoint_r, adjoint_radial_r, adjoint_voronoi_r],
    [
        "Ground truth",
        "No density compensation",
        "Analytical (radial) weights",
        "Voronoi weights",
    ],
)

print(
    f"No compensation   PSNR: {psnr(mri_2D, adjoint_r, data_range=mri_2D.max()):.2f} dB"
)
print(
    f"Radial weights    PSNR: {psnr(mri_2D, adjoint_radial_r, data_range=mri_2D.max()):.2f} dB"
)
print(
    f"Voronoi weights   PSNR: {psnr(mri_2D, adjoint_voronoi_r, data_range=mri_2D.max()):.2f} dB"
)

# %%
# Both compensation schemes visibly reduce the low-frequency bias and improve
# PSNR over the uncompensated adjoint.

# %%
# The limits of density compensation
# =====================================
#
# Density compensation is a *heuristic* correction of the adjoint, not a
# statistically optimal estimator: it approximately equalizes the
# contribution of each k-space region, but it does not solve the underlying
# inverse problem, and it does not account for noise or for prior knowledge
# about the image.
#
# It is also **not appropriate inside an iterative / compressed-sensing
# reconstruction**: iterative solvers already minimize a data-fidelity term
# built directly from the forward operator :math:`A` (e.g.
# :math:`\|Ax - y\|_2^2`). Inserting a density-compensated adjoint there
# would apply the weighting twice -- once implicitly through the solver's own
# handling of :math:`A^H A`, and once explicitly through the compensation --
# distorting the data-fidelity term and biasing the solution.
#
# :ref:`tutorial_compressed_sensing_recon` shows how to instead solve the
# inverse problem directly, without relying on density compensation at all.
