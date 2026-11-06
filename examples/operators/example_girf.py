"""
=======================================
GIRF-based correction of the encoding
=======================================

An example to show how to predict the fields actually played by the gradient
chain with a gradient impulse response function (GIRF), and how to use each
order of this prediction in the reconstruction, on top of the static
off-resonance.

The GIRF predicts the deviations of the field from the nominal gradients,
expanded on spherical harmonics. Each order enters the reconstruction
differently:

- the zeroth order is a phase of the data,
- the first order is the trajectory played,
- the second order, as the static off-resonance, varies in both time and space,
  and requires an extended model.

See :ref:`girf` for the details of the model and its conventions.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from brainweb_dl import get_mri
from scipy.ndimage import zoom
from scipy.spatial.transform import Rotation

from mrinufft import get_operator
from mrinufft.density import voronoi
from mrinufft.extras import (
    cg,
    compute_girf_interpolators,
    get_orc_factorization,
    make_b0map,
)
from mrinufft.operators import MRIFourierCorrected
from mrinufft.trajectories import (
    Acquisition,
    Hardware,
    apply_girf,
    initialize_2D_spiral,
    spherical_harmonics,
)

plt.rcParams["image.cmap"] = "gray"

# %%
# Acquisition
# ===========
#
# Image and geometry
# ------------------
#
# A slice of the BrainWeb dataset, installable using ``pip install brainweb-dl``,
# is imaged in an oblique plane, 5 cm above the isocentre. The geometry is given
# by an affine from the voxel indices to the physical coordinates in m, as in the
# header of the raw data.

volume = np.flip(get_mri(0, "T1"), axis=(0, 1, 2))[90]
volume = np.pad(volume, ((0, 0), (18, 18)))
image = zoom(volume, 128 / volume.shape[0], order=1)
image = image / image.max()
mask = image > 0.1
shape = image.shape

fov = 0.24
rotation = Rotation.from_euler("xyz", [20, -15, 30], degrees=True).as_matrix()
res = fov / shape[0]
affine = np.eye(4)
affine[:3, :3] = rotation * res
affine[:3, 3] = [0.01, -0.02, 0.05] - affine[:3, :2] @ (np.array(shape) // 2)

raster_time = 10e-6
acq = Acquisition(
    fov=(fov, fov),
    img_size=shape,
    hardware=Hardware(grad_raster_time=raster_time),
    affine=affine,
)

# %%
# Trajectory and static off-resonance
# -----------------------------------
#
# The image is sampled by a spiral of 16 shots of 15 ms. The static
# off-resonance is given in Hz, in the convention of
# :class:`~mrinufft.operators.off_resonance.MRIFourierCorrected`.

nominal = initialize_2D_spiral(16, 1536, nb_revolutions=4)
readout_time = np.arange(nominal.shape[1]) * raster_time
b0_map, _ = make_b0map(shape, b0range=(-80, 80), mask=mask)

fig, axs = plt.subplots(1, 3, figsize=(10, 3.5))
axs[0].imshow(image)
axs[0].set_title("Image")
axs[1].imshow(np.where(mask, b0_map, np.nan), cmap="bwr", vmin=-80, vmax=80)
axs[1].set_title("Static off-resonance (Hz)")
for shot in nominal[::4]:
    axs[2].plot(*shot.T, lw=0.5)
axs[2].set_aspect("equal")
axs[2].set_title("Nominal trajectory")
for ax in axs[:2]:
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# A synthetic GIRF
# ================
#
# Measured GIRFs are read with :func:`mrinufft.io.read_girf`. Here, a synthetic
# one models
#
# - the self terms as a low-pass filter of 20 µs, delayed by 4 µs,
# - eddy currents of 1 ms towards the B0 and second-order terms,
# - a mechanical resonance at 1.2 kHz of the x and y coils towards the
#   :math:`xz` and :math:`zy` terms, as often measured.
#
# The terms follow the order of
# :func:`~mrinufft.trajectories.spherical_harmonics`:
# ``(B0, x, y, z, xy, zy, 3z^2 - r^2, xz, x^2 - y^2)``.

freqs = np.arange(0, 0.5 / raster_time + 1, 50.0)
s = 2j * np.pi * freqs
taper = 0.5 * (1 + np.cos(np.pi * np.clip((freqs - 20e3) / 20e3, 0, 1)))
eddy = s * 1e-3 / (1 + s * 1e-3)
f0, quality = 1.2e3, 5
resonance = s / (2 * np.pi * f0 * quality)
resonance = resonance / (1 + resonance + (s / (2 * np.pi * f0)) ** 2)

girf = np.zeros((3, 9, len(freqs)), dtype=complex)
girf[[0, 1, 2], [1, 2, 3]] = np.exp(-s * 4e-6) / (1 + s * 20e-6)
girf[:, 0] = np.array([[3e-5], [-2e-5], [4e-5]]) * eddy  # m
girf[0, 7] = 0.3 * resonance  # x -> xz, 1/m
girf[1, 5] = 0.3 * resonance  # y -> zy
girf[0, 4] = 0.05 * eddy  # x -> xy
girf[1, 8] = 0.05 * eddy  # y -> x2 - y2
girf[2, 6] = 0.1 * eddy  # z -> 3z2 - r2
girf *= taper

fig, axs = plt.subplots(1, 3, figsize=(11, 3), sharex=True)
for i, name in enumerate("xyz"):
    axs[0].plot(freqs / 1e3, np.abs(girf[i, i + 1]), label=f"{name} -> {name}")
    axs[1].plot(freqs / 1e3, 1e6 * np.abs(girf[i, 0]), label=f"{name} -> B0")
for (i, j), name in {
    (0, 7): "x -> xz",
    (1, 5): "y -> zy",
    (0, 4): "x -> xy",
    (2, 6): "z -> 3z2-r2",
}.items():
    axs[2].plot(freqs / 1e3, np.abs(girf[i, j]), label=name)
axs[0].set_title("Self terms")
axs[1].set_title("B0 terms (µm)")
axs[2].set_title("Second-order terms (1/m)")
for ax in axs:
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (kHz)")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()

# %%
# Prediction of the encoding
# ==========================
#
# :func:`~mrinufft.trajectories.apply_girf` returns the trajectory played and the
# fields of the deviations from the nominal encoding, expanded in the frame of the
# image, between the samples of the trajectory. As the slice is off-centre, the
# second-order terms also deviate the first and zeroth orders. The phases accrued
# by these fields, in rad/m^l for a term of order l, give the zeroth and higher
# orders at the samples of the trajectory.

played, fields = apply_girf(girf=girf, freqs=freqs, trajectory=nominal, acq=acq)
phase = 2 * np.pi * acq.gamma * raster_time * np.cumsum(fields, axis=1)
phase = np.pad(phase, ((0, 0), (1, 0), (0, 0)))

# Positions of the voxels of the brain in the frame of the image, in m.
positions = (np.argwhere(mask) - np.array(shape) // 2) * res
higher = phase[0, :, 4:] @ spherical_harmonics(positions, 9)[:, 4:].T
time = 1e3 * readout_time

fig, axs = plt.subplots(1, 3, figsize=(11, 3))
axs[0].plot(time, phase[0, :, 0])
axs[0].set_title("Zeroth order (rad)")
deviation = (played - nominal + 0.5) % 1 - 0.5
axs[1].plot(time, shape[0] * deviation[0])
axs[1].set_title("First order, k-space (cells)")
axs[2].plot(time, np.abs(higher).max(axis=1))
axs[2].set_title("Second order, max over the brain (rad)")
for ax in axs:
    ax.set_xlabel("Time (ms)")
plt.tight_layout()
plt.show()

# %%
# Simulation of the data
# ======================
#
# The data are simulated with the whole model: the zeroth order as a phase, the
# NUFFT on the trajectory played, and the second order with the static
# off-resonance, factorized with a high accuracy.

b0_factors = get_orc_factorization("svd")(
    field_map=b0_map, readout_time=readout_time, mask=mask, L=20
)[:2]
interpolators = compute_girf_interpolators(
    fields, acq, tol=1e-6, interpolators=b0_factors
)
nufft = get_operator("finufft")(played, shape)
truth = MRIFourierCorrected(nufft, interpolator=interpolators)
kspace = np.exp(-1j * phase[..., 0].ravel()) * truth.op(image)

# %%
# Reconstructions
# ===============
#
# All the reconstructions correct for the static off-resonance, as usual, and
# add the orders of the GIRF prediction one by one. They use the conjugate
# gradient, started from the density compensated adjoint, and are compared with
# the reconstruction by the exact model, to isolate the errors of the models
# from those of the solver. The second order and the static off-resonance are
# factorized together with :func:`~mrinufft.extras.compute_girf_interpolators`,
# from the factorization of the off-resonance alone.


def nufft_for(samples):
    """NUFFT operator, with a density compensation for the samples."""
    return get_operator("finufft")(samples, shape, density=voronoi(samples))


def reconstruct(operator, data):
    """Least-squares reconstruction, scaled to the image."""
    estimate = np.squeeze(cg(operator, data, max_iter=20, progressbar=False))
    return (
        estimate
        * np.vdot(estimate[mask], image[mask])
        / np.vdot(estimate[mask], estimate[mask])
    )


compensated = kspace * np.exp(1j * phase[..., 0].ravel())
reference = reconstruct(
    MRIFourierCorrected(nufft_for(played), interpolator=interpolators), compensated
)

b0_factors = get_orc_factorization("svd")(
    field_map=b0_map, readout_time=readout_time, mask=mask, L=10
)[:2]
girf_factors = compute_girf_interpolators(
    fields, acq, mask=mask, interpolators=b0_factors
)
print(
    f"{b0_factors[0].shape[1]} interpolators for the static off-resonance, "
    f"{girf_factors[0].shape[1]} with the second order"
)
estimates = {
    "Nominal trajectory": reconstruct(
        MRIFourierCorrected(nufft_for(nominal), interpolator=b0_factors), kspace
    ),
    "+ GIRF orders 0 and 1": reconstruct(
        MRIFourierCorrected(nufft_for(played), interpolator=b0_factors), compensated
    ),
    "+ GIRF order 2": reconstruct(
        MRIFourierCorrected(nufft_for(played), interpolator=girf_factors),
        compensated,
    ),
}

fig, axs = plt.subplots(2, len(estimates) + 1, figsize=(14, 6.2))
axs[0, 0].imshow(np.abs(reference), vmin=0, vmax=1)
axs[0, 0].set_title("Exact model")
axs[1, 0].imshow(image, vmin=0, vmax=1)
axs[1, 0].set_title("Image")
for (name, estimate), column in zip(estimates.items(), axs[:, 1:].T, strict=True):
    difference = np.abs(estimate - reference) * mask
    nrmse = np.linalg.norm(difference) / np.linalg.norm(reference[mask])
    column[0].imshow(np.abs(estimate), vmin=0, vmax=1)
    column[0].set_title(f"{name}\nNRMSE {nrmse:.1e}")
    column[1].imshow(difference, vmin=0, vmax=0.1)
    column[1].set_title("Difference to the exact model")
for ax in axs.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
# Correcting for the static off-resonance alone leaves the errors of the
# encoding. The zeroth and first orders of the GIRF prediction remove most of
# them, and the second order the rest, which grows away from the centre of the
# image. Off the isocentre, the second-order terms also deviate the zeroth and
# first orders, which :func:`~mrinufft.trajectories.apply_girf` accounts for by
# expanding the fields in the frame of the image.
