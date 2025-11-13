"""
======================================
Least Squares Image Reconstruction
======================================

An example to show how to reconstruct volumes using the least square estimate.

This script demonstrates the use of the Conjugate Gradient (CG), LSQR and LSMR
methods, to reconstruct images from non-uniform k-space data.

"""

import os
import time

import cupy as cp
import numpy as np
from brainweb_dl import get_mri
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

import mrinufft
from mrinufft.extras.optim import loss_l2_reg, loss_l2_AHreg


BACKEND = os.environ.get("MRINUFFT_BACKEND", "cufinufft")

# %%
# Setup Inputs
samples_loc = mrinufft.initialize_2D_spiral(Nc=64, Ns=512, nb_revolutions=8)
ground_truth = get_mri(sub_id=4)
ground_truth = ground_truth[90]
# Normalize the ground truth image
ground_truth = ground_truth / np.sqrt(np.mean(abs(ground_truth) ** 2))
image_gpu = cp.array(ground_truth)  # convert to cupy array for GPU processing

print("image size: ", ground_truth.shape)
# %%
# Setup the NUFFT operator
NufftOperator = mrinufft.get_operator(BACKEND)  # get the operator

nufft = NufftOperator(
    samples_loc,
    shape=ground_truth.shape,
    squeeze_dims=True,
)  # create the NUFFT operator

# %%
# Reconstruct the image using the CG method
kspace_data_gpu = nufft.op(image_gpu)  # get the k-space data
kspace_data = kspace_data_gpu.get()  # convert back to numpy array for display
adjoint = nufft.adj_op(kspace_data_gpu).get()  # adjoint NUFFT


def mixed_cb(*args, **kwargs):
    """A compound callback function, to track iterations time and convergence."""
    return [
        time.perf_counter(),
        loss_l2_reg(*args, **kwargs),
        loss_l2_AHreg(*args, **kwargs),
        psnr(
            abs(args[0].get().squeeze()),
            abs(ground_truth.squeeze()),
            data_range=ground_truth.max(),
        ),
        time.perf_counter(),
    ]


def process_cb_results(cb_results):
    t0, r, rH, psnrs, t1 = list(zip(*cb_results))
    t1 = (t0[0], *t1[:-1])
    time_it = np.cumsum(np.array(t0) - np.array(t1))
    r = [rr.get() for rr in r]
    rH = [rr.get() for rr in rH]

    return {"time": time_it, "res": r, "AHres": rH, "psnr": psnrs}


# Run the least-square minimization for all the solvers:

OPTIM = ["cg", "lsqr", "lsmr"]
METRICS = {
    "res": r"$\|Ax-b\|$",
    "AHres": r"$\|A^H(Ax-b)\|$",
    "psnr": "PSNR",
}


images = dict()
iterations_cb = dict()
for optim in OPTIM:
    image, iter_cb = nufft.pinv_solver(
        kspace_data=kspace_data_gpu,
        max_iter=1000,
        callback=mixed_cb,
        optim=optim,
    )
    images[optim] = image.get().squeeze()  # retrieve image from GPU.
    iterations_cb[optim] = process_cb_results(iter_cb)


# %%
# Display Convergence
# -------------------


fig, axs = plt.subplots(len(METRICS), 1, sharex=True, figsize=(8, 12))
for i, metric in enumerate(METRICS):
    for optim in OPTIM:
        if "res" in metric:
            axs[i].set_yscale("log")
        axs[i].plot(
            iterations_cb[optim]["time"],
            iterations_cb[optim][metric],
            marker="o",
            markevery=20,
            label=f"{optim} {np.mean(1/np.diff(iterations_cb[optim]['time'])):.2f}iters/s",
        )
    axs[i].grid()
    axs[i].set_ylabel(METRICS[metric])
axs[0].legend()
axs[-1].set_xlabel("time (s)")
fig.tight_layout()
plt.show()

# %%
# Display images
# --------------

fig, axs = plt.subplots(1, len(OPTIM) + 2, figsize=(20, 7))

for i, optim in enumerate(OPTIM):
    axs[i].imshow(abs(images[optim]), cmap="gray", origin="lower")
    axs[i].axis("off")
    axs[i].set_title(
        f"{optim} reconstruction\n PSNR: {iterations_cb[optim]['psnr'][-1]:.2f}dB \n"
        f"{len(iterations_cb[optim]['time'])} iters ({iterations_cb[optim]['time'][-1]:.2f}s)"
    )

axs[-1].imshow(abs(ground_truth), cmap="gray", origin="lower")
axs[-1].axis("off")
axs[-1].set_title("Original image")
axs[-2].imshow(
    abs(adjoint),
    cmap="gray",
    origin="lower",
)
axs[-2].axis("off")
axs[-2].set_title(
    f"Adjoint NUFFT \n PSNR: {psnr(abs(adjoint), abs(ground_truth), data_range=ground_truth.max()):.2f}dB"
)

fig.suptitle("Reconstructed images using different optimizers")
fig.tight_layout()
plt.show()


# %%
# Using a damping regularization term
# ===================================
#
# The least-square problem can be regularized using a damping term to improve the
# conditioning of the problem.
# This is done by solving the following optimization problem:
#
# .. math::
#    \min_x \|Ax - b\|_2^2 + \gamma \|x\|_2^2
#    where :math:`\gamma` is the regularization parameter.


images = dict()
iterations_cb = dict()
for optim in OPTIM:
    image, iter_cb = nufft.pinv_solver(
        kspace_data=kspace_data_gpu,
        max_iter=1000,
        callback=mixed_cb,
        damp=0.1,
        optim=optim,
    )
    images[optim] = image.get().squeeze()  # retrieve image from GPU.
    iterations_cb[optim] = process_cb_results(iter_cb)


# %%
# Display Convergence
# -------------------


fig, axs = plt.subplots(len(METRICS), 1, sharex=True, figsize=(8, 12))
for i, metric in enumerate(METRICS):
    for optim in OPTIM:
        if "res" in metric:
            axs[i].set_yscale("log")
        axs[i].plot(
            iterations_cb[optim]["time"],
            iterations_cb[optim][metric],
            marker="o",
            markevery=20,
            label=f"{optim} {np.mean(1/np.diff(iterations_cb[optim]['time'])):.2f}iters/s",
        )
    axs[i].grid()
    axs[i].set_ylabel(METRICS[metric])
axs[0].legend()
axs[-1].set_xlabel("time (s)")
fig.tight_layout()
plt.show()

# %%
# Display images
# --------------

fig, axs = plt.subplots(1, len(OPTIM) + 2, figsize=(20, 7))

for i, optim in enumerate(OPTIM):
    axs[i].imshow(abs(images[optim]), cmap="gray", origin="lower")
    axs[i].axis("off")
    axs[i].set_title(
        f"{optim} reconstruction\n PSNR: {iterations_cb[optim]['psnr'][-1]:.2f}dB \n"
        f"{len(iterations_cb[optim]['time'])} iters ({iterations_cb[optim]['time'][-1]:.2f}s)"
    )

axs[-1].imshow(abs(ground_truth), cmap="gray", origin="lower")
axs[-1].axis("off")
axs[-1].set_title("Original image")
axs[-2].imshow(
    abs(adjoint),
    cmap="gray",
    origin="lower",
)
axs[-2].axis("off")
axs[-2].set_title(
    f"Adjoint NUFFT \n PSNR: {psnr(abs(adjoint), abs(ground_truth), data_range=ground_truth.max()):.2f}dB"
)

fig.suptitle("Reconstructed images using different optimizers")
fig.tight_layout()
plt.show()
