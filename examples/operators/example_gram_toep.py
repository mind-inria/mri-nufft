"""
=====================================
Gram Operator with Toeplitz Embedding
=====================================

This example demonstrates how the auto-adjoint Gram operator of the NUFFT
operator can be efficiently implemented using Toeplitz embedding.

This approach leverages the convolutional structure of the Gram operator to
reduce computational complexity, making it suitable for large-scale imaging
problems.
"""

import matplotlib.pyplot as plt
import numpy as np

from mrinufft import get_operator
from mrinufft.operators.toeplitz import compute_toeplitz_kernel, apply_toeplitz_kernel
from mrinufft import display_2D_trajectory

plt.rcParams["image.cmap"] = "gray"

# %%
# Data preparation
# ================
#
# Image loading
# -------------
#
# For realistic a 2D image we will use the BrainWeb dataset,
# installable using ``pip install brainweb-dl``.

from brainweb_dl import get_mri

mri_data = get_mri(0, "T1")
mri_data = np.flip(mri_data, axis=(0, 1, 2))[90]
mri_data = np.ascontiguousarray(mri_data)
# %%

plt.imshow(mri_data)
plt.axis("off")
plt.title("Groundtruth")
plt.show()

# %%
# Trajectory generation
# ---------------------

from mrinufft import initialize_2D_spiral
from mrinufft.density import voronoi

samples = initialize_2D_spiral(Nc=16, Ns=500, nb_revolutions=10)
density = voronoi(samples)

# %%

display_2D_trajectory(samples)
plt.show()


# %%
# Operator setup
# ==============
nufft = get_operator("finufft")(samples, mri_data.shape, density=density)

# %%
# Naive Gram operator
# -------------------

gram_naive = nufft.adj_op(nufft.op(mri_data))

gram_optim = nufft.gram_op(mri_data)  # will compute the kernel internally once.
# %%

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(np.abs(gram_naive), vmin=0, vmax=np.percentile(np.abs(gram_naive), 99))
plt.title("Gram Naive")
plt.axis("off")
plt.subplot(132)
plt.imshow(np.abs(gram_optim), vmin=0, vmax=np.percentile(np.abs(gram_optim), 99))
plt.title("Gram Optimized")
plt.axis("off")
plt.subplot(133)
plt.imshow((np.abs(gram_naive - gram_optim) / np.abs(gram_naive)) ** 2)
plt.title("Relative Squared Error")
plt.axis("off")
plt.show()
# %%
# Comparing the timings
# ---------------------
from time import perf_counter

naive_times = []
for _ in range(100):
    tic = perf_counter()
    gram_naive = nufft.adj_op(nufft.op(mri_data))
    toc = perf_counter()
    naive_times.append(toc - tic)

setup_times = []
apply_times = []

for _ in range(10):
    tic = perf_counter()
    toeplitz_kernel = nufft.compute_toeplitz_kernel()
    toc = perf_counter()
    setup_times.append(toc - tic)

padded_array = np.zeros_like(toeplitz_kernel)
for _ in range(100):
    tic = perf_counter()
    gram_optim = apply_toeplitz_kernel(
        mri_data, nufft._toeplitz_kernel, padded_array=padded_array
    )
    toc = perf_counter()
    apply_times.append(toc - tic)

print(
    f"Naive Gram time: {np.mean(naive_times)*1e3:.2f} ms ± {np.std(naive_times)*1e3:.2f} ms"
)
print(
    f"Toeplitz setup time: {np.mean(setup_times)*1e3:.2f} ms ± {np.std(setup_times)*1e3:.2f} ms"
)
print(
    f"Toeplitz apply time: {np.mean(apply_times)*1e3:.2f} ms ± {np.std(apply_times)*1e3:.2f} ms"
)


# %%
# Going Further
# ----------------
# The Toeplitz embedding technique only speeds up the application of the Gram operator for the NUFFT steps,
# and the expected speedup depends on the problem size and the trajectory.
# For more details on the implication for the different backends refer to the `Benchmark <https://github.com/mind-inria/mri-nufft-benchmark>`_
