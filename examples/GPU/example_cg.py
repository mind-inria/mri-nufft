"""
======================================
Reconstruction with conjugate gradient
======================================

An example to show how to reconstruct volumes using conjugate gradient method.

This script demonstrates the use of the Conjugate Gradient (CG) method
for solving systems of linear equations of the form Ax = b, where A is a symmetric
positive-definite matrix. The CG method is an iterative algorithm that is particularly
useful for large, sparse systems where direct methods are computationally expensive.

The Conjugate Gradient method is widely used in various scientific and engineering
applications, including solving partial differential equations, optimization problems,
and machine learning tasks.

References
----------
- Inpirations:
        - https://sigpy.readthedocs.io/en/latest/_modules/sigpy/alg.html#ConjugateGradient
        - https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/05_IterativeMethods/05_02_Conjugate_Gradient.html
- Wikipedia:
        - https://en.wikipedia.org/wiki/Conjugate_gradient_method
        - https://en.wikipedia.org/wiki/Momentum
"""

# %%
# .. colab-link::
#    :needs_gpu: 1
#
#    !pip install mri-nufft[gpunufft] scikit-image

# %%
# Imports
import numpy as np
import mrinufft
from brainweb_dl import get_mri
from mrinufft.extras.gradient import cg
from mrinufft.density import voronoi
from matplotlib import pyplot as plt

# %%
# Setup Inputs
samples_loc = mrinufft.initialize_2D_spiral(Nc=64, Ns=256)
image = get_mri(sub_id=4)
image = np.flipud(image[90])

# %%
# Setup the NUFFT operator
NufftOperator = mrinufft.get_operator("gpunufft")  # get the operator
density = voronoi(samples_loc)  # get the density

nufft = NufftOperator(
    samples_loc,
    shape=image.shape,
    density=density,
    n_coils=1,
)  # create the NUFFT operator

# %%
# Reconstruct the image using the CG method
kspace_data = nufft.op(image)  # get the k-space data
reconstructed_image = cg(nufft, kspace_data)  # reconstruct the image

# %%
# Display the results
plt.figure(figsize=(9, 3))
plt.subplot(1, 3, 1)
plt.title("Original image")
plt.imshow(abs(image), cmap="gray")

plt.subplot(1, 3, 2)
plt.title("Conjugate gradient")
plt.imshow(abs(reconstructed_image), cmap="gray")

plt.subplot(1, 3, 3)
plt.title("Adjoint NUFFT")
plt.imshow(abs(nufft.adj_op(kspace_data)), cmap="gray")

plt.show()
