"""
Minimal Example script
======================

This script shows how to use the package to perform a simple NUFFT.
"""

import matplotlib.pyplot as plt
from scipy.datasets import face

import mrinufft
from mrinufft.trajectories import display

# Create a 2D Radial trajectory for demo
samples_loc = mrinufft.initialize_2D_radial(Nc=100, Ns=500)
# Get a 2D image for the demo
image = face(gray=True)[256:768, 256:768]

## The real deal starts here ##
# Choose your NUFFT backend (installed independly from the package)
# And create the associated operator.
NufftOperator = mrinufft.get_operator("finufft")
nufft = NufftOperator(
    samples_loc.reshape(-1, 2), shape=(512, 512), density=True, n_coils=1
)

kspace_data = nufft.op(image)  # Image -> Kspace
image2 = nufft.adj_op(kspace_data)  # Kspace -> Image

# Show the results
fig, ax = plt.subplots(1, 3)

ax[0].imshow(image)
ax[0].set_title("original image")
display.display_2D_trajectory(samples_loc, subfigure=ax[1])
ax[1].set_aspect("equal")
ax[1].set_title("Sampled points in k-space")
ax[2].imshow(abs(image2))
ax[2].set_title("Auto adjoint image")
plt.show()


# %%
# .. note::
# This image is not the same as the original one because the NUFFT operator
# is not a perfect adjoint, and we undersampled by a factor of 5.
# The artefact of reconstruction can be remove by using an iterative reconstruction method.
# Check PySAP-mri documentation for examples.
