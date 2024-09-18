"""Example of using the Conjugate Gradient method."""

import numpy as np
import mrinufft
from brainweb_dl import get_mri
from mrinufft.extras.gradient import cg
from mrinufft.density import voronoi
from matplotlib import pyplot as plt
from scipy.datasets import face

samples_loc = mrinufft.initialize_2D_radial(Nc=64, Ns=172)
image = get_mri(sub_id=4)
image = np.flipud(image[90])

NufftOperator = mrinufft.get_operator("gpunufft")  # get the operator
density = voronoi(samples_loc)  # get the density

nufft = NufftOperator(
    samples_loc, shape=image.shape, density=density, n_coils=1
)  # create the NUFFT operator

kspace_data = nufft.op(image)  # get the k-space data
reconstructed_image = cg(nufft, kspace_data)  # reconstruct the image


# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(abs(image), cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(abs(reconstructed_image), cmap="gray")
plt.show()
