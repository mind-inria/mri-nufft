# %%
"""
Using MRI-NUFFT with MRPro
=========================

Wrap an MRI-NUFFT operator as an MRPro linear operator and use its adjoint,
composition, and PyTorch differentiation. Install ``mrpro`` and ``finufft``
to run this example.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mrpro.operators import IdentityOp
from mrinufft import get_operator
from mrinufft.trajectories import initialize_2D_radial

# %%
# MRPro receives full batch and coil dimensions, with a flattened sample axis
# for k-space. Density compensation is disabled to keep the adjoint exact.
shape = (32, 32)
samples = initialize_2D_radial(32, 32).astype(np.float32)
nufft = get_operator("finufft")(samples, shape, density=False)
operator = nufft.make_mrpro()

grid = torch.linspace(-1, 1, shape[0])
yy, xx = torch.meshgrid(grid, grid, indexing="ij")
image = ((xx**2 + yy**2) < 0.5**2).to(torch.complex64)[None, None]

(kspace,) = operator(image)
(adjoint,) = operator.H(kspace)
(gram,) = (operator.H @ operator @ IdentityOp())(image)
torch.testing.assert_close(gram, adjoint)

# %%
# The adapter also supplies gradients with respect to the input image.
estimate = torch.zeros_like(image, requires_grad=True)
loss = (operator(estimate)[0] - kspace).abs().square().sum()
loss.backward()
print(f"Initial loss: {loss.item():.3f}, gradient norm: {estimate.grad.norm():.3f}")

fig, axes = plt.subplots(1, 2)
axes[0].imshow(image[0, 0].abs().numpy(), cmap="gray")
axes[0].set_title("Reference")
axes[1].imshow(adjoint[0, 0].abs().numpy(), cmap="gray")
axes[1].set_title("MRPro adjoint")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
