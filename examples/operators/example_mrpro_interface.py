"""
======================================
MRPro Interface for mri-nufft
======================================

An example to show how to use the MRPro interface for mri-nufft.

This example demonstrates wrapping an mri-nufft operator as an MRPro
``LinearOperator``, enabling integration with MRPro's reconstruction algorithms.
"""

# Authors: mrinufft contributors
# License: BSD-3-Clause

import numpy as np
from mrinufft import get_operator
from mrinufft.trajectories import initialize_2D_spiral

# Generate a simple 2D spiral trajectory
shape = (32, 32)
n_shots = 2
n_samples_per_shot = 256
samples = initialize_2D_spiral(n_shots, n_samples_per_shot)
# Flatten and normalize to [-0.5, 0.5]
samples = samples.reshape(-1, 2) / (2 * np.pi)

# Create a dummy image
image = np.random.randn(*shape) + 1j * np.random.randn(*shape)

# Generate k-space using mri-nufft
nufft = get_operator("finufft")(samples=samples, shape=shape)
kspace = nufft.op(image)

print(f"MRInufft operator created: {nufft}")
print(f"Backend: {nufft.backend}")
print(f"Shape: {nufft.shape}")
print(f"Number of samples: {nufft.n_samples}")

# Wrap the operator for MRPro
# Note: This requires MRPro to be installed
try:
    from mrinufft.operators.outerfaces import MRProNufftInterface

    mrpro_op = MRProNufftInterface(nufft)
    print(f"\nMRPro LinearOperator created successfully!")
    print(f"Operator type: {type(mrpro_op)}")
    print(f"Input type: {mrpro_op.linop.tin}")
    print(f"Output type: {mrpro_op.linop.tout}")
except ImportError as e:
    print(f"\nMRPro not available: {e}")
    print("Install MRPro to use this interface: pip install mrpro")
