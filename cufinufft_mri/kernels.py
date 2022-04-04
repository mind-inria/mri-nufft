"""Kernel function for GPUArray data."""
import os

from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel


def get_grid_size():
    return (1, 1, 1)


def get_block_size():
    return (1, 1, 1)


# Kernels #

update_density_kernel = ElementwiseKernel(
    "pycuda::complex<float> *density, pycuda::complex<float> *update",
    "density[i] /= sqrt(abs(update[i]))",
    "update_density_kernel",
    preamble="#include <pycuda-complex-impl.hpp>",
)


def update_density(density, update):
    """Perform an element wise normalization.

    Parameters
    ----------
    density: GPUArray
    update: GPUArray

    Notes
    -----
    ``density[i] /= sqrt(abs(update[i]))``
    """
    update_density_kernel(density, update)


sense_all_coils_kernel = ElementwiseKernel(
    "pycuda::complex<float> *image, pycuda::complex<float> *smaps, pycuda::complex",
    ""
)

def sense_forward_global_cached(data_d, smaps, inplace=True):

    pass
def sense_forward_coil_cached(data_d, smaps, coil_idx, inplace=False):
    pass
def sense_forward_coil(data, smaps, coil_idx, stream):
    pass
