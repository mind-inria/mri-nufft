"""Kernel function for GPUArray data."""
from pycuda.elementwise import ElementwiseKernel

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


sense_forward_inplace_kernel = ElementwiseKernel(
    "pycuda::complex<float> *img, pycuda::complex<float> *smap",
    "img[i] = img[i] * smaps[i]",
    preamble="#include <pycuda-complex-impl.hpp")

sense_forward_all_kernel = ElementwiseKernel(
    "pycuda::complex<float> *dest, "
    "pycuda::complex<float> *img, "
    "pycuda::complex<float> *smaps",
    "dest[i] = img[i] * smaps[i]",
    preamble="#include <pycuda-complex-impl.hpp")


def sense_forward(img, smap, dest=None, **kwargs):
    """Perform a multiplication of all smaps with all coils.

    Parameters
    ----------
    img: GPUArray
        The coil image estimation
    smap: GPUArray
        The sensitivity profiles for one coil.
    dest: destination array
        If None, perform the operation in place.
    """
    if dest is None:
        sense_forward_inplace_kernel(img, smap, **kwargs)
    else:
        sense_forward_all_kernel(dest, img, smap, **kwargs)


sense_adj_mono_kernel = ElementwiseKernel(
    "pycuda::complex<float> *dest, "
    "pycuda::complex<float> *coil_img, "
    "pycuda::complex<float> *smap",
    "dest[i] = dest[i] + coil_img[i]*smap[i].real() - coil_img[i]*smap[i].imag()",  # noqa: E501
    "sense_adj_mono_kernel",
    preamble="#include <pycuda-complex-impl.hpp>"
)


def sense_adj_mono(dest, coil, smap, **kwargs):
    """Perform a sense reduction for one coil.

    Parameters
    ----------
    dest: GPUArray
        The image to update with the sense updated data
    coil_img: GPUArray
        The coil image estimation
    smap: GPUArray
        The sensitivity profile of the coil.
    """
    sense_adj_mono_kernel(dest, coil, smap, **kwargs)
