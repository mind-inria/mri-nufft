"""Kernel function for GPUArray data."""
from .utils.gpu_utils import get_maxThreadBlock, CUPY_AVAILABLE

update_density_kernel = lambda *args, **kwargs: None  # noqa: E731
sense_adj_mono = lambda *args, **kwargs: None  # noqa: E731


if CUPY_AVAILABLE:
    import cupy as cp

    update_density_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void update_density_kernel(float2* density, const float2* update, const unsigned long len)
        {
          unsigned long t = blockDim.x * blockIdx.x + threadIdx.x;
          if(t < len)
            density[t].x *= rsqrtf(update[t].x *update[t].x + update[t].y * update[t].y);
        }
        """,  # noqa: E501
        "update_density_kernel",
    )

    sense_adj_mono_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void sense_adj_mono_kernel(float2* dest, const float2* img, const float2* smap, const unsigned long len)
        {
          unsigned long t = blockDim.x * blockIdx.x + threadIdx.x;
          if (t < len)
          {
            dest[t].x += img[t].x * smap[t].x + img[t].y * smap[t].y;
            dest[t].y += img[t].y * smap[t].x - img[t].x * smap[t].y;
          }
        }
        """,  # noqa: E501
        "sense_adj_mono_kernel",
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
    block_size = get_maxThreadBlock()
    update_density_kernel(
        ((len(density) // block_size) + 1,),
        (block_size,),
        (density, update, len(density)),
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
    block_size = get_maxThreadBlock()
    sense_adj_mono_kernel(
        (dest.size // block_size + 1,),
        (block_size,),
        (dest, coil, smap, dest.size),
        **kwargs,
    )
