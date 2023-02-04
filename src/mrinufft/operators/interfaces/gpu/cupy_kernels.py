"""Kernel function for GPUArray data."""
CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False
# Kernels #


update_density_kernel = lambda *args, **kwargs: None  # noqa: E731
sense_adj_mono = lambda *args, **kwargs: None  # noqa: E731

if CUPY_AVAILABLE:
    update_density_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void update_density_kernel(float2* density, const float2* update)
        {
          int t = blockDim.x * blockIdx.x + threadIdx.x;
          density[t].x *= rsqrtf(update[t].x *update[t].x + update[t].y * update[t].y);
        }
        """,
        "update_density_kernel",
    )

    sense_adj_mono_kernel = cp.RawKernel(
        """
        extern "C" __global__
        void sense_adj_mono_kernel(float2* dest, const float2* img, const float2* smap)
        {
          int t = blockDim.x * blockIdx.x + threadIdx.x;
          dest[t].x += img[t].x * smap[t].x + img[t].y * smap[t].y;
          dest[t].y += img[t].y * smap[t].x - img[t].x * smap[t].y;
        }
        """,
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
    update_density_kernel((len(density) // 1024,), (1024,), (density, update))


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
    sense_adj_mono_kernel((dest.size // 1024,), (1024,), (dest, coil, smap), **kwargs)
