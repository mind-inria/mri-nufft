"""Utilities for interfaces."""

from .utils import (
    check_error,
    sizeof_fmt,
)

from .gpu_utils import (
    get_maxThreadBlock,
    CUPY_AVAILABLE,
    is_host_array,
    is_cuda_array,
    is_cuda_tensor,
    pin_memory,
    nvtx_mark,
)

__all__ = [
    "check_error",
    "sizeof_fmt",
    "get_maxThreadBlock",
    "CUPY_AVAILABLE",
    "is_host_array",
    "is_cuda_array",
    "is_cuda_tensor",
    "pin_memory",
    "nvtx_mark",
]
