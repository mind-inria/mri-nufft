"""Utility functions for GPU Interface."""

from functools import wraps
from hashlib import md5

import numpy as np
from .css_colors import CSS4_COLORS_CODE

CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False


def is_cuda_array(var):
    """Check if var implement the CUDA Array interface."""
    try:
        return hasattr(var, "__cuda_array_interface__")
    except Exception:
        return False


def is_host_array(var):
    """Check if var is a host contiguous np array."""
    try:
        return isinstance(var, np.ndarray) and var.flags.c_contiguous
    except Exception:
        return False


def pin_memory(array):
    """Create a copy of the array in pinned memory."""
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret


def check_error(ier, message):  # noqa: D103
    if ier != 0:
        raise RuntimeError(message)


def nvtx_mark(color=-1):
    """Decorate to annotate function for profiling."""

    def decorator(func):
        # get litteral arg names
        name = func.__name__
        id_col = md5(func.__name__.encode("utf-8")).hexdigest()
        id_col = int(id_col, 16) % len(CSS4_COLORS_CODE)

        @wraps(func)
        def new_func(*args, **kwargs):
            cp.cuda.nvtx.RangePush(name, id_color=CSS4_COLORS_CODE[id_col])
            ret = func(*args, **kwargs)
            cp.cuda.nvtx.RangePop()
            return ret

        return new_func

    return decorator


def sizeof_fmt(num, suffix="B"):
    """
    Return a number as a XiB format.

    Parameters
    ----------
    num: int
        The number to format
    suffix: str, default "B"
        The unit suffix

    References
    ----------
    https://stackoverflow.com/a/1094933
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"
