"""Utils for GPU."""

import numpy as np
from pathlib import Path
from hashlib import md5
from functools import wraps

CUPY_AVAILABLE = True
try:
    import cupy as cp
except ImportError:
    CUPY_AVAILABLE = False

TORCH_AVAILABLE = True
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False


def get_maxThreadBlock():
    """Get the warp size of the current device."""
    if CUPY_AVAILABLE:
        device = cp.cuda.runtime.getDevice()
        return cp.cuda.runtime.getDeviceProperties(device)["maxThreadsPerBlock"]
    raise RuntimeError("Cupy is not available")


def is_cuda_array(var):
    """Check if var implement the CUDA Array interface."""
    try:
        return hasattr(var, "__cuda_array_interface__")
    except Exception:
        return False


def is_cuda_tensor(var):
    """Check if var is a CUDA tensor."""
    return TORCH_AVAILABLE and isinstance(var, torch.Tensor) and var.is_cuda


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


# Load CSS4 colors
with open(str(Path(__file__).parent / "css_color.txt")) as f:
    CSS4_COLORS_CODE = f.read().splitlines()[1:]
CSS4_COLORS_CODE = [int(c) for c in CSS4_COLORS_CODE]


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
