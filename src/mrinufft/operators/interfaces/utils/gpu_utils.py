"""Utils for GPU."""

import numpy as np
from pathlib import Path
from hashlib import md5
from functools import wraps
from mrinufft._array_compat import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp


def get_maxThreadBlock():
    """Get the warp size of the current device."""
    if CUPY_AVAILABLE:
        device = cp.cuda.runtime.getDevice()
        return cp.cuda.runtime.getDeviceProperties(device)["maxThreadsPerBlock"]
    raise RuntimeError("Cupy is not available")


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
