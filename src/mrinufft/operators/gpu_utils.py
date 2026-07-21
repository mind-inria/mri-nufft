"""Utils for GPU."""

import numpy as np
from pathlib import Path
from hashlib import md5
from functools import wraps
from mrinufft._array_compat import CUPY_AVAILABLE

if CUPY_AVAILABLE:
    import cupy as cp

    _coil_combine_kernel = cp.ElementwiseKernel(
        "raw T data, raw T smaps, int64 b, int32 n_t, int64 vol",
        "raw T img",
        """
        long long off = b * vol + i;
        for (int t = 0; t < n_t; t++) {
            T d = data[t * vol + i];
            T s = smaps[t * vol + i];
            img[off] += d * T(s.real(), -s.imag());
        }
        """,
        "coil_combine_kernel",
    )


# Load CSS4 colors
# List of CSS colors, as int, encoding #AARRGGBB
# fmt: off
CSS4_COLORS_CODE = [
    15792383, 16444375, 65535, 8388564, 15794175, 16119260, 16770244, 0,
    16772045, 255, 9055202, 10824234, 14596231, 6266528, 8388352, 13789470,
    16744272, 6591981, 16775388, 14423100, 65535, 139, 35723, 12092939,
    11119017, 25600, 11119017, 12433259, 9109643, 5597999, 16747520, 10040012,
    9109504, 15308410, 9419919, 4734347, 3100495, 3100495, 52945, 9699539,
    16716947, 49151, 6908265, 6908265, 2003199, 11674146, 16775920, 2263842,
    16711935, 14474460, 16316671, 16766720, 14329120, 8421504, 32768, 11403055,
    8421504, 15794160, 16738740, 13458524, 4915330, 16777200, 15787660,
    15132410, 16773365, 8190976, 16775885, 11393254, 15761536, 14745599,
    16448210, 13882323, 9498256, 13882323, 16758465, 16752762, 2142890, 8900346,
    7833753, 7833753, 11584734, 16777184, 65280, 3329330, 16445670, 16711935,
    8388608, 6737322, 205, 12211667, 9662683, 3978097, 8087790, 64154, 4772300,
    13047173, 1644912, 16121850, 16770273, 16770229, 16768685, 128, 16643558,
    8421376, 7048739, 16753920, 16729344, 14315734, 15657130, 10025880,
    11529966, 14381203, 16773077, 16767673, 13468991, 16761035, 14524637,
    11591910, 8388736, 6697881, 16711680, 12357519, 4286945, 9127187, 16416882,
    16032864, 3050327, 16774638, 10506797, 12632256, 8900331, 6970061, 7372944,
    7372944, 16775930, 65407, 4620980, 13808780, 32896, 14204888, 16737095,
    4251856, 15631086, 16113331, 16777215, 16119285, 16776960, 10145074,
]
# fmt: on
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
