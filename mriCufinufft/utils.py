"""Utils functions."""

import pycuda.gpuarray as gp
import pycuda.driver as cuda
import numpy as np

def sizeof_fmt(num, suffix="B"):
    """https://stackoverflow.com/a/1094933 """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"

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


def ensure_on_gpu(data):
    """Ensure the data is on gpu, if not copy it."""
    if is_cuda_array(data):
        return data
    return gp.to_gpu(data)


def extract_col(mat, idx=None):
    """Extract a column from a GPUArray, and return a copy of it."""
    dtype = mat.dtype
    itemsize = np.dtype(dtype).itemsize
    n_rows, n_cols = mat.shape

    if not mat.flags.c_contiguous:
        raise ValueError("Array should be C-Contiguous")
    assert n_cols > idx >= 0

    new_mat = gp.empty(n_rows, dtype)

    copy = cuda.Memcpy2D()   # pylint: disable=E1101 # noqa: E1101
    copy.set_src_device(mat.gpudata)

    # Offset of the  column in bytes
    copy.src_x_in_bytes = idx * itemsize
    copy.set_dst_device(new_mat.gpudata)
    # Width of a row in bytes in the source array
    copy.src_pitch = n_cols * itemsize
    # Width of sliced row
    copy.dst_pitch = copy.width_in_bytes = itemsize
    copy.height = n_rows
    copy(aligned=True)

    return new_mat


def extract_cols(mat, start=0, stop=None):
    """Extract columns from a GPUArray, and return a copy of it."""
    dtype = mat.dtype
    itemsize = np.dtype(dtype).itemsize
    n_rows, n_cols = mat.shape
    len_cols = stop - start

    assert mat.flags.c_contiguous
    assert n_cols >= stop > start >= 0

    new_mat = gp.empty((n_rows, len_cols), dtype)

    copy = cuda.Memcpy2D()  # pylint: disable=E1101 # noqa: E1101
    copy.set_src_device(mat.gpudata)

    # Offset of the first column in bytes
    copy.src_x_in_bytes = start * itemsize
    copy.set_dst_device(new_mat.gpudata)
    # Width of a row in bytes in the source array
    copy.src_pitch = n_cols * itemsize
    # Width of sliced row
    copy.dst_pitch = copy.width_in_bytes = len_cols * itemsize
    copy.height = n_rows
    copy(aligned=True)

    return new_mat


def get_best_grid_bloc():
    """Get the best grid and bloc dimension."""
    pass
