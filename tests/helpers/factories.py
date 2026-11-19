"""Useful factories to create matching data for an operator."""

import numpy as np

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

TORCH_CUDA_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()


def image_from_op(operator):
    """Generate a random image."""
    img = np.random.randn(*operator.img_full_shape).astype(operator.cpx_dtype)
    img += 1j * np.random.randn(*img.shape).astype(operator.cpx_dtype)
    return img


def kspace_from_op(operator):
    """Generate a random kspace data."""
    kspace = (1j * np.random.randn(*operator.ksp_full_shape)).astype(operator.cpx_dtype)
    kspace += np.random.randn(*operator.ksp_full_shape).astype(operator.cpx_dtype)
    return kspace


def batched_smaps_from_op(operator):
    """Generate random batched smaps."""
    smaps = 1j * np.random.randn(
        operator.paired_batch, operator.n_coils, *operator.shape
    ).astype(np.complex64)
    smaps += np.random.randn(
        operator.paired_batch, operator.n_coils, *operator.shape
    ).astype(np.complex64)
    smaps /= np.linalg.norm(smaps, axis=1, keepdims=True)
    return smaps


def to_interface(data, interface):
    """Make DATA an array from INTERFACE."""
    if interface == "cupy":
        return cp.array(data)
    elif interface == "torch-cpu":
        return torch.from_numpy(data)
    elif interface == "torch-gpu":
        return torch.from_numpy(data).to("cuda")
    return data


def from_interface(data, interface):
    """Get DATA from INTERFACE as a numpy array."""
    if isinstance(data, np.ndarray):
        return data
    if interface == "cupy":
        return data.get()
    elif "torch" in interface:
        return data.cpu().numpy()
    return data
