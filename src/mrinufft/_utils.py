"""General utility functions for MRI-NUFFT."""

import warnings
from collections import defaultdict
from functools import wraps
import numpy as np
from numpy.typing import DTypeLike


ARRAY_LIBS = {
    "numpy": (np, np.ndarray),
    "cupy": None,
    "torch": None,
    "tensorflow": None,
}
try:
    import cupy

    ARRAY_LIBS["cupy"] = (cupy, cupy.ndarray)
except ImportError:
    pass
try:
    import torch

    ARRAY_LIBS["torch"] = (torch, torch.Tensor)
except ImportError:
    pass

try:
    from tensorflow.experimental import numpy as tnp

    ARRAY_LIBS["tensorflow"] = (tnp, tnp.ndarray)
except ImportError:
    pass


NP2TORCH = {
    np.dtype("float64"): torch.float64,
    np.dtype("float32"): torch.float32,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}


def get_array_module(array):
    """Get the module of the array."""
    for lib, array_type in ARRAY_LIBS.values():
        if lib is not None and isinstance(array, array_type):
            return lib
    raise ValueError("Unknown array library.")


def auto_cast(array, dtype: DTypeLike):
    module = get_array_module(array)
    if module.__name__ == "torch":
        return array.to(NP2TORCH[np.dtype(dtype)])
    else:
        return array.astype(dtype)


def proper_trajectory(trajectory, normalize="pi"):
    """Normalize the trajectory to be used by NUFFT operators.

    Parameters
    ----------
    trajectory: np.ndarray
        The trajectory to normalize, it might be of shape (Nc, Ns, dim) of (Ns, dim)

    normalize: str
        if "pi" trajectory will be rescaled in [-pi, pi], if it was in [-0.5, 0.5]
        if "unit" trajectory will be rescaled in [-0.5, 0.5] if it was not [-0.5, 0.5]

    Returns
    -------
    new_traj: np.ndarray
        The normalized trajectory of shape (Nc * Ns, dim) or (Ns, dim) in -pi, pi
    """
    # flatten to a list of point
    try:
        new_traj = np.asarray(trajectory).copy()
    except Exception as e:
        raise ValueError(
            "trajectory should be array_like, with the last dimension being coordinates"
        ) from e
    new_traj = new_traj.reshape(-1, trajectory.shape[-1])

    if normalize == "pi" and np.max(abs(new_traj)) - 1e-4 < 0.5:
        warnings.warn(
            "Samples will be rescaled to [-pi, pi), assuming they were in [-0.5, 0.5)"
        )
        new_traj *= 2 * np.pi
    elif normalize == "unit" and np.max(abs(new_traj)) - 1e-4 > 0.5:
        warnings.warn(
            "Samples will be rescaled to [-0.5, 0.5), assuming they were in [-pi, pi)"
        )
        new_traj /= 2 * np.pi
    if normalize == "unit" and np.max(new_traj) >= 0.5:
        new_traj = (new_traj + 0.5) % 1 - 0.5
    return new_traj


class MethodRegister:
    """
    A Decorator to register methods of the same type in dictionnaries.

    Parameters
    ----------
    name: str
        The  register
    """

    registry = defaultdict(dict)

    def __init__(self, register_name):
        self.register_name = register_name

    def __call__(self, method_name=None):
        """Register the function in the registry."""

        def decorator(func):
            self.registry[self.register_name][method_name] = func

            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        if callable(method_name):
            func = method_name
            method_name = func.__name__
            return decorator(func)
        else:
            return decorator
