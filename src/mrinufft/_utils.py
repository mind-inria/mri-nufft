"""General utility functions for MRI-NUFFT."""

import warnings
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
import numpy as np
from numpy.typing import DTypeLike, NDArray


ARRAY_LIBS = {
    "numpy": (np, np.ndarray),
    "cupy": (None, None),
    "torch": (None, None),
    "tensorflow": (None, None),
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
    NP2TORCH = {}
else:
    NP2TORCH = {
        np.dtype("float64"): torch.float64,
        np.dtype("float32"): torch.float32,
        np.dtype("complex64"): torch.complex64,
        np.dtype("complex128"): torch.complex128,
    }
try:
    from tensorflow.experimental import numpy as tnp

    ARRAY_LIBS["tensorflow"] = (tnp, tnp.ndarray)
except ImportError:
    pass


def get_array_module(array):
    """Get the module of the array."""
    for lib, array_type in ARRAY_LIBS.values():
        if lib is not None and isinstance(array, array_type):
            return lib
    raise ValueError(f"Unknown array library (={type(array)}.")


def auto_cast(array, dtype: DTypeLike):
    module = get_array_module(array)
    if module.__name__ == "torch":
        return array.to(NP2TORCH[np.dtype(dtype)], copy=False)
    else:
        return array.astype(dtype, copy=False)


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
    xp = get_array_module(trajectory)  # check if the trajectory is a tensor
    try:
        new_traj = (
            trajectory.clone()
            if xp.__name__ == "torch"
            else np.asarray(trajectory).copy()
        )
    except Exception as e:
        raise ValueError(
            "trajectory should be array_like, with the last dimension being coordinates"
        ) from e
    new_traj = new_traj.reshape(-1, trajectory.shape[-1])

    max_abs_val = xp.max(xp.abs(new_traj))

    if normalize == "pi" and max_abs_val - 1e-4 < 0.5:
        warnings.warn(
            "Samples will be rescaled to [-pi, pi), assuming they were in [-0.5, 0.5)"
        )
        new_traj *= 2 * xp.pi
    elif normalize == "unit" and max_abs_val - 1e-4 > 0.5:
        warnings.warn(
            "Samples will be rescaled to [-0.5, 0.5), assuming they were in [-pi, pi)"
        )
        new_traj *= 1 / (2 * xp.pi)

    if normalize == "unit" and max_abs_val >= 0.5:
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

    def make_getter(self) -> Callable:
        def getter(method_name, *args, **kwargs):
            try:
                method = self.registry[self.register_name][method_name]
            except KeyError as e:
                raise ValueError(
                    f"Unknown {self.register_name} method {method_name}. Available methods are \n"
                    f"{list(self.registry[self.register_name].keys())}"
                ) from e

            if args or kwargs:
                return method(*args, **kwargs)
            return method
        getter.__doc__ = f"""Get the {self.register_name} function from its name."""
        getter.__name__ = f"get_{self.register_name}"
        return getter
