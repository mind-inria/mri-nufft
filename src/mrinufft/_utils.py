"""General utility functions for MRI-NUFFT."""

import warnings
from collections import defaultdict
from functools import wraps
import numpy as np
from numpy.typing import DTypeLike


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


def power_method(max_iter, operator, norm_func=None, x=None):
    """Power method to find the Lipschitz constant of an operator.

    Parameters
    ----------
    max_iter: int
        Maximum number of iterations
    operator: FourierOperatorBase or child class
        NUFFT Operator of which to estimate the lipchitz constant.
    norm_func: callable, optional
        Function to compute the norm , by default np.linalg.norm.
        Change this if you want custom norm, or for computing on GPU.
    x: array_like, optional
        Initial value to use, by default a random numpy array is used.

    Returns
    -------
    float
        The lipschitz constant of the operator.
    """

    def AHA(x):
        return operator.adj_op(operator.op(x))

    if norm_func is None:
        norm_func = np.linalg.norm
    if x is None:
        x = np.random.random(operator.shape).astype(operator.cpx_dtype)
    x_norm = norm_func(x)
    x /= x_norm
    for i in range(max_iter):  # noqa: B007
        x_new = AHA(x)
        x_new_norm = norm_func(x_new)
        x_new /= x_new_norm
        if abs(x_norm - x_new_norm) < 1e-6:
            break
        x_norm = x_new_norm
        x = x_new

    if i == max_iter - 1:
        warnings.warn("Lipschitz constant did not converge")

    if hasattr(x_new_norm, "__cuda_array_interface__"):
        import cupy as cp

        x_new_norm = cp.asarray(x_new_norm).get().item()
    return x_new_norm


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
