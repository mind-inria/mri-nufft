"""General utility functions for MRI-NUFFT."""

import warnings
from inspect import cleandoc
from collections import defaultdict
from collections.abc import Callable
from functools import wraps

import numpy as np
from numpy.typing import DTypeLike, NDArray

from mrinufft._array_compat import get_array_module


def check_error(ier, message):  # noqa: D103
    if ier != 0:
        raise RuntimeError(message)


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
        if xp.__name__ == "torch":
            new_traj = trajectory.clone()
        else:
            new_traj = np.asarray(trajectory).copy()
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


def _apply_docstring_subs(func: Callable, docstring_subs: dict[str, str]) -> Callable:
    if func.__doc__:
        docstring = cleandoc(func.__doc__)
        for key, sub in docstring_subs.items():
            docstring = docstring.replace(f"${{{key}}}", sub)
        func.__doc__ = docstring
    return func


def _fill_doc(docstring_subs: dict[str, str]) -> Callable:
    """Fill in docstrings with substitutions."""

    @wraps(_fill_doc)
    def wrapper(func: Callable) -> Callable:
        return _apply_docstring_subs(func, docstring_subs)

    return wrapper


class MethodRegister:
    """
    A Decorator to register methods of the same type in dictionnaries.

    Parameters
    ----------
    name: str
        The  register name
    docstring_sub: dict[str,str]
        List of potential subsititutions to apply to the docstring.
    """

    registry = defaultdict(dict)

    def __init__(
        self, register_name: str, docstring_subs: dict[str, str] | None = None
    ):
        self.register_name = register_name
        self.docstring_subs = docstring_subs

    def __call__(self, method_name=None):
        """Register the function in the registry.

        It also substitute placeholder in docstrings.
        """

        def decorator(func):
            self.registry[self.register_name][method_name] = func
            func = _apply_docstring_subs(func, self.docstring_subs or {})
            return func

        if callable(method_name):
            func = method_name
            method_name = func.__name__
            return decorator(func)
        else:
            return decorator

    def make_getter(self) -> Callable:
        """Create a `get_{register_name}` function to get methods from the registry."""

        def getter(method_name, *args, **kwargs):
            try:
                method = self.registry[self.register_name][method_name]
            except KeyError as e:
                raise ValueError(
                    f"Unknown {self.register_name} method {method_name}."
                    " Available methods are \n"
                    f"{list(self.registry[self.register_name].keys())}"
                ) from e

            if args or kwargs:
                return method(*args, **kwargs)
            return method

        getter.__doc__ = f"""Get the {self.register_name} function from its name."""
        getter.__name__ = f"get_{self.register_name}"
        return getter
