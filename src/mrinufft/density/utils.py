"""Utilities for density compensation."""

from functools import wraps
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from mrinufft._utils import MethodRegister, proper_trajectory

register_density = MethodRegister("density")
get_density: Callable[[str], Callable[..., NDArray[np.floating]]] = (
    register_density.make_getter()
)


def flat_traj(normalize="unit"):
    """Decorate function to ensure that the trajectory is flatten before calling."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            args = list(args)
            if len(args) == 0:
                # use the first kwargs instead
                first_key = list(kwargs.keys())[0]
                first_arg = kwargs[first_key]
            else:
                first_arg = args[0]
            first_arg = proper_trajectory(first_arg, normalize=normalize)
            if len(args) == 0:
                kwargs[first_key] = first_arg
            else:
                args[0] = first_arg
            return func(*args, **kwargs)

        return wrapper

    if callable(normalize):  # call without argument
        func = normalize
        normalize = "unit"
        return decorator(func)
    else:
        return decorator


def _normalize_weights(weights):
    """Normalize samples weights to reflect their importance.

    Higher weights have lower importance.
    """
    inv_weights = np.sum(weights) / weights
    return inv_weights / (np.sum(inv_weights))
