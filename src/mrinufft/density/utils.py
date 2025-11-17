"""Utilities for density compensation."""

from functools import wraps

import numpy as np

from mrinufft._utils import MethodRegister, proper_trajectory

register_density = MethodRegister("density")
get_density = register_density.make_getter()


def flat_traj(normalize="unit"):
    """Decorate function to ensure that the trajectory is flatten before calling."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            args = list(args)
            args[0] = proper_trajectory(args[0], normalize=normalize)
            return func(*args, **kwargs)

        return wrapper

    if callable(normalize):  # call without argument
        func = normalize
        normalize = "unit"
        return decorator(func)
    else:
        return decorator


def normalize_weights(weights):
    """Normalize samples weights to reflect their importance.

    Higher weights have lower importance.
    """
    inv_weights = np.sum(weights) / weights
    return inv_weights / (np.sum(inv_weights))
