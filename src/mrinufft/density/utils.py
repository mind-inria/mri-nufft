"""Utilities for density compensation."""

from functools import wraps

import numpy as np

from mrinufft._utils import MethodRegister, proper_trajectory

register_density = MethodRegister("density_compensation")


def get_density(name, *args, **kwargs):
    """Get the density compensation function from its name."""
    try:
        method = register_density.registry["density_compensation"][name]
    except KeyError as e:
        raise ValueError(
            f"Unknown density compensation method {name}. Available methods are \n"
            f"{list(register_density.registry['density_compensation'].keys())}"
        ) from e

    if args or kwargs:
        return method(*args, **kwargs)
    return method


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


def normalize_density(kspace_loc, shape, density, backend, **kwargs):
    """Normalize the density to ensure that the reconstruction is stable."""
    from mrinufft import get_operator
    xp = np
    if backend == "cufinufft":
        import cupy as cp
        xp = cp
    test_op = get_operator(backend)(samples=kspace_loc, shape=shape, **kwargs)
    test_im = xp.ones(shape, dtype=test_op.cpx_dtype)
    test_im_recon = test_op.adj_op(density * test_op.op(test_im))
    density /= xp.mean(xp.abs(test_im_recon))
    return density
