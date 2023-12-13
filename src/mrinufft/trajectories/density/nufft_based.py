"""Density compensation weights using the NUFFT-based methods."""

from .cufinufft import pipe_cufinufft
from .tfnufft import pipe_tfnufft
from .gpunufft import pipe_gpunufft


def pipe(kspace_traj, grid_size, backend="cufinufft", **kwargs):
    """Compute the density compensation weights using the pipe method.

    Parameters
    ----------
    kspace_traj: array_like
        array of shape (M, 2) or (M, 3) containing the coordinates of the points.
    grid_size: array_like
        array of shape (2,) or (3,) containing the size of the grid.
    backend: str
        backend to use for the computation. Either "cufinufft" or "tensorflow".
    """
    if backend == "cufinufft":
        return pipe_cufinufft(kspace_traj, grid_size, **kwargs)
    elif backend == "tensorflow":
        return pipe_tfnufft(kspace_traj, grid_size, **kwargs)
    elif backend == "gpunufft":
        return pipe_gpunufft(kspace_traj, grid_size, **kwargs)
    else:
        raise ValueError("backend not supported")
