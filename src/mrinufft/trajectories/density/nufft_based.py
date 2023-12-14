"""Density compensation weights using the NUFFT-based methods."""

from mrinufft import get_operator


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
    nufft_class = get_operator(backend)
    if hasattr(nufft_class, "pipe"):
        return nufft_class.pipe(kspace_traj, grid_size, **kwargs)
    raise ValueError("backend does not have pipe iterations method.")
