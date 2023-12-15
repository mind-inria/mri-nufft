"""Density compensation weights using the NUFFT-based methods."""

from .utils import flat_traj


@flat_traj
def pipe(traj, shape, backend="cufinufft", **kwargs):
    """Compute the density compensation weights using the pipe method.

    Parameters
    ----------
    traj: array_like
        array of shape (M, 2) or (M, 3) containing the coordinates of the points.
    shape: array_like
        array of shape (2,) or (3,) containing the size of the grid.
    backend: str
        backend to use for the computation. Either "cufinufft" or "tensorflow".
    """
    # here to avoid circular import
    from mrinufft.operators.base import get_operator

    nufft_class = get_operator(backend)
    if hasattr(nufft_class, "pipe"):
        return nufft_class.pipe(traj, shape, **kwargs)
    raise ValueError("backend does not have pipe iterations method.")
