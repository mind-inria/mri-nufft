"""Density compensation weights using the NUFFT-based methods."""

from .utils import flat_traj, register_density


@register_density
@flat_traj
def pipe(traj, shape, backend="gpunufft", **kwargs):
    """Compute the density compensation weights using the pipe method.

    Parameters
    ----------
    traj: array_like
        array of shape (M, 2) or (M, 3) containing the coordinates of the points.
    shape: array_like
        array of shape (2,) or (3,) containing the size of the grid.
    backend: str
        backend to use for the computation. Either "cufinufft" or "tensorflow".
    **kwargs:
        Extra argument for the NUFFT operator.
    """
    # here to avoid circular import
    from mrinufft.operators.base import get_operator

    pipe_method = getattr(get_operator(backend), "pipe", None)
    if pipe_method is None:
        raise ValueError("backend does not have pipe iterations method.")
    return pipe_method(traj, shape, **kwargs)
