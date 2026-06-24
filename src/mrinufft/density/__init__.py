"""Density compensation methods.

.. tip::

    Density methods are available through the `get_density` function, that can be used
    by calling ``get_density(<key>)`` or with ``get_density(<key>, *args, **kwargs)``.
    Here are the function available from the registry:

    .. autoregistry:: density
"""

from .geometry_based import cell_count, voronoi
from .nufft_based import pipe
from .utils import flat_traj, get_density, register_density

__all__ = [
    "cell_count",
    "flat_traj",
    "get_density",
    "pipe",
    "register_density",
    "voronoi",
]
