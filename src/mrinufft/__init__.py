"""MRI-NUFFT.

MRI-NUFFT provides an easy to use Fourier operator for non-Cartesian
reconstruction.

Doing non-Cartesian MRI has never been so easy.
"""

from . import display, trajectories, operators, density, extras, io

from mrinufft.operators import get_operator

__all__ = [
    "display",
    "trajectories",
    "operators",
    "density",
    "extras",
    "io",
    "get_operator",
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
