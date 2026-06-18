"""MRI-NUFFT.

MRI-NUFFT provides an easy to use Fourier operator for non-Cartesian
reconstruction.

Doing non-Cartesian MRI has never been so easy.
"""

from . import display, trajectories, operators, density, extras, io

__all__ = [
    "display",
    "trajectories",
    "operators",
    "density",
    "extras",
    "io",
]

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
