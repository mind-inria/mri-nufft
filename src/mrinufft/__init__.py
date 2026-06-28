"""MRI-NUFFT.

MRI-NUFFT provides an easy to use Fourier operator for non-Cartesian
reconstruction.

Doing non-Cartesian MRI has never been so easy.
"""

import importlib as _importlib
from typing import TYPE_CHECKING

submodules = ["display", "trajectories", "operators", "density", "extras", "io"]

__all__ = submodules + ["get_operator", "__version__"]


def __getattr__(name):
    """Lazily import submodules on first access (PEP 562)."""
    if name in submodules:
        return _importlib.import_module(f"mrinufft.{name}")
    if name == "get_operator":
        return _importlib.import_module("mrinufft.operators").get_operator
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None


def __dir__():
    return __all__


if TYPE_CHECKING:
    # Static visibility for the one re-exported symbol; the submodules above
    # are real packages and resolve without help.
    from mrinufft.operators import get_operator


from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
