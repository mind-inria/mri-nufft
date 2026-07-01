"""
Base module.

This module contains the main entry points for the package, including the
submodules and utility functions.
"""

import importlib as _importlib
from typing import TYPE_CHECKING

from mrinufft._utils import proper_trajectory, set_log_level, MethodRegister
from mrinufft._array_compat import get_array_module

submodules = ["display", "trajectories", "operators", "density", "extras", "io"]

__all__ = submodules + [
    "get_operator",
    "set_log_level",
    "proper_trajectory",
    "MethodRegister",
    "get_array_module",
    "__version__",
]


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
