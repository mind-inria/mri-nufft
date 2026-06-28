"""Collection of operators applying the NUFFT used in a MRI context."""

import importlib
import pkgutil
import pathlib

from .base import (
    FourierOperatorBase,
    get_operator,
    list_backends,
    check_backend,
)
from .off_resonance import MRIFourierCorrected
from .stacked import MRIStackedNUFFT
from .subspace import MRISubspace
from .cartesian import MRICartesianOperator

__all__ = [
    "FourierOperatorBase",
    "MRIFourierCorrected",
    "MRIStackedNUFFT",
    "MRISubspace",
    "MRICartesianOperator",
    "check_backend",
    "get_operator",
    "list_backends",
]
#
# load all the interfaces modules
for _, name, _ in pkgutil.iter_modules(
    [str(pathlib.Path(__file__).parent / "interfaces")]
):
    if name.startswith("_"):
        continue
    importlib.import_module(".interfaces." + name, __name__)


for v in FourierOperatorBase.interfaces.values():
    __all__.append(v[1].__name__)  # add the interface to the __all__ list
    globals()[v[1].__name__] = v[1]  # add the interface to the module namespace

# Autodiff helpers live in `.autodiff`, which imports torch + deepinv at module
# scope. Expose them lazily so importing `operators` (e.g. via `get_operator`)
# does not drag in those heavy deps; they load only on first access.
from mrinufft._array_compat import AUTOGRAD_AVAILABLE, DEEPINV_AVAILABLE

_AUTODIFF_HELPERS = (
    "kspace_as_real",
    "kspace_as_cpx",
    "image_as_real",
    "image_as_cpx",
    "DeepInvPhyNufft",
)

if AUTOGRAD_AVAILABLE and DEEPINV_AVAILABLE:
    __all__ += list(_AUTODIFF_HELPERS)


def __getattr__(name):
    """Lazily resolve the autodiff helpers (PEP 562)."""
    if name in _AUTODIFF_HELPERS:
        from . import autodiff

        return getattr(autodiff, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
