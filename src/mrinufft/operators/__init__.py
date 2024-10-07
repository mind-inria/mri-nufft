"""Collection of operators applying the NUFFT used in a MRI context."""

import importlib
import pkgutil
from pathlib import Path

from .base import (
    FourierOperatorBase,
    get_operator,
    list_backends,
    check_backend,
)
from .off_resonance import MRIFourierCorrected, get_interpolators_from_fieldmap
from .stacked import MRIStackedNUFFT
from .subspace import MRISubspace

#
# load all the interfaces modules
for _, name, _ in pkgutil.iter_modules([str(Path(__file__).parent / "interfaces")]):
    if name.startswith("_"):
        continue
    importlib.import_module(".interfaces." + name, __name__)


__all__ = [
    "FourierOperatorBase",
    "MRIFourierCorrected",
    "MRIStackedNUFFT",
    "MRISubspace",
    "check_backend",
    "get_operator",
    "list_backends",
    "get_interpolators_from_fieldmap",
]
