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

try:
    from .autodiff import kspace_as_real, kspace_as_cpx, image_as_real, image_as_cpx

    __all__ += ["kspace_as_real", "kspace_as_cpx", "image_as_real", "image_as_cpx"]
except ImportError:
    pass
