"""Collection of operators applying the NUFFT used in a MRI context."""
from .interfaces import (
    FourierOperatorBase,
    FOURIER_OPERATORS,
    get_operator,
    check_backend,
    list_backends,
)

from .off_resonnance import MRIFourierCorrected

__all__ = [
    "get_operator",
    "check_backend",
    "FourierOperatorBase",
    "FOURIER_OPERATORS",
    "list_backends",
    "MRIFourierCorrected",
]
