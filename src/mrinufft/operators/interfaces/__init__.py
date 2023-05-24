"""Interface for the NUFFT operator of each backend."""
from .gpu import (
    MRICufiNUFFT,
    MRITensorflowNUFFT,
    CUFINUFFT_AVAILABLE,
    TENSORFLOW_AVAILABLE,
)

from .cpu import MRIfinufft, FINUFFT_AVAILABLE, MRIPynufft, PYNUFFT_CPU_AVAILABLE

__all__ = [
    "MRICufiNUFFT",
    "MRITensorflowNUFFT",
    "MRIfinufft",
    "MRIPynufft",
    "check_backend",
    "get_operator",
]

_REGISTERED_BACKENDS = {
    "finufft": (FINUFFT_AVAILABLE, MRIfinufft),
    "cufinufft": (CUFINUFFT_AVAILABLE, MRICufiNUFFT),
    "tensorflow": (TENSORFLOW_AVAILABLE, MRITensorflowNUFFT),
    "pynufft-cpu": (PYNUFFT_CPU_AVAILABLE, MRIPynufft),
}


def check_backend(backend_name: str):
    """Check if a specific backend is available."""
    try:
        return _REGISTERED_BACKENDS[backend_name][0]
    except KeyError as e:
        raise ValueError(f"unknown backend: '{backend_name}'") from e


def get_operator(backend_name: str):
    """Return an MRI Fourier operator interface using the correct backend.

    Parameters
    ----------
    backend_name: str
        Backend name,

    Returns
    -------
    FourierOperatorBase Interface

    Raises
    ------
    ValueError if the backend is not available.
    """
    try:
        available, operator = _REGISTERED_BACKENDS[backend_name]
    except KeyError as exc:
        raise ValueError("backend is not available") from exc
    if not available:
        raise ValueError("backend is registered, but dependencies are not met.")
    return operator
