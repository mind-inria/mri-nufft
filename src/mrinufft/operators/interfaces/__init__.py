"""Interface for the NUFFT operator of each backend."""

from .tfnufft import MRITensorflowNUFFT, TENSORFLOW_AVAILABLE
from .cufinufft import MRICufiNUFFT, CUFINUFFT_AVAILABLE
from .finufft import MRIfinufft, FINUFFT_AVAILABLE
from .pynufft_cpu import MRIPynufft, PYNUFFT_CPU_AVAILABLE
from .nudft_numpy import MRInumpy

from .base import proper_trajectory

__all__ = [
    "MRICufiNUFFT",
    "MRITensorflowNUFFT",
    "MRIfinufft",
    "MRInumpy",
    "MRIPynufft",
    "check_backend",
    "get_operator",
    "proper_trajectory",
]

_REGISTERED_BACKENDS = {
    "finufft": (FINUFFT_AVAILABLE, MRIfinufft),
    "cufinufft": (CUFINUFFT_AVAILABLE, MRICufiNUFFT),
    "tensorflow": (TENSORFLOW_AVAILABLE, MRITensorflowNUFFT),
    "pynufft-cpu": (PYNUFFT_CPU_AVAILABLE, MRIPynufft),
    "numpy": (True, MRInumpy),
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
