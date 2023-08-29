"""Interface for the NUFFT operator of each backend."""

from .tfnufft import MRITensorflowNUFFT, TENSORFLOW_AVAILABLE
from .cufinufft import MRICufiNUFFT, CUFINUFFT_AVAILABLE
from .finufft import MRIfinufft, FINUFFT_AVAILABLE
from .pynufft_cpu import MRIPynufft, PYNUFFT_CPU_AVAILABLE
from .nudft_numpy import MRInumpy
from .nfft import MRInfft, PYNFFT_AVAILABLE
from .pykeops import MRIKeops, PYKEOPS_AVAILABLE

from .base import proper_trajectory

__all__ = [
    "MRICufiNUFFT",
    "MRIKeops",
    "MRIPynufft",
    "MRITensorflowNUFFT",
    "MRIfinufft",
    "MRInumpy",
    "check_backend",
    "get_operator",
    "proper_trajectory",
]

_REGISTERED_BACKENDS = {
    "cufinufft": (CUFINUFFT_AVAILABLE, MRICufiNUFFT),
    "finufft": (FINUFFT_AVAILABLE, MRIfinufft),
    "numpy": (True, MRInumpy),
    "pykeops": (PYKEOPS_AVAILABLE, MRIKeops),
    "pynfft": (PYNFFT_AVAILABLE, MRInfft),
    "pynufft-cpu": (PYNUFFT_CPU_AVAILABLE, MRIPynufft),
    "tensorflow": (TENSORFLOW_AVAILABLE, MRITensorflowNUFFT),
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
