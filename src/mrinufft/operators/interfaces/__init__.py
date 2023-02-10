from .gpu import (
    MRICufiNUFFT,
    MRITensorflowNUFFT,
    CUFINUFFT_AVAILABLE,
    TENSORFLOW_AVAILABLE,
)

from .cpu import MRIfinufft, FINUFFT_AVAILABLE

__all__ = [
    "MRICufiNUFFT",
    "MRITensorflowNUFFT",
    "MRIfinufft",
    "check_backend",
]


def check_backend(backend_name: str):
    """Check if a specific backend is available."""
    if backend_name == "finufft":
        return FINUFFT_AVAILABLE
    elif backend_name == "cufinufft":
        return CUFINUFFT_AVAILABLE
    elif backend_name == "tensorflow":
        return TENSORFLOW_AVAILABLE
    else:
        raise ValueError(f"unknown backend: '{backend_name}'")


def get_operator(backend_name: str):
    """Return an MRI Fourier operator interface using the correct backend,

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

    backend_ops = {
        "cufinufft": MRICufiNUFFT,
        "tensorflow": MRITensorflowNUFFT,
        "finufft": MRIfinufft,
    }
    try:
        return backend_ops[backend_name]
    except KeyError as exc:
        raise ValueError("backend is not available") from exc
