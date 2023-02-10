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
        print(f"unknown backend: '{backend_name}'")
        return False
