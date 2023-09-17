"""Interface for the NUFFT operator of each backend."""

from .base import proper_trajectory, FourierOperatorBase, FOURIER_OPERATORS

__all__ = [
    "FourierOperatorBase",
    "check_backend",
    "get_operator",
    "proper_trajectory",
    "list_backends",
]


def check_backend(backend_name: str):
    """Check if a specific backend is available."""
    try:
        return FOURIER_OPERATORS[backend_name][0]
    except KeyError as e:
        raise ValueError(f"unknown backend: '{backend_name}'") from e


def get_operator(backend_name: str, *args, **kwargs):
    """Return an MRI Fourier operator interface using the correct backend.

    Parameters
    ----------
    backend_name: str
        Backend name
    *args, **kwargs:
        Arguments to pass to the operator constructor.

    Returns
    -------
    FourierOperator
        class or instance of class if args or kwargs are given.

    Raises
    ------
    ValueError if the backend is not available.
    """
    try:
        available, operator = FOURIER_OPERATORS[backend_name]
    except KeyError as exc:
        raise ValueError("backend is not available") from exc
    if not available:
        raise ValueError("backend is registered, but dependencies are not met.")

    if args or kwargs:
        operator = operator(*args, **kwargs)
    return operator


def list_backends(available_only=False):
    """Return a list of backend.

    Parameters
    ----------
    available_only: bool, optional
        If True, only return backends that are available. If False, return all
        backends, regardless of whether they are available or not.
    """
    return [
        name
        for name, (available, _) in FOURIER_OPERATORS.items()
        if available or not available_only
    ]
