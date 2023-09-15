"""Useful factories to create matching data for an operator."""
import numpy as np


def image_from_op(operator):
    """Generate a random image."""
    if operator.smaps is None:
        img = np.random.randn(operator.n_coils, *operator.shape).astype(
            operator.cpx_dtype
        )
    elif operator.smaps is not None and operator.n_coils > 1:
        img = np.random.randn(*operator.shape).astype(operator.cpx_dtype)

    img += 1j * np.random.randn(*img.shape).astype(operator.cpx_dtype)
    return img


def kspace_from_op(operator):
    """Generate a random kspace data."""
    kspace = (1j * np.random.randn(operator.n_coils, operator.n_samples)).astype(
        operator.cpx_dtype
    )
    kspace += np.random.randn(operator.n_coils, operator.n_samples).astype(
        operator.cpx_dtype
    )
    return kspace
