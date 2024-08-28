"""Utility functions for GPU Interface."""

import numpy as np


def check_error(ier, message):  # noqa: D103
    if ier != 0:
        raise RuntimeError(message)


def sizeof_fmt(num, suffix="B"):
    """
    Return a number as a XiB format.

    Parameters
    ----------
    num: int
        The number to format
    suffix: str, default "B"
        The unit suffix

    References
    ----------
    https://stackoverflow.com/a/1094933
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def check_size(array_like, shape):
    """Check if array_like has a matching shape."""
    if np.prod(array_like.shape) != np.prod(shape):
        raise ValueError(f"Expected array with {shape}, got {array_like.shape}.")


def check_shape_adj_op(self_, image):
    """Validate that the shape of the provided image matches the expected shape.

    This validation is defined by the operator during initialization.

    Parameters
    ----------
    image : np.ndarray or Tensor

    Returns
    -------
    None
        This function does not return any value. It raises a ValueError if the
        image shape does not match the expected shape.
    """
    image_samples = image.shape[-1]

    if image_samples != self_.n_samples:
        raise ValueError(
            f"Image shape {image_samples} is not compatible "
            f"with the operator shape {self_.n_samples}"
        )
