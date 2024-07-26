"""Test for the check_shape function."""

import re
import pytest
import numpy as np
from pytest_cases import parametrize

# from mrinufft.operators.interfaces.utils.utils import check_shape


def check_shape(self_shape, image):
    """Check if the image shape is compatible with the operator's init shape."""
    image_shape_comparaison = image.shape[-len(self_shape) :]
    if image_shape_comparaison != self_shape:
        raise ValueError(
            f"Image shape {image.shape[-len(self_shape):]} is not compatible "
            f"with the operator shape {self_shape}"
        )


@parametrize(
    "self_shape, image_shape",
    [
        ((256, 256), (16, 10, 256, 256)),
        ((256, 256, 176), (16, 10, 256, 256, 176)),
    ],
)
def test_check_shape_pass(self_shape, image_shape):
    """
    Test function for check_shape to ensure it passes with valid shapes.

    Parameters
    ----------
    self_shape (tuple): The expected shape to check against.
    image_shape (tuple): The shape of the image to be generated and tested.

    Raises
    ------
    pytest.fail: If check_shape raises a ValueError unexpectedly.
    """
    image = np.random.rand(*image_shape)
    try:
        check_shape(self_shape, image)
    except ValueError:
        pytest.fail("Unexpected ValueError raised")


@parametrize(
    "self_shape, image_shape",
    [
        ((256, 256), (16, 10, 256, 254)),
        ((256, 256, 176), (16, 10, 256, 256)),
    ],
)
def test_check_shape_fail(self_shape, image_shape):
    """
    Test function for check_shape to ensure it raises a ValueError with invalid shapes.

    Parameters
    ----------
    self_shape (tuple): The expected shape to check against.
    image_shape (tuple): The shape of the image to be generated and tested.

    Raises
    ------
    pytest.raises: If check_shape does not raise a ValueError with the expected message.
    """
    image = np.random.rand(*image_shape)
    expected_message = (
        f"Image shape {image.shape[-len(self_shape):]} is not compatible "
        f"with the operator shape {self_shape}"
    )
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        check_shape(self_shape, image)
