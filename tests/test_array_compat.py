"""Test array libraries compatibility utils."""

import pytest

from pytest_cases import parametrize, fixture
from unittest.mock import patch, MagicMock

import numpy as np

from mrinufft._array_compat import (
    with_numpy,
    with_numpy_cupy,
    with_torch,
    with_tensorflow,
    _get_device,
    CUPY_AVAILABLE,
    AUTOGRAD_AVAILABLE,
    TENSORFLOW_AVAILABLE,
    tf_cuda_is_available,
)

from helpers import to_interface
from helpers.factories import _param_array_interface

if CUPY_AVAILABLE:
    import cupy as cp

torch_cuda_is_available = False
if AUTOGRAD_AVAILABLE:
    import torch

    torch_cuda_is_available = torch.cuda.is_available()

if TENSORFLOW_AVAILABLE:
    import tensorflow as tf


def dummy_func(*args):
    return args


@fixture(scope="module")
@parametrize("decorator", [with_numpy, with_numpy_cupy, with_torch, with_tensorflow])
def decorator_factory(request, decorator):
    @decorator
    def test_func(*args):
        return dummy_func(*args)

    return test_func, decorator


@_param_array_interface
def test_decorators_outcome(decorator_factory, array_interface):
    decorated_function, _ = decorator_factory

    # Create input array
    array = to_interface(np.asarray([1.0, 2.0, 3.0]), array_interface)

    # Output device and type must be the same as leading argument
    expected_type = type(array)
    expected_device = _get_device(array)

    # Execute function
    outputs = decorated_function(array, array, array)

    # Assert the output has correct type
    for output in outputs:
        assert isinstance(
            output, expected_type
        ), f"Expected {expected_type} but got {type(output)}"
        assert (
            _get_device(output) == expected_device
        ), f"Expected {expected_device} but got {_get_device(output)}"


@_param_array_interface
def test_internal_conversions(decorator_factory, array_interface):
    decorated_function, decorator = decorator_factory

    if decorator.__name__ == "with_tensorflow" and TENSORFLOW_AVAILABLE is False:
        pytest.skip("tensorflow not available")

    # Create input array
    array = to_interface(np.asarray([1.0, 2.0, 3.0]), array_interface)

    # Execute function and monitor internal conversion
    if decorator.__name__ == "with_numpy":
        if array_interface == "cupy":
            with patch("cupy.asnumpy", wraps=cp.asnumpy) as mock_fun:
                _ = decorated_function(array, array, array, 1.0, "a string")
                assert mock_fun.call_count == 3
        if array_interface in ["torch-cpu", "torch-gpu"]:
            array.numpy = MagicMock(wraps=array.numpy)
            _ = decorated_function(array, array, array, 1.0, "a string")
            assert array.numpy.call_count == 3

    elif decorator.__name__ == "with_numpy_cupy":
        if array_interface == "torch-cpu":
            array.numpy = MagicMock(wraps=array.numpy)
            _ = decorated_function(array, array, array, 1.0, "a string")
            assert array.numpy.call_count == 3
        if array_interface == "torch-gpu":
            with patch("cupy.from_dlpack", wraps=cp.from_dlpack) as mock_fun:
                _ = decorated_function(array, array, array, 1.0, "a string")
                assert mock_fun.call_count == 3

    elif decorator.__name__ == "with_torch":
        if array_interface == "numpy":
            with patch("torch.as_tensor", wraps=torch.as_tensor) as mock_fun:
                _ = decorated_function(array, array, array, 1.0, "a string")
                assert mock_fun.call_count == 3
        if array_interface == "cupy" and torch_cuda_is_available:
            with patch("torch.from_dlpack", wraps=torch.from_dlpack) as mock_fun:
                _ = decorated_function(array, array, array, 1.0, "a string")
                assert mock_fun.call_count == 3
        if array_interface == "cupy" and not torch_cuda_is_available:
            with patch("torch.as_tensor", wraps=torch.as_tensor) as mock_fun:
                _ = decorated_function(array, array, array, 1.0, "a string")
                assert mock_fun.call_count == 3

    elif decorator.__name__ == "with_tensorflow":
        if array_interface in ["numpy", "torch-cpu"]:
            with patch(
                "tensorflow.convert_to_tensor", wraps=tf.convert_to_tensor
            ) as mock_fun:
                _ = decorated_function(array, array, array, 1.0, "a string")
                assert mock_fun.call_count == 3
        if array_interface in ["cupy", "torch-gpu"] and tf_cuda_is_available:
            with patch(
                "tensorflow.experimental.dlpack.from_dlpack",
                wraps=tf.experimental.dlpack.from_dlpack,
            ) as mock_fun:
                _ = decorated_function(array, array, array, 1.0, "a string")
                assert mock_fun.call_count == 3
        if array_interface in ["cupy", "torch-gpu"] and not tf_cuda_is_available:
            with patch(
                "tensorflow.convert_to_tensor", wraps=tf.convert_to_tensor
            ) as mock_fun:
                _ = decorated_function(array, array, array, 1.0, "a string")
                assert mock_fun.call_count == 3
