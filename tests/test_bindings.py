"""Test for mrinufft interfaces."""


import mrinufft
import pytest


@pytest.mark.parametrize(
    "backend,name",
    [
        ("finufft", "MRIfinufft"),
        ("cufinufft", "MRICufiNUFFT"),
        ("tensorflow", "MRITensorflowNUFFT"),
        ("pynufft-cpu", "MRIPynufft"),
        ("pynfft", "MRInfft"),
        ("numpy", "MRInumpy"),
        ("gpunufft", "MRIGpuNUFFT"),
    ],
)
def test_get_operator(backend, name):
    """Test the get_operator function."""
    assert mrinufft.get_operator(backend).__name__ == name


def test_get_operator_fail():
    """Test the get_operator function."""
    with pytest.raises(ValueError):
        mrinufft.get_operator("unknown")
