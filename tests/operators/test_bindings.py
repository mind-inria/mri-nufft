"""Test for mrinufft interfaces."""

import mrinufft
import pytest


@pytest.mark.parametrize(
    "backend,name",
    [
        pytest.param(backend, name, marks=pytest.mark.backend(backend), id=backend)
        for backend, name in [
            ("finufft", "MRIfinufft"),
            ("mlx-nufft", "MRIMlxNUFFT"),
            ("cufinufft", "MRICufiNUFFT"),
            ("ducc0", "MRIDUCC0"),
            ("tensorflow", "MRITensorflowNUFFT"),
            ("pynufft-cpu", "MRIPynufft"),
            ("pynfft", "MRInfft"),
            ("gpunufft", "MRIGpuNUFFT"),
            ("torchkbnufft-cpu", "TorchKbNUFFTcpu"),
            ("torchkbnufft-gpu", "TorchKbNUFFTgpu"),
        ]
    ]
    + [("numpy", "MRInumpy")],
)
def test_get_operator(backend, name):
    """Test the get_operator function."""
    assert mrinufft.get_operator(backend).__name__ == name


def test_get_operator_fail():
    """Test the get_operator function."""
    with pytest.raises(ValueError):
        mrinufft.get_operator("unknown")
