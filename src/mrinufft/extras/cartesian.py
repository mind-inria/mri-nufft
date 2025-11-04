"""Cartesian FFT and iFFT utilities."""

from mrinufft._array_compat import get_array_module


def fft(image, dim=3, shape=None):
    """n-dimensional FFT along the last dim axes."""
    axes = range(-dim, 0)
    xp = get_array_module(image)
    return xp.fft.fftshift(
        xp.fft.fftn(
            xp.fft.ifftshift(image, axes=axes),
            norm="ortho",
            axes=axes,
            s=shape,
        ),
        axes=axes,
    )


def ifft(kspace, dim=3, shape=None):
    """n-dimensional inverse FFT along the last dim axes."""
    axes = range(-dim, 0)
    xp = get_array_module(kspace)
    return xp.fft.fftshift(
        xp.fft.ifftn(
            xp.fft.ifftshift(kspace, axes=axes),
            norm="ortho",
            axes=axes,
            s=shape,
        ),
        axes=axes,
    )
