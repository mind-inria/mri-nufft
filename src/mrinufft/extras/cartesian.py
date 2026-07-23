"""Cartesian FFT and iFFT utilities."""

from mrinufft._array_compat import get_array_module


def fft(image, dims, shape=None):
    """Compute n-dimensional FFT along the last ``dims`` axes.

    Parameters
    ----------
    image: NDArray
    dims: int
        Number of dimensions on which the fft is performed, starting from
        last. Required (no default): for an array with a leading
        non-spatial axis (e.g. coils), guessing the wrong `dims` silently
        transforms that axis too.
    shape: tuple[int, ...], optional
        Output shape, by default same as output.

    Returns
    -------
    NDArray:
       Fourier Transform of image.
    """
    axes = range(-dims, 0)
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


def ifft(kspace, dims, shape=None):
    """Compute n-dimensional IFFT along the last ``dims`` axes.

    Parameters
    ----------
    image: NDArray
    dims: int
        Number of dimensions on which the ifft is performed, starting from
        last. Required (no default): for an array with a leading
        non-spatial axis (e.g. coils), guessing the wrong `dims` silently
        transforms that axis too.
    shape: tuple[int, ...], optional
        Output shape, by default same as output.

    Returns
    -------
    NDArray:
       Fourier Transform of image.
    """
    axes = range(-dims, 0)
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
