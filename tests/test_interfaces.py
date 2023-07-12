"""Test the interfaces module."""
import numpy as np
import scipy as sp
import pytest
from pytest_cases import parametrize_with_cases, parametrize

from mrinufft.operators.interfaces.nudft_numpy import RawNDFT


class CasesTrajectories:
    """Trajectories cases we want to test."""

    def case_random2D(self, N=64, pdf="uniform"):
        pass

    def case_random3D(self, N=64, pdf="uniform"):
        pass

    def case_radial2D(self, Nc=10, Ns=10):
        pass

    def case_radial3D(self, Nc=10, Ns=10, Nr=10, expansion="rotation"):
        pass

    def case_grid1D(self, N=256):
        freq_1d = sp.fft.fftfreq(N)
        return freq_1d.reshape(-1, 1), (N,)

    def case_grid2D(self, N=16):
        """Create a cartesian grid of frequencies locations."""
        freq_1d = sp.fft.fftfreq(N)
        freq_2d = np.stack(np.meshgrid(freq_1d, freq_1d), axis=-1)
        return freq_2d.reshape(-1, 2), (N, N)

    def case_grid3D(self, N=16):
        """Create a cartesian grid of frequencies locations."""
        freq_1d = sp.fft.fftfreq(N)
        freq_3d = np.stack(np.meshgrid(freq_1d, freq_1d, freq_1d), axis=-1)
        return freq_3d.reshape(-1, 3), (N, N, N)


@parametrize_with_cases(
    "kspace_grid, shape",
    cases=[CasesTrajectories.case_grid1D, CasesTrajectories.case_grid2D],
)
def test_ndft_matrix(kspace_grid, shape):
    """Test that  the ndft matrix is a good matrix for doing fft."""
    ndft_op = RawNDFT(kspace_grid, shape, explicit_matrix=True)
    # Create a random image
    fft_matrix = [None] * len(shape)
    for i in range(len(shape)):
        fft_matrix[i] = sp.fft.fft(np.eye(shape[i]), norm="ortho")
    fft_mat = fft_matrix[0]
    if len(shape) == 2:
        fft_mat = fft_matrix[0].flatten()[:, None] @ fft_matrix[1].flatten()[None, :]
        fft_mat = (
            fft_mat.reshape(shape * 2)
            .transpose(2, 0, 1, 3)
            .reshape((np.prod(shape),) * 2)
        )
    assert np.allclose(ndft_op._fourier_matrix, fft_mat)


@parametrize_with_cases("kspace_grid, shape", cases=CasesTrajectories, glob="*grid*")
def test_ndft_fft(kspace_grid, shape):
    """Test the raw ndft implementation."""
    ndft_op = RawNDFT(kspace_grid, shape, explicit_matrix=True)
    # Create a random image
    # TODO: use a fixture
    img = np.random.randn(*shape) + 1j * np.random.randn(*shape)

    kspace = ndft_op.op(img).reshape(img.shape)
    if len(shape) >= 2:
        kspace = kspace.swapaxes(0, 1)
    kspace_fft = sp.fft.fftn(img, norm="ortho")

    assert np.allclose(kspace, kspace_fft)
