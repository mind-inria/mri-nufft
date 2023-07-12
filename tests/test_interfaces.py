"""Test the interfaces module."""
import numpy as np
import scipy as sp
import pytest
from pytest_cases import parametrize_with_cases, parametrize

from mrinufft.interfaces.nudft_numpy import RawNDFT


class CasesTrajectories:
    """Trajectories cases we want to test"""

    def case_random2D(self, N=64, pdf="uniform"):
        pass

    def case_random3D(self, N=64, pdf="uniform"):
        pass

    def case_radial2D(self, Nc=10, Ns=10):
        pass

    def case_radial3D(self, Nc=10, Ns=10, Nr=10, expansion="rotation"):
        pass

    def case_grid2D(self, N=64):
        """Create a cartesian grid of frequencies locations."""
        freq_1d = sp.fft.fftfreq(N)
        freq_2d = np.stack(np.meshgrid(freq_1d, freq_1d), axis=-1)
        return freq_2d.reshape(-1, 2)

    def case_grid3D(self, N=64):
        pass


@parametrize_with_cases("kspace_grid, shape", cases=CasesTrajectories, glob="*grid*")
def test_ndft_fft(kspace_grid, shape):
    """Test the raw ndft implementation."""
    ndft_op = RawNDFT(kspace_grid, shape, explicit_matrix=True)

    # Create a random image
    # TODO: use a fixture
    img = np.random.randn(*shape) + 1j * np.random.randn(*shape)

    # Compute the ndft
    kspace = ndft_op.op(img)

    kspace_fft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))

    assert np.allclose(kspace, kspace_fft)
