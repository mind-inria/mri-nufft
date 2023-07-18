"""Trajectories cases we want to test."""
import numpy as np
import scipy as sp

from mrinufft.trajectories import initialize_2D_radial, initialize_3D_from_2D_expansion


class CasesTrajectories:
    """Trajectories cases we want to test.

    Each case return a sampling pattern in k-space and the shape of the image.
    """

    def case_random2D(self, M=1000, N=64, pdf="uniform", seed=0):
        """Create a random 2D trajectory."""
        np.random.seed(seed)
        samples = np.random.rand(M, 2) - 0.5
        samples /= samples.max()
        samples -= 0.5
        return samples, (N, N)

    def case_random3D(self, M=200000, N=64, pdf="uniform", seed=0):
        """Create a random 3D trajectory."""
        np.random.seed(seed)
        samples = np.random.randn(M, 3)
        samples /= samples.max()
        samples -= 0.5
        return samples, (N, N, N)

    def case_radial2D(self, Nc=10, Ns=500, N=64):
        """Create a 2D radial trajectory."""
        trajectory = initialize_2D_radial(Nc, Ns)
        return trajectory, (N, N)

    def case_radial3D(self, Nc=20, Ns=1000, Nr=20, N=64, expansion="rotations"):
        """Create a 3D radial trajectory."""
        trajectory = initialize_3D_from_2D_expansion("radial", expansion, Nc, Ns, Nr)
        return trajectory, (N, N, N)

    def case_grid2D(self, N=16):
        """Create a 2D cartesian grid of frequencies locations."""
        freq_1d = sp.fft.fftfreq(N)
        freq_2d = np.stack(np.meshgrid(freq_1d, freq_1d), axis=-1)
        return freq_2d.reshape(-1, 2), (N, N)

    def case_grid3D(self, N=16):
        """Create a 3D cartesian grid of frequencies locations."""
        freq_1d = sp.fft.fftfreq(N)
        freq_3d = np.stack(np.meshgrid(freq_1d, freq_1d, freq_1d), axis=-1)
        return freq_3d.reshape(-1, 3), (N, N, N)


# 1D grid is only use once, so we don't want to include systematically
#  fsin the cases collection.
def case_grid1D(N=256):
    """Create a 1D cartesian grid of frequencies locations."""
    freq_1d = sp.fft.fftfreq(N)
    return freq_1d.reshape(-1, 1), (N,)
