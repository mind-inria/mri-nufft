"""Trajectories cases we want to test."""

import numpy as np
import scipy as sp

from mrinufft.trajectories import initialize_2D_radial
from mrinufft.trajectories.tools import stack, rotate


class CasesTrajectories:
    """Trajectories cases we want to test.

    Each case return a sampling pattern in k-space and the shape of the image.
    """

    def case_random2D(self, M=1000, N=64, pdf="uniform", seed=0):
        """Create a random 2D trajectory."""
        np.random.seed(seed)
        samples = sp.stats.truncnorm(-3, 3, loc=0, scale=0.16).rvs(size=M * 2)
        samples = samples.reshape(M, 2)
        # Have assymetric image size to better catch shape mismatch issues
        return samples, (N, N * 2)

    def case_random3D(self, M=200000, N=64, pdf="uniform", seed=0):
        """Create a random 3D trajectory."""
        np.random.seed(seed)
        samples = sp.stats.truncnorm(-3, 3, loc=0, scale=0.16).rvs(size=M * 3)
        samples = samples.reshape(M, 3)
        # Have assymetric image size to better catch shape mismatch issues
        return samples, (N, N * 2, N + 10)

    def case_radial2D(self, Nc=10, Ns=500, N=64):
        """Create a 2D radial trajectory."""
        trajectory = initialize_2D_radial(Nc, Ns)
        return trajectory, (N, N)

    def case_nyquist_radial2D(self, Nc=32 * 4, Ns=16, N=32):
        """Create a 2D radial trajectory."""
        trajectory = initialize_2D_radial(Nc, Ns)
        return trajectory, (N, N)

    def case_radial3D(self, Nc=20, Ns=1000, Nr=20, N=64, expansion="rotations"):
        """Create a 3D radial trajectory."""
        trajectory = initialize_2D_radial(Nc, Ns)
        trajectory = stack(trajectory, nb_stacks=Nr)
        return trajectory, (N, N, N)

    def case_nyquist_radial3D(self, Nc=32 * 4, Ns=16, Nr=32 * 4, N=32):
        """Create a 3D radial trajectory."""
        trajectory = initialize_2D_radial(Nc, Ns)
        trajectory = rotate(trajectory, nb_rotations=Nr)
        return trajectory, (N, N, N)

    def case_nyquist_lowmem_radial3D(self, Nc=2, Ns=16, Nr=2, N=10):
        """Create a 3D radial trajectory with low memory."""
        trajectory = initialize_2D_radial(Nc, Ns)
        trajectory = rotate(trajectory, nb_rotations=Nr)
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
# in the cases collection.
def case_grid1D(N=256):
    """Create a 1D cartesian grid of frequencies locations."""
    freq_1d = sp.fft.fftfreq(N)
    return freq_1d.reshape(-1, 1), (N,)


# multicontrast is only use once, so we don't want to include systematically
# in the cases collection.
def case_multicontrast2D(Nt=48, Nc=10, Ns=500, N=64):
    """Create a 2D radial trajectory."""
    trajectory = initialize_2D_radial(Nc * Nt, Ns, tilt="mri-golden")
    return trajectory.reshape(Nt, Nc, Ns, 2), (N, N)
