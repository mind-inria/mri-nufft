"""Test cases, as plain functions usable as fixture or test parameters."""

import numpy as np
import scipy as sp

from mrinufft.trajectories import initialize_2D_radial, rotate, stack


def cases_of(cases, *names):
    """Return the ``case_*`` methods of a cases class, in definition order.

    Parameters
    ----------
    cases : type
        Class holding ``case_<name>`` methods, which return the case data.
    names : str
        Only return these cases (``<name>``, without the ``case_`` prefix).
    """
    instance = cases()
    all_names = [n.removeprefix("case_") for n in vars(cases) if n.startswith("case_")]
    if unknown := set(names) - set(all_names):
        raise ValueError(f"Unknown cases {unknown} in {cases.__name__}.")
    return [getattr(instance, f"case_{n}") for n in (names or all_names)]


def case_id(case):
    """Return the test id of a case: its name, without the ``case_`` prefix."""
    return case.__name__.removeprefix("case_")


class CasesTrajectories:
    """Trajectories cases we want to test.

    Each case return a sampling pattern in k-space and the shape of the image.
    """

    def case_random2D(self, M=1000, N=64, seed=0):
        """Create a random 2D trajectory."""
        np.random.seed(seed)
        samples = sp.stats.truncnorm(-3, 3, loc=0, scale=0.16).rvs(size=M * 2)
        samples = samples.reshape(M, 2)
        # Have assymetric image size to better catch shape mismatch issues
        return samples, (N, N * 2)

    def case_random3D(self, M=20000, N=32, seed=0):
        """Create a random 3D trajectory."""
        np.random.seed(seed)
        samples = sp.stats.truncnorm(-3, 3, loc=0, scale=0.16).rvs(size=M * 3)
        samples = samples.reshape(M, 3)
        # Have assymetric image size to better catch shape mismatch issues
        return samples, (N, N * 2, N + 10)

    def case_in_out_radial2D(self, Nc=10, Ns=500, N=64):
        """Create a 2D radial trajectory with in-out sampling."""
        trajectory = initialize_2D_radial(Nc, Ns, in_out=True)
        return trajectory, (N, N)

    def case_radial2D(self, Nc=10, Ns=500, N=64):
        """Create a 2D radial trajectory."""
        trajectory = initialize_2D_radial(Nc, Ns)
        return trajectory, (N, N)

    def case_nyquist_radial2D(self, Nc=32 * 4, Ns=16, N=32):
        """Create a 2D radial trajectory."""
        trajectory = initialize_2D_radial(Nc, Ns)
        return trajectory, (N, N)

    def case_radial3D(self, Nc=10, Ns=500, Nr=10, N=32):
        """Create a 3D radial trajectory."""
        trajectory = initialize_2D_radial(Nc, Ns)
        trajectory = stack(trajectory, nb_stacks=Nr)
        return trajectory, (N, N, N)

    def case_nyquist_radial3D(self, Nc=20 * 4, Ns=10, Nr=20 * 4, N=20):
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


class CasesBatch:
    """Operator configurations.

    Each case returns ``(n_batchs, n_coils, n_trans, sense)``.
    """

    def case_single(self):
        """One image, one coil."""
        return 1, 1, 1, False

    def case_batch(self):
        """Several images, one coil."""
        return 3, 1, 1, False

    def case_sense(self):
        """One image, several coils combined with SENSE."""
        return 1, 4, 1, True

    def case_coils_ntrans(self):
        """One image, several coils, transformed two at a time."""
        return 1, 4, 2, False

    def case_batch_coils(self):
        """Several images, several coils."""
        return 3, 2, 1, False

    def case_batch_sense(self):
        """Several images, several coils combined with SENSE."""
        return 3, 2, 1, True

    def case_batch_sense_ntrans(self):
        """Several images, several coils with SENSE, transformed two at a time."""
        return 3, 4, 2, True

    def case_batch_coils_ntrans(self):
        """Several images, several coils, transformed two at a time."""
        return 3, 4, 2, False
