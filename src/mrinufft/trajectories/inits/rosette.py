"""Rosette trajectory initialization."""

import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.maths import compute_coprime_factors
from mrinufft.trajectories.utils import KMAX, initialize_tilt


def initialize_2D_rosette(
    Nc: int, Ns: int, in_out: bool = False, coprime_index: int = 0
) -> NDArray:
    """Initialize a 2D rosette trajectory.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    in_out : bool, optional
        Whether to start from the center or not, by default False
    coprime_index : int, optional
        Index of the coprime factor, by default 0

    Returns
    -------
    NDArray
        2D rosette trajectory

    """
    # Prepare to parametrize with coprime factor according to Nc parity
    odd = Nc % 2
    coprime = compute_coprime_factors(
        Nc // (2 - odd),
        coprime_index + 1,
        start=1 if odd else (Nc // 2) % 2 + 1,
        update=2,
    )[-1]

    # Define the whole curve in polar coordinates
    angles = np.pi * np.linspace(-1, 1, Nc * Ns) / (1 + odd)
    shift = np.pi * (odd - in_out) / 2
    radius = KMAX * np.sin(Nc / (2 - odd) * angles + shift)

    # Convert polar to Cartesian coordinates
    trajectory = np.zeros((Nc, Ns, 2))
    trajectory[:, :, 0] = (radius * np.cos(angles * coprime)).reshape((Nc, Ns))
    trajectory[:, :, 1] = (radius * np.sin(angles * coprime)).reshape((Nc, Ns))
    return trajectory
