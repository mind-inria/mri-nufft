"""Lissajous and polar Lissajous trajectory initializations."""

import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.maths import R2D, compute_coprime_factors
from mrinufft.trajectories.utils import KMAX, initialize_tilt


def initialize_2D_lissajous(Nc: int, Ns: int, density: float = 1) -> NDArray:
    """Initialize a 2D Lissajous trajectory.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    density : float, optional
        Density of the trajectory, by default 1

    Returns
    -------
    NDArray
        2D Lissajous trajectory
    """
    # Define the whole curve in Cartesian coordinates
    segment = np.linspace(-1, 1, Ns)
    angles = np.pi / 2 * np.sign(segment) * np.abs(segment)

    # Define each shot independenty
    trajectory = np.zeros((Nc, Ns, 2))
    tilt = initialize_tilt("uniform", Nc)
    for i in range(Nc):
        trajectory[i, :, 0] = KMAX * np.sin(angles)
        trajectory[i, :, 1] = KMAX * np.sin(angles * density + i * tilt)
    return trajectory


def initialize_2D_polar_lissajous(
    Nc: int, Ns: int, in_out: bool = False, nb_segments: int = 1, coprime_index: int = 0
) -> NDArray:
    """Initialize a 2D polar Lissajous trajectory.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    in_out : bool, optional
        Whether to start from the center or not, by default False
    nb_segments : int, optional
        Number of segments, by default 1
    coprime_index : int, optional
        Index of the coprime factor, by default 0

    Returns
    -------
    NDArray
        2D polar Lissajous trajectory
    """
    # Adapt the parameters to subcases
    nb_segments = nb_segments * (2 - in_out)
    Nc = Nc // nb_segments

    # Define the whole curve in polar coordinates
    segment = np.pi / 2 * np.linspace(-1, 1, Nc * Ns)
    shift = np.pi * (Nc % 2 - in_out) / 2
    radius = KMAX * np.sin(Nc * segment + shift)
    coprime_factors = compute_coprime_factors(Nc, coprime_index + 1, start=Nc % 2 + 1)
    angles = (
        np.pi
        / (1 + in_out)
        / nb_segments
        * np.sin((Nc - coprime_factors[-1]) * segment)
    )

    # Convert polar to Cartesian coordinates for one segment
    trajectory = np.zeros((Nc * nb_segments, Ns, 2))
    trajectory[:Nc, :, 0] = (radius * np.cos(angles)).reshape((Nc, Ns))
    trajectory[:Nc, :, 1] = (radius * np.sin(angles)).reshape((Nc, Ns))

    # Duplicate and rotate each segment
    rotation = R2D(initialize_tilt("uniform", (1 + in_out) * nb_segments))
    for i in range(Nc, Nc * nb_segments):
        trajectory[i] = trajectory[i - Nc] @ rotation
    return trajectory
