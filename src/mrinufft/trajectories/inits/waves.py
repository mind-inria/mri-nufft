"""2D wave and sinusoide trajectory initializations."""

import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.maths import R2D
from mrinufft.trajectories.utils import KMAX, initialize_tilt


def initialize_2D_sinusoide(
    Nc: int,
    Ns: int,
    tilt: str | float = "uniform",
    in_out: bool = False,
    nb_zigzags: float = 5,
    width: float = 1,
) -> NDArray:
    """Initialize a 2D sinusoide trajectory.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    tilt : str | float, optional
        Tilt of the shots, by default "uniform"
    in_out : bool, optional
        Whether to start from the center or not, by default False
    nb_zigzags : float, optional
        Number of zigzags, by default 5
    width : float, optional
        Width of the sinusoide, by default 1

    Returns
    -------
    NDArray
        2D sinusoide trajectory

    """
    # Initialize a first shot
    segment = np.linspace(-1 if (in_out) else 0, 1, Ns)
    radius = KMAX * segment
    angles = 2 * np.pi * nb_zigzags * segment
    trajectory = np.zeros((Nc, Ns, 2))
    trajectory[0, :, 0] = radius
    trajectory[0, :, 1] = KMAX * np.sin(angles) * width * np.pi / Nc / (1 + in_out)

    # Rotate the first shot Nc times
    rotation = R2D(initialize_tilt(tilt, Nc) / (1 + in_out)).T
    for i in range(1, Nc):
        trajectory[i] = trajectory[i - 1] @ rotation
    return trajectory


def initialize_2D_waves(
    Nc: int, Ns: int, nb_zigzags: float = 5, width: float = 1
) -> NDArray:
    """Initialize a 2D waves trajectory.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    nb_zigzags : float, optional
        Number of zigzags, by default 5
    width : float, optional
        Width of the trajectory, by default 1

    Returns
    -------
    NDArray
        2D waves trajectory
    """
    # Initialize a first shot
    segment = np.linspace(-1, 1, Ns)
    segment = np.sign(segment) * np.abs(segment)
    curl = KMAX * width / Nc * np.cos(nb_zigzags * np.pi * segment)
    line = KMAX * segment

    # Define each shot independently
    trajectory = np.zeros((Nc, Ns, 2))
    delta = 2 * KMAX / (Nc + width)
    for i in range(Nc):
        trajectory[i, :, 0] = line
        trajectory[i, :, 1] = curl + delta * (i + 0.5) - (KMAX - width / Nc / 2)
    return trajectory
