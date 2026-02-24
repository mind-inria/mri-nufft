"""PROPELLER trajectory initialization."""

import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.tools import rotate
from mrinufft.trajectories.utils import KMAX


def initialize_2D_propeller(Nc: int, Ns: int, nb_strips: int) -> NDArray:
    """Initialize a 2D PROPELLER trajectory, as proposed in [Pip99]_.

    The PROPELLER trajectory is generally used along a specific
    reconstruction pipeline described in [Pip99]_ to correct for
    motion artifacts.

    The acronym PROPELLER stands for Periodically Rotated
    Overlapping ParallEL Lines with Enhanced Reconstruction,
    and the method is also commonly known under other aliases
    depending on the vendor, with some variations: BLADE,
    MulitVane, RADAR, JET.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    nb_strips : int
        Number of rotated strips, must divide ``Nc``

    References
    ----------
    .. [Pip99] Pipe, James G. "Motion correction with PROPELLER MRI:
       application to head motion and free-breathing cardiac imaging."
       Magnetic Resonance in Medicine 42, no. 5 (1999): 963-969.
    """
    # Check for value errors
    if Nc % nb_strips != 0:
        raise ValueError("Nc should be divisible by nb_strips.")

    # Initialize single shot
    Nc_per_strip = Nc // nb_strips
    trajectory = np.linspace(-1, 1, Ns).reshape((1, Ns, 1))

    # Convert single shot to single strip
    trajectory = np.tile(trajectory, reps=(Nc_per_strip, 1, 2))
    y_axes = np.pi / 2 / nb_strips * np.linspace(-1, 1, Nc_per_strip)
    trajectory[:, :, 1] = y_axes[:, None]

    # Rotate single strip into multiple strips
    trajectory = rotate(trajectory, nb_rotations=nb_strips, z_tilt=np.pi / nb_strips)
    trajectory = trajectory[..., :2]  # Remove dim added by rotate

    return KMAX * trajectory
