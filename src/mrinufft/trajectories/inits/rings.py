"""Ring trajectory initialization."""

import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.utils import KMAX


def initialize_2D_rings(Nc: int, Ns: int, nb_rings: int) -> NDArray:
    """Initialize a 2D ring trajectory, as proposed in [HHN08]_.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    nb_rings : int
        Number of rings partitioning the k-space.

    Returns
    -------
    NDArray
        2D ring trajectory

    References
    ----------
    .. [HHN08] Wu, Hochong H., Jin Hyung Lee, and Dwight G. Nishimura.
       "MRI using a concentric rings trajectory." Magnetic Resonance
       in Medicine 59, no. 1 (2008): 102-112.

    """
    if Nc < nb_rings:
        raise ValueError("Argument nb_rings should not be higher than Nc.")

    # Choose number of shots per rings
    nb_shots_per_rings = np.ones(nb_rings).astype(int)
    rings_radius = (0.5 + np.arange(nb_rings)) / nb_rings
    for _ in range(nb_rings, Nc):
        longest_shot = np.argmax(rings_radius / nb_shots_per_rings)
        nb_shots_per_rings[longest_shot] += 1

    # Decompose each ring into shots
    trajectory = []
    for rid in range(nb_rings):
        ring = np.zeros(((nb_shots_per_rings[rid]) * Ns, 2))
        angles = np.linspace(0, 2 * np.pi, Ns * nb_shots_per_rings[rid])
        ring[:, 0] = rings_radius[rid] * np.cos(angles)
        ring[:, 1] = rings_radius[rid] * np.sin(angles)
        for i in range(nb_shots_per_rings[rid]):
            trajectory.append(ring[i * Ns : (i + 1) * Ns])
    return KMAX * np.array(trajectory)
