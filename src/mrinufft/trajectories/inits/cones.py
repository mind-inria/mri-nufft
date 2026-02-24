"""2D and 3D cone trajectory initializations."""

import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.maths import R2D, CIRCLE_PACKING_DENSITY
from mrinufft.trajectories.utils import KMAX, initialize_tilt
from mrinufft.trajectories.tools import conify, precess


def initialize_2D_cones(
    Nc: int,
    Ns: int,
    tilt: str | float = "uniform",
    in_out: bool = False,
    nb_zigzags: float = 5,
    width: float = 1,
) -> NDArray:
    """Initialize a 2D cone trajectory.

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
        Width of the cone, by default 1

    Returns
    -------
    NDArray
        2D cone trajectory

    """
    # Initialize a first shot
    segment = np.linspace(-1 if (in_out) else 0, 1, Ns)
    radius = KMAX * segment
    angles = 2 * np.pi * nb_zigzags * np.abs(segment)
    trajectory = np.zeros((Nc, Ns, 2))
    trajectory[0, :, 0] = radius
    trajectory[0, :, 1] = radius * np.sin(angles) * width * np.pi / Nc / (1 + in_out)

    # Rotate the first shot Nc times
    rotation = R2D(initialize_tilt(tilt, Nc) / (1 + in_out)).T
    for i in range(1, Nc):
        trajectory[i] = trajectory[i - 1] @ rotation
    return trajectory


def initialize_3D_cones(
    Nc: int,
    Ns: int,
    tilt: str | float = "golden",
    in_out: bool = False,
    nb_zigzags: float = 5,
    spiral: str | float = "archimedes",
    width: float = 1,
) -> NDArray:
    """Initialize 3D trajectories with cones.

    Initialize a trajectory consisting of 3D cones duplicated
    in each direction and almost evenly distributed using a Fibonacci
    lattice spherical projection when the tilt is set to "golden".

    The cone width is automatically determined based on the optimal
    circle packing of a sphere surface, as discussed in [CK90]_.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    tilt : str, float, optional
        Tilt of the cones, by default "golden"
    in_out : bool, optional
        Whether the curves are going in-and-out or start from the center,
        by default False
    nb_zigzags : float, optional
        Number of zigzags of the cones, by default 5
    spiral : str, float, optional
        Spiral type, by default "archimedes"
    width : float, optional
        Cone width normalized such that `width=1` avoids cone overlaps, by default 1

    Returns
    -------
    NDArray
        3D cones trajectory

    References
    ----------
    .. [CK90] Clare, B. W., and D. L. Kepert.
       "The optimal packing of circles on a sphere."
       Journal of mathematical chemistry 6, no. 1 (1991): 325-349.
    """
    from mrinufft.trajectories.inits.spiral import initialize_2D_spiral

    # Initialize first spiral
    single_spiral = initialize_2D_spiral(
        Nc=1,
        Ns=Ns,
        spiral=spiral,
        in_out=in_out,
        nb_revolutions=nb_zigzags,
    )

    # Estimate best cone angle based on the ratio between
    # sphere volume divided by Nc and spherical sector packing optimaly a sphere
    max_angle = np.pi / 2 - width * np.arccos(
        1 - CIRCLE_PACKING_DENSITY * 2 / Nc / (1 + in_out)
    )

    # Initialize first cone
    ## Create three cones for proper partitioning, but only one is needed
    cones = conify(
        single_spiral,
        nb_cones=3,
        z_tilt=None,
        in_out=in_out,
        max_angle=max_angle,
        borderless=False,
    )[-1:]

    # Apply precession to the first cone
    trajectory = precess(
        cones,
        tilt=tilt,
        nb_rotations=Nc,
        half_sphere=in_out,
        partition="axial",
        axis=2,
    )

    return trajectory
