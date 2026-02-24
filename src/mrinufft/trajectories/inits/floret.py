"""FLORET 3D trajectory initialization."""

import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.tools import conify, duplicate_along_axes


def initialize_3D_floret(
    Nc: int,
    Ns: int,
    in_out: bool = False,
    nb_revolutions: float = 1,
    spiral: str | float = "fermat",
    cone_tilt: str | float = "golden",
    max_angle: float = np.pi / 2,
    axes: tuple = (2,),
) -> NDArray:
    """Initialize 3D trajectories with FLORET.

    This implementation is based on the work from [Pip+11]_.
    The acronym FLORET stands for Fermat Looped, Orthogonally
    Encoded Trajectories. It consists of Fermat spirals
    folded into 3D cones along one or several axes.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    in_out : bool, optional
        Whether to start from the center or not, by default False
    nb_revolutions : float, optional
        Number of revolutions of the spirals, by default 1
    spiral : str, float, optional
        Spiral type, by default "fermat"
    cone_tilt : str, float, optional
        Tilt of the cones around the :math:`k_z`-axis, by default "golden"
    max_angle : float, optional
        Maximum polar angle starting from the :math:`k_x-k_y` plane,
        by default pi / 2
    axes : tuple, optional
        Axes over which cones are created, by default (2,)

    Returns
    -------
    NDArray
        3D FLORET trajectory

    References
    ----------
    .. [Pip+11] Pipe, James G., Nicholas R. Zwart, Eric A. Aboussouan,
       Ryan K. Robison, Ajit Devaraj, and Kenneth O. Johnson.
       "A new design and rationale for 3D orthogonally
       oversampled k-space trajectories."
       Magnetic resonance in medicine 66, no. 5 (2011): 1303-1311.
    """
    from mrinufft.trajectories.inits.spiral import initialize_2D_spiral

    # Define convenience variables and check argument errors
    Nc_per_axis = Nc // len(axes)
    if Nc % len(axes) != 0:
        raise ValueError("Nc should be divisible by len(axes).")

    # Initialize first spiral
    single_spiral = initialize_2D_spiral(
        Nc=1,
        Ns=Ns,
        spiral=spiral,
        in_out=in_out,
        nb_revolutions=nb_revolutions,
    )

    # Initialize first cone
    cones = conify(
        single_spiral,
        nb_cones=Nc_per_axis,
        z_tilt=cone_tilt,
        in_out=in_out,
        max_angle=max_angle,
    )

    # Duplicate cone along axes
    axes = tuple(2 - ax for ax in axes)  # Default axis is kz, not kx
    trajectory = duplicate_along_axes(cones, axes=axes)
    return trajectory
