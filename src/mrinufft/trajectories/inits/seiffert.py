"""Seiffert spiral 3D trajectory initializations."""

import numpy as np
from numpy.typing import NDArray
from scipy.special import ellipj, ellipk

from mrinufft.trajectories.maths import Ra, Rz
from mrinufft.trajectories.utils import KMAX, initialize_tilt
from mrinufft.trajectories.tools import precess


def initialize_3D_seiffert_spiral(
    Nc: int,
    Ns: int,
    curve_index: float = 0.2,
    nb_revolutions: float = 1,
    axis_tilt: str | float = "golden",
    spiral_tilt: str | float = "golden",
    in_out: bool = False,
) -> NDArray:
    """Initialize 3D trajectories with modulated Seiffert spirals.

    Initially introduced in [SMR18]_, but also proposed later as "Yarnball"
    in [SB21]_ as a nod to [IN95]_. The implementation is based on work
    from [Er00]_ and [Br09]_, using Jacobi elliptic functions rather than
    auxiliary theta functions.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    curve_index : float
        Index controlling curve from 0 (flat) to 1 (curvy), by default 0.3
    nb_revolutions : float
        Number of revolutions, i.e. times the polar angle of the curves
        passes through 0, by default 1
    axis_tilt : str, float, optional
        Angle between shots over a precession around the z-axis, by default "golden"
    spiral_tilt : str, float, optional
        Angle of the spiral within its own axis defined from center to its outermost
        point, by default "golden"
    in_out : bool
        Whether the curves are going in-and-out or start from the center,
        by default False

    Returns
    -------
    NDArray
        3D Seiffert spiral trajectory

    References
    ----------
    .. [IN95] Irarrazabal, Pablo, and Dwight G. Nishimura.
       "Fast three dimensional magnetic resonance imaging."
       Magnetic Resonance in Medicine 33, no. 5 (1995): 656-662.
    .. [Er00] Erdos, Paul.
       "Spiraling the earth with C. G. J. jacobi."
       American Journal of Physics 68, no. 10 (2000): 888-895.
    .. [Br09] Brizard, Alain J.
       "A primer on elliptic functions with applications in classical mechanics."
       European journal of physics 30, no. 4 (2009): 729.
    .. [SMR18] Speidel, Tobias, Patrick Metze, and Volker Rasche.
       "Efficient 3D Low-Discrepancy k-Space Sampling
       Using Highly Adaptable Seiffert Spirals."
       IEEE Transactions on Medical Imaging 38, no. 8 (2018): 1833-1840.
    .. [SB21] Stobbe, Robert W., and Christian Beaulieu.
       "Three-dimensional Yarnball k-space acquisition for accelerated MRI."
       Magnetic Resonance in Medicine 85, no. 4 (2021): 1840-1854.

    """
    # Normalize ellipses integrations by the requested period
    spiral = np.zeros((1, Ns // (1 + in_out), 3))
    period = 4 * ellipk(curve_index**2)
    times = np.linspace(0, nb_revolutions * period, Ns // (1 + in_out), endpoint=False)

    # Initialize first shot
    jacobi = ellipj(times, curve_index**2)
    spiral[0, :, 0] = jacobi[0] * np.cos(curve_index * times)
    spiral[0, :, 1] = jacobi[0] * np.sin(curve_index * times)
    spiral[0, :, 2] = jacobi[1]

    # Make it volumetric instead of just a sphere surface
    magnitudes = np.sqrt(np.linspace(0, 1, Ns // (1 + in_out)))
    spiral = magnitudes.reshape((1, -1, 1)) * spiral

    # Apply precession to the first spiral
    trajectory = precess(
        spiral,
        tilt=axis_tilt,
        nb_rotations=Nc,
        half_sphere=in_out,
        partition="axial",
        axis=None,
    )

    # Tilt the spiral along its own axis
    for i in range(Nc):
        angle = i * initialize_tilt(spiral_tilt)
        rotation = Ra(trajectory[i, -1], angle).T
        trajectory[i] = trajectory[i] @ rotation

    # Handle in_out case
    if in_out:
        first_half_traj = np.copy(trajectory)
        first_half_traj = -first_half_traj[:, ::-1]
        trajectory = np.concatenate([first_half_traj, trajectory], axis=1)
    return KMAX * trajectory


def initialize_3D_seiffert_shells(
    Nc: int,
    Ns: int,
    nb_shells: int,
    curve_index: float = 0.5,
    nb_revolutions: float = 1,
    shell_tilt: str = "uniform",
    shot_tilt: str = "uniform",
) -> NDArray:
    """Initialize 3D trajectories with Seiffert shells.

    The implementation is based on work from [Er00]_ and [Br09]_,
    using Jacobi elliptic functions to define Seiffert spirals
    over shell/sphere surfaces.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    nb_shells : int
        Number of concentric shells/spheres
    curve_index : float
        Index controlling curve from 0 (flat) to 1 (curvy), by default 0.5
    nb_revolutions : float
        Number of revolutions, i.e. times the curve passes through the upper-half
        of the z-axis, by default 1
    shell_tilt : str, float, optional
        Angle between consecutive shells along z-axis, by default "uniform"
    shot_tilt : str, float, optional
        Angle between shots over a shell surface along z-axis, by default "uniform"

    Returns
    -------
    NDArray
        3D Seiffert shell trajectory

    References
    ----------
    .. [IN95] Irarrazabal, Pablo, and Dwight G. Nishimura.
       "Fast three dimensional magnetic resonance imaging."
       Magnetic Resonance in Medicine 33, no. 5 (1995): 656-662.
    .. [Er00] Erdos, Paul.
       "Spiraling the earth with C. G. J. jacobi."
       American Journal of Physics 68, no. 10 (2000): 888-895.
    .. [Br09] Brizard, Alain J.
       "A primer on elliptic functions with applications in classical mechanics."
       European journal of physics 30, no. 4 (2009): 729.

    """
    # Check arguments validity
    if Nc < nb_shells:
        raise ValueError("Argument nb_shells should not be higher than Nc.")
    trajectory = np.zeros((Nc, Ns, 3))

    # Attribute shots to shells following a prescribed density
    Nc_per_shell = np.ones(nb_shells).astype(int)
    density = np.arange(1, nb_shells + 1) ** 2  # simplified version
    for _ in range(Nc - nb_shells):
        idx = np.argmax(density / Nc_per_shell)
        Nc_per_shell[idx] += 1

    # Normalize ellipses integrations by the requested period
    period = 4 * ellipk(curve_index**2)
    times = np.linspace(0, nb_revolutions * period, Ns, endpoint=False)

    # Create shells one by one
    count = 0
    radii = (0.5 + np.arange(nb_shells)) / nb_shells
    for i in range(nb_shells):
        # Prepare shell parameters
        Ms = Nc_per_shell[i]
        k0 = radii[i]

        # Initialize first shot
        jacobi = ellipj(times, curve_index**2)
        trajectory[count, :, 0] = k0 * jacobi[0] * np.cos(curve_index * times)
        trajectory[count, :, 1] = k0 * jacobi[0] * np.sin(curve_index * times)
        trajectory[count, :, 2] = k0 * jacobi[1]

        # Rotate first shot Ms times to create the shell
        rotation = Rz(initialize_tilt(shot_tilt, Ms))
        for j in range(1, Ms):
            trajectory[count + j] = trajectory[count + j - 1] @ rotation

        # Apply shell tilt
        rotation = Rz(i * initialize_tilt(shell_tilt, nb_shells))
        trajectory[count : count + Ms] = trajectory[count : count + Ms] @ rotation
        count += Ms
    return KMAX * trajectory
