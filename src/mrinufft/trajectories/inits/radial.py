"""2D and 3D radial trajectory initializations."""

import numpy as np
import numpy.linalg as nl
from numpy.typing import NDArray

from mrinufft.trajectories.maths import R2D, EIGENVECTOR_2D_FIBONACCI
from mrinufft.trajectories.utils import KMAX, initialize_tilt


def initialize_2D_radial(
    Nc: int, Ns: int, tilt: str | float = "uniform", in_out: bool = False
) -> NDArray:
    """Initialize a 2D radial trajectory.

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

    Returns
    -------
    NDArray
        2D radial trajectory
    """
    # Initialize a first shot
    segment = np.linspace(-1 if (in_out) else 0, 1, Ns)
    radius = KMAX * segment
    trajectory = np.zeros((Nc, Ns, 2))
    trajectory[0, :, 0] = radius

    # Rotate the first shot Nc times
    rotation = R2D(initialize_tilt(tilt, Nc) / (1 + in_out)).T
    for i in range(1, Nc):
        trajectory[i] = trajectory[i - 1] @ rotation
    return trajectory


def initialize_3D_phyllotaxis_radial(
    Nc: int, Ns: int, nb_interleaves: int = 1, in_out: bool = False
) -> NDArray:
    """Initialize 3D radial trajectories with phyllotactic structure.

    The radial shots are oriented according to a Fibonacci sphere
    lattice, supposed to reproduce the phyllotaxis found in nature
    through flowers, etc. It ensures an almost uniform distribution.

    This function reproduces the proposition from [Pic+11]_, but the name
    "spiral phyllotaxis" was changed to avoid confusion with
    actual spirals.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    nb_interleaves : int, optional
        Number of implicit interleaves defining the shots order
        for a more uniform k-space distribution over time. When the
        number of interleaves belong to the Fibonacci sequence, the
        shots from one interleave are structured into a continuous
        spiral over the surface the k-space sphere, by default 1
    in_out : bool, optional
        Whether the curves are going in-and-out or start from the center,
        by default False

    Returns
    -------
    NDArray
        3D phyllotaxis radial trajectory

    References
    ----------
    .. [Pic+11] Piccini, Davide, Arne Littmann,
       Sonia Nielles-Vallespin, and Michael O. Zenge.
       "Spiral phyllotaxis: the natural way to construct
       a 3D radial trajectory in MRI."
       Magnetic resonance in medicine 66, no. 4 (2011): 1049-1056.
    """
    from mrinufft.trajectories.inits.cones import initialize_3D_cones
    trajectory = initialize_3D_cones(Nc, Ns, tilt="golden", width=0, in_out=in_out)
    trajectory = trajectory.reshape((-1, nb_interleaves, Ns, 3))
    trajectory = np.swapaxes(trajectory, 0, 1)
    trajectory = trajectory.reshape((Nc, Ns, 3))
    return trajectory


def initialize_3D_golden_means_radial(
    Nc: int, Ns: int, in_out: bool = False
) -> NDArray:
    """Initialize 3D radial trajectories with golden means-based structure.

    The radial shots are oriented using multidimensional golden means,
    which are derived from modified Fibonacci sequences by an eigenvalue
    approach, to provide a temporally stable acquisition with widely
    spread shots at all time.

    This function reproduces the proposition from [Cha+09]_, with
    in addition the option to switch between center-out
    and in-out radial shots.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    in_out : bool, optional
        Whether the curves are going in-and-out or start from the center,
        by default False

    Returns
    -------
    NDArray
        3D golden means radial trajectory

    References
    ----------
    .. [Cha+09] Chan, Rachel W., Elizabeth A. Ramsay,
       Charles H. Cunningham, and Donald B. Plewes.
       "Temporal stability of adaptive 3D radial MRI
       using multidimensional golden means."
       Magnetic Resonance in Medicine 61, no. 2 (2009): 354-363.
    """
    m1 = (EIGENVECTOR_2D_FIBONACCI[0] * np.arange(Nc)) % 1
    m2 = (EIGENVECTOR_2D_FIBONACCI[1] * np.arange(Nc)) % 1

    polar_angle = np.arccos(m1).reshape((-1, 1))
    azimuthal_angle = (2 * np.pi * m2).reshape((-1, 1))

    radius = np.linspace(-1 * in_out, 1, Ns).reshape((1, -1))
    sign = 1 if in_out else (-1) ** np.arange(Nc).reshape((-1, 1))

    trajectory = np.zeros((Nc, Ns, 3))
    trajectory[..., 0] = radius * np.sin(polar_angle) * np.cos(azimuthal_angle)
    trajectory[..., 1] = radius * np.sin(polar_angle) * np.sin(azimuthal_angle)
    trajectory[..., 2] = radius * np.cos(polar_angle) * sign

    return KMAX * trajectory


def initialize_3D_wong_radial(
    Nc: int, Ns: int, nb_interleaves: int = 1, in_out: bool = False
) -> NDArray:
    """Initialize 3D radial trajectories with a spiral structure.

    The radial shots are oriented according to an archimedean spiral
    over a sphere surface, for each interleave.

    This function reproduces the proposition from [WR94]_, with
    in addition the option to switch between center-out
    and in-out radial shots.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    nb_interleaves : int, optional
        Number of implicit interleaves defining the shots order
        for a more structured k-space distribution over time,
        by default 1
    in_out : bool, optional
        Whether the curves are going in-and-out or start from the center,
        by default False

    Returns
    -------
    NDArray
        3D Wong radial trajectory

    References
    ----------
    .. [WR94] Wong, Sam TS, and Mark S. Roos.
       "A strategy for sampling on a sphere applied
       to 3D selective RF pulse design."
       Magnetic Resonance in Medicine 32, no. 6 (1994): 778-784.
    """
    N = Nc // nb_interleaves
    M = nb_interleaves

    points = np.zeros((M, N, 3))
    points[..., 2] = (
        (2 - in_out) * np.arange(1, N + 1) - (1 - in_out) * N - 1
    ).reshape((1, -1)) / N

    planar_radius = np.sqrt(1 - points[..., 2] ** 2)
    azimuthal_angle = np.sqrt(N * np.pi / M) * np.arcsin(points[..., 2])
    azimuthal_angle += 2 * np.pi * np.arange(1, M + 1).reshape((-1, 1)) / M

    points[..., 0] = planar_radius * np.cos(azimuthal_angle)
    points[..., 1] = planar_radius * np.sin(azimuthal_angle)
    points = points.reshape((Nc, 3))

    trajectory = np.linspace(-points * in_out, points, Ns)
    trajectory = np.swapaxes(trajectory, 0, 1)
    trajectory = KMAX * trajectory / np.max(nl.norm(trajectory, axis=-1))
    return trajectory


def initialize_3D_park_radial(
    Nc: int, Ns: int, nb_interleaves: int = 1, in_out: bool = False
) -> NDArray:
    """Initialize 3D radial trajectories with a spiral structure.

    The radial shots are oriented according to an archimedean spiral
    over a sphere surface, shared uniformly between all interleaves.

    This function reproduces the proposition from [Par+16]_,
    itself based on the work from [WR94]_, with
    in addition the option to switch between center-out
    and in-out radial shots.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    nb_interleaves : int, optional
        Number of implicit interleaves defining the shots order
        for a more structured k-space distribution over time,
        by default 1
    in_out : bool, optional
        Whether the curves are going in-and-out or start from the center,
        by default False

    Returns
    -------
    NDArray
        3D Park radial trajectory

    References
    ----------
    .. [Par+16] Park, Jinil, Taehoon Shin, Soon Ho Yoon,
       Jin Mo Goo, and Jang-Yeon Park.
       "A radial sampling strategy for uniform k-space coverage
       with retrospective respiratory gating
       in 3D ultrashort-echo-time lung imaging."
       NMR in Biomedicine 29, no. 5 (2016): 576-587.
    """
    trajectory = initialize_3D_wong_radial(Nc, Ns, nb_interleaves=1, in_out=in_out)
    trajectory = trajectory.reshape((-1, nb_interleaves, Ns, 3))
    trajectory = np.swapaxes(trajectory, 0, 1)
    trajectory = trajectory.reshape((Nc, Ns, 3))
    return trajectory
