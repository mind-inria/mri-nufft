"""Functions to initialize 3D trajectories."""

from functools import partial
from typing import Literal

import numpy as np
import numpy.linalg as nl
from numpy.typing import NDArray
from scipy.special import ellipj, ellipk

from .maths import (
    CIRCLE_PACKING_DENSITY,
    EIGENVECTOR_2D_FIBONACCI,
    R2D,
    Ra,
    Ry,
    Rz,
    generate_fibonacci_circle,
)
from .tools import conify, duplicate_along_axes, epify, precess, stack
from .trajectory2D import initialize_2D_radial, initialize_2D_spiral
from .utils import KMAX, Packings, Spirals, initialize_shape_norm, initialize_tilt

##############
# 3D RADIALS #
##############


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
       Sonia Nielles‐Vallespin, and Michael O. Zenge.
       "Spiral phyllotaxis: the natural way to construct
       a 3D radial trajectory in MRI."
       Magnetic resonance in medicine 66, no. 4 (2011): 1049-1056.
    """
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
       Jin Mo Goo, and Jang‐Yeon Park.
       "A radial sampling strategy for uniform k‐space coverage
       with retrospective respiratory gating
       in 3D ultrashort‐echo‐time lung imaging."
       NMR in Biomedicine 29, no. 5 (2016): 576-587.
    """
    trajectory = initialize_3D_wong_radial(Nc, Ns, nb_interleaves=1, in_out=in_out)
    trajectory = trajectory.reshape((-1, nb_interleaves, Ns, 3))
    trajectory = np.swapaxes(trajectory, 0, 1)
    trajectory = trajectory.reshape((Nc, Ns, 3))
    return trajectory


############################
# FREEFORM 3D TRAJECTORIES #
############################


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


def initialize_3D_floret(
    Nc: int,
    Ns: int,
    in_out: bool = False,
    nb_revolutions: float = 1,
    spiral: str | float = "fermat",
    cone_tilt: str | float = "golden",
    max_angle: float = np.pi / 2,
    axes: tuple[int, ...] = (2,),
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
       oversampled k‐space trajectories."
       Magnetic resonance in medicine 66, no. 5 (2011): 1303-1311.
    """
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


def initialize_3D_wave_caipi(
    Nc: int,
    Ns: int,
    nb_revolutions: float = 5,
    width: float = 1,
    packing: str = "triangular",
    shape: str | float = "square",
    spacing: tuple[int, int] = (1, 1),
) -> NDArray:
    """Initialize 3D trajectories with Wave-CAIPI.

    This implementation is based on the work from [Bil+15]_.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    nb_revolutions : float, optional
        Number of revolutions, by default 5
    width : float, optional
        Diameter of the helices normalized such that `width=1`
        densely covers the k-space without overlap for square packing,
        by default 1.
    packing : str, optional
        Packing method used to position the helices:
        "triangular"/"hexagonal", "square", "circular"
        or "random"/"uniform", by default "triangular".
    shape : str | float, optional
        Shape over the 2D :math:`k_x-k_y` plane to pack with shots,
        either defined as `str` ("circle", "square", "diamond")
        or as `float` through p-norms following the conventions
        of the `ord` parameter from `numpy.linalg.norm`,
        by default "circle".
    spacing : tuple[int, int]
        Spacing between helices over the 2D :math:`k_x-k_y` plane
        normalized similarly to `width` to correspond to
        helix diameters, by default (1, 1).

    Returns
    -------
    NDArray
        3D wave-CAIPI trajectory

    References
    ----------
    .. [Bil+15] Bilgic, Berkin, Borjan A. Gagoski, Stephen F. Cauley, Audrey P. Fan,
       Jonathan R. Polimeni, P. Ellen Grant, Lawrence L. Wald, and Kawin Setsompop.
       "Wave‐CAIPI for highly accelerated 3D imaging."
       Magnetic resonance in medicine 73, no. 6 (2015): 2152-2162.
    """
    trajectory = np.zeros((Nc, Ns, 3))

    # Initialize first shot
    angles = nb_revolutions * 2 * np.pi * np.arange(0, Ns) / Ns
    trajectory[0, :, 0] = width * np.cos(angles)
    trajectory[0, :, 1] = width * np.sin(angles)
    trajectory[0, :, 2] = np.linspace(-1, 1, Ns)

    # Choose the helix positions according to packing
    packing_enum = Packings[packing]
    side = 2 * int(np.ceil(np.sqrt(Nc))) * np.max(spacing)
    if packing_enum == Packings.RANDOM:
        positions = 2 * side * (np.random.random((side * side, 2)) - 0.5)
    elif packing_enum == Packings.CIRCLE:
        positions = [[0, 0]]
        counter = 0
        while len(positions) < side**2:
            counter += 1
            perimeter = 2 * np.pi * counter
            nb_shots = int(np.trunc(perimeter))
            # Add the full circle
            radius = 2 * counter
            angles = 2 * np.pi * np.arange(nb_shots) / nb_shots
            circle = radius * np.exp(1j * angles)
            positions = np.concatenate(
                [positions, np.array([circle.real, circle.imag]).T], axis=0
            )
    elif packing_enum in [Packings.SQUARE, Packings.TRIANGLE, Packings.HEXAGONAL]:
        # Square packing or initialize hexagonal/triangular packing
        px, py = np.meshgrid(
            np.arange(-side + 1, side, 2), np.arange(-side + 1, side, 2)
        )
        positions = np.stack([px.flatten(), py.flatten()], axis=-1).astype(float)

    if packing_enum in [Packings.HEXAGON, Packings.TRIANGLE]:
        # Hexagonal/triangular packing based on square packing
        positions[::2, 1] += 1 / 2
        positions[1::2, 1] -= 1 / 2
        ratio = nl.norm(np.diff(positions[:2], axis=-1))
        positions[:, 0] /= ratio / 2

    if packing_enum == Packings.FIBONACCI:
        # Estimate helix width based on the k-space 2D surface
        # and an optimal circle packing
        positions = np.sqrt(
            Nc * 2 / CIRCLE_PACKING_DENSITY
        ) * generate_fibonacci_circle(Nc * 2)

    # Remove points by distance to fit both shape and Nc
    main_order = initialize_shape_norm(shape)
    tie_order = 2 if (main_order != 2) else np.inf  # breaking ties
    positions = np.array(positions) * np.array(spacing)
    positions = sorted(positions, key=partial(nl.norm, ord=tie_order))
    positions = sorted(positions, key=partial(nl.norm, ord=main_order))
    positions = positions[:Nc]

    # Shifting copies of the initial shot to all positions
    initial_shot = np.copy(trajectory[0])
    positions = np.concatenate([positions, np.zeros((Nc, 1))], axis=-1)
    for i in range(len(positions)):
        trajectory[i] = initial_shot + positions[i]

    trajectory[..., :2] /= np.max(np.abs(trajectory))
    return KMAX * trajectory


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
    .. [Er00] Erdös, Paul.
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
       "Three‐dimensional Yarnball k‐space acquisition for accelerated MRI."
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


###############################
# SHELL-BASED 3D TRAJECTORIES #
###############################


def initialize_3D_helical_shells(
    Nc: int,
    Ns: int,
    nb_shells: int,
    spiral_reduction: float = 1,
    shell_tilt: str = "intergaps",
    shot_tilt: str = "uniform",
) -> NDArray:
    """Initialize 3D trajectories with helical shells.

    The implementation follows the proposition from [YRB06]_
    but the idea can be traced back much further [IN95]_.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    nb_shells : int
        Number of concentric shells/spheres
    spiral_reduction : float, optional
        Factor used to reduce the automatic spiral length, by default 1
    shell_tilt : str, float, optional
        Angle between consecutive shells along z-axis, by default "intergaps"
    shot_tilt : str, float, optional
        Angle between shots over a shell surface along z-axis, by default "uniform"

    Returns
    -------
    NDArray
        3D helical shell trajectory

    References
    ----------
    .. [YRB06] Shu, Yunhong, Stephen J. Riederer, and Matt A. Bernstein.
       "Three‐dimensional MRI with an undersampled spherical shells trajectory."
       Magnetic Resonance in Medicine 56, no. 3 (2006): 553-562.
    .. [IN95] Irarrazabal, Pablo, and Dwight G. Nishimura.
       "Fast three dimensional magnetic resonance imaging."
       Magnetic Resonance in Medicine 33, no. 5 (1995): 656-662

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

    # Create shells one by one
    count = 0
    radii = (0.5 + np.arange(nb_shells)) / nb_shells
    for i in range(nb_shells):
        # Prepare shell parameters
        Ms = Nc_per_shell[i]
        k0 = radii[i]

        # Initialize first shot cylindrical coordinates
        kz = k0 * np.linspace(-1, 1, Ns)
        magnitudes = k0 * np.sqrt(1 - (kz / k0) ** 2)
        angles = (
            np.sqrt(Ns / spiral_reduction * np.pi / Ms) * np.arcsin(kz / k0)
            + 2 * np.pi * (i + 1) / Ms
        )

        # Format first shot into trajectory
        trajectory[count, :, 0] = magnitudes * np.cos(angles)
        trajectory[count, :, 1] = magnitudes * np.sin(angles)
        trajectory[count : count + Ms, :, 2] = kz[None]

        # Rotate first shot Ms times to create the shell
        rotation = Rz(initialize_tilt(shot_tilt, Ms))
        for j in range(1, Ms):
            trajectory[count + j] = trajectory[count + j - 1] @ rotation

        # Apply shell tilt
        rotation = Rz(i * initialize_tilt(shell_tilt, nb_shells))
        trajectory[count : count + Ms] = trajectory[count : count + Ms] @ rotation
        count += Ms
    return KMAX * trajectory


def initialize_3D_annular_shells(
    Nc: int,
    Ns: int,
    nb_shells: int,
    shell_tilt: float = np.pi,
    ring_tilt: float = np.pi / 2,
) -> NDArray:
    """Initialize 3D trajectories with annular shells.

    An exclusive trajectory inspired from the work proposed in [HM11]_.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    nb_shells : int
        Number of concentric shells/spheres
    shell_tilt : str, float, optional
        Angle between consecutive shells along z-axis, by default pi
    ring_tilt : str, float, optional
        Angle controlling approximately the ring halves rotation, by default pi / 2

    Returns
    -------
    NDArray
        3D annular shell trajectory

    References
    ----------
    .. [HM11] Gerlach, Henryk, and Heiko von der Mosel.
       "On sphere-filling ropes."
       The American Mathematical Monthly 118, no. 10 (2011): 863-876

    """
    # Check arguments validity
    if Nc < nb_shells:
        raise ValueError("Argument nb_shells should not be higher than Nc.")
    trajectory = np.zeros((Nc, Ns, 3))

    # Attribute shots to shells following a prescribed density
    Nc_per_shell = np.ones(nb_shells).astype(int)
    shell_radii = (0.5 + np.arange(nb_shells)) / nb_shells
    density = np.arange(1, nb_shells + 1) ** 2  # simplified version
    for _ in range(Nc - nb_shells):
        idx = np.argmax(density / Nc_per_shell)
        Nc_per_shell[idx] += 1

    # Create shells one by one
    count = 0
    shell_radii = (0.5 + np.arange(nb_shells)) / nb_shells
    for i in range(nb_shells):
        # Prepare shell parameters
        Ms = Nc_per_shell[i]
        k0 = shell_radii[i]
        segment = (0.5 + np.arange(Ms)) / Ms - 0.5
        ring_radii = np.cos(np.pi * segment)
        kz = np.sin(np.pi * segment)

        # Create rings
        shell = np.zeros((Ms, Ns, 3))
        for j in range(Ms):
            radius = ring_radii[j]
            angles = 2 * np.pi * (np.arange(Ns) - Ns / 2) / Ns
            shell[j, :, 0] = k0 * radius * np.cos(angles)
            shell[j, :, 1] = k0 * radius * np.sin(angles)
            shell[j, :, 2] = k0 * kz[j]

        # Rotate rings halves
        rotation = Ry(initialize_tilt("uniform", Ms * 2))
        ring_tilt = initialize_tilt(ring_tilt, Ms)
        power = int(np.around(ring_tilt / np.pi * Ms))
        shell[:, Ns // 2 :] = shell[:, Ns // 2 :] @ nl.matrix_power(rotation, power)

        # Brute force re-ordering and reversing
        shell = shell.reshape((Ms * 2, Ns // 2, 3))
        for j in range(2 * Ms - 1):
            traj_end = shell[j, -1]

            # Select closest half-shot excluding itself
            self_exclusion = np.zeros((2 * Ms, 1))
            self_exclusion[j] = np.inf
            dist_forward = nl.norm(shell[:, 0] - traj_end + self_exclusion, axis=-1)
            dist_backward = nl.norm(shell[:, -1] - traj_end + self_exclusion, axis=-1)

            # Check if closest shot is reversed
            reverse_flag = False
            if np.min(dist_forward) < np.min(dist_backward):
                j_next = np.argmin(dist_forward)
            else:
                j_next = np.argmin(dist_backward)
                reverse_flag = True

            # If closest shot is already known, move on to the next continuous curve
            if j_next <= j:
                continue
            if reverse_flag:
                shell[j_next, :] = shell[j_next, ::-1]

            # Swap shots to place the closest in direct continuity
            shell_inter = np.copy(shell[j + 1])
            shell[j + 1] = shell[j_next]
            shell[j_next] = shell_inter

        # Apply shell tilt
        rotation = Rz(i * initialize_tilt(shell_tilt, nb_shells))
        shell = shell.reshape((Ms, Ns, 3))
        shell = shell @ rotation

        # Reformat shots to trajectory and iterate
        trajectory[count : count + Ms] = shell
        count += Ms

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
    .. [Er00] Erdös, Paul.
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


#####################
# fMRI TRAJECTORIES #
#####################


def initialize_3D_turbine(
    Nc: int,
    Ns_readouts: int,
    Ns_transitions: int,
    nb_blades: int,
    blade_tilt: str = "uniform",
    nb_trains: int | Literal["auto"] = "auto",
    skip_factor: int = 1,
    in_out: bool = True,
) -> NDArray:
    """Initialize 3D TURBINE trajectory.

    This is an implementation of the TURBINE (Trajectory Using Radially
    Batched Internal Navigator Echoes) trajectory proposed in [MGM10]_.
    It consists of EPI-like multi-echo planes rotated around any axis
    (here :math:`k_z`-axis) in a radial fashion.

    Note that our implementation also proposes to segment the planes
    into several shots instead of just one, and includes the proposition
    from [GMC22]_ to also accelerate within the blades by skipping lines
    but while alternating them between blades.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns_readouts : int
        Number of samples per readout
    Ns_transitions : int
        Number of samples per transition between two readouts
    nb_blades : int
        Number of line stacks over the :math:`k_z`-axis axis
    blade_tilt : str, float, optional
        Tilt between individual blades, by default "uniform"
    nb_trains : int, Literal["auto"], optional
        Number of resulting shots, or readout trains, such that each of
        them will be composed of :math:`n` readouts with
        ``Nc = n * nb_trains``. If ``"auto"`` then ``nb_trains`` is set
        to ``nb_blades``.
    skip_factor : int, optional
        Factor defining the way different blades alternate to skip lines,
        forming groups of `skip_factor` non-redundant blades.
    in_out : bool, optional
        Whether the curves are going in-and-out or start from the center

    Returns
    -------
    NDArray
        3D TURBINE trajectory

    References
    ----------
    .. [MGM10] McNab, Jennifer A., Daniel Gallichan, and Karla L. Miller.
       "3D steady‐state diffusion‐weighted imaging with trajectory using
       radially batched internal navigator echoes (TURBINE)."
       Magnetic Resonance in Medicine 63, no. 1 (2010): 235-242.
    .. [GMC22] Graedel, Nadine N., Karla L. Miller, and Mark Chiew.
       "Ultrahigh resolution fMRI at 7T using radial‐cartesian TURBINE sampling."
       Magnetic Resonance in Medicine 88, no. 5 (2022): 2058-2073.
    """
    # Assess arguments validity
    if nb_trains == "auto":
        nb_trains = nb_blades
    if not isinstance(skip_factor, int):
        raise ValueError("`skip_factor` must be an integer.")
    if Nc % nb_blades != 0:
        raise ValueError("`nb_blades` should divide `Nc`.")
    if Nc % nb_trains != 0:
        raise ValueError("`nb_trains` should divide `Nc`.")
    if nb_trains and (Nc % nb_trains != 0):
        raise ValueError("`nb_trains` should divide `Nc`.")
    nb_shot_per_blade = Nc // nb_blades

    # Initialize the first shot of each blade on a plane
    single_plane = initialize_2D_radial(
        nb_blades, Ns_readouts, in_out=in_out, tilt=blade_tilt
    )

    # Stack the blades first shots with tilt to create each full blade
    trajectory = stack(single_plane, nb_shot_per_blade * skip_factor, z_tilt=0)

    # Re-order the shot to be EPI-compatible
    trajectory = trajectory.reshape(
        (nb_shot_per_blade * skip_factor, nb_blades, Ns_readouts, 3)
    )
    trajectory = np.swapaxes(trajectory, 0, 1)
    trajectory = trajectory.reshape((Nc * skip_factor, Ns_readouts, 3))

    # Skip lines but alternate which ones between blades
    skip_mask = (
        np.arange(nb_blades)[:, None]
        + np.arange(nb_shot_per_blade * skip_factor)[None, :]
    )
    skip_mask = ((skip_mask % skip_factor) == 0).astype(bool).flatten()
    trajectory = trajectory[skip_mask]

    # Merge shots into EPI-like multi-readout trains
    if nb_trains and nb_trains != Nc:
        trajectory = epify(
            trajectory,
            Ns_transitions=Ns_transitions,
            nb_trains=nb_trains,
            reverse_odd_shots=True,
        )

    return trajectory


def initialize_3D_repi(
    Nc: int,
    Ns_readouts: int,
    Ns_transitions: int,
    nb_blades: int,
    nb_blade_revolutions: float = 0,
    blade_tilt: str = "uniform",
    nb_spiral_revolutions: float = 0,
    spiral: str = "archimedes",
    nb_trains: int | Literal["auto"] = "auto",
    in_out: bool = True,
) -> NDArray:
    """Initialize 3D REPI trajectory.

    This is an implementation of the REPI (Radial Echo Planar Imaging)
    trajectory proposed in [RMS22]_ and officially inspired
    from TURBINE proposed in [MGM10]_.
    It consists of multi-echo stacks of lines or spirals rotated around any axis
    (here :math:`k_z`-axis) in a radial fashion, but each stack is also slightly
    shifted along the rotation axis in order to be entangled with the others
    without redundancy. This feature is similar to choosing ``skip_factor``
    equal to ``nb_blades`` in TURBINE.

    Note that our implementation also proposes to segment the planes/stacks
    into several shots, instead of just one. Spirals can also be customized
    beyond the classic Archimedean spiral.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns_readouts : int
        Number of samples per readout
    Ns_transitions : int
        Number of samples per transition between two readouts
    nb_blades : int
        Number of line stacks over the kz axis
    nb_blade_revolutions : float
        Number of revolutions over lines/spirals within a blade
        over the kz axis.
    blade_tilt : str, float, optional
        Tilt between individual blades, by default "uniform"
    nb_spiral_revolutions : float, optional
        Number of revolutions of the spirals over the readouts, by default 0
    spiral : str, float, optional
        Spiral type, by default "archimedes"
    nb_trains : int, Literal["auto"], optional
        Number of trains dividing the readouts, such that each
        shot will be composed of `n` readouts with `Nc = n * nb_trains`.
        If "auto" then `nb_trains` is set to `nb_blades`.
    in_out : bool, optional
        Whether the curves are going in-and-out or start from the center

    Returns
    -------
    NDArray
        3D REPI trajectory

    References
    ----------
    .. [RMS22] Rettenmeier, Christoph A., Danilo Maziero, and V. Andrew Stenger.
       "Three dimensional radial echo planar imaging for functional MRI."
       Magnetic Resonance in Medicine 87, no. 1 (2022): 193-206.
    """
    # Assess arguments validity
    if nb_trains == "auto":
        nb_trains = nb_blades
    if Nc % nb_blades != 0:
        raise ValueError("`nb_blades` should divide `Nc`.")
    if Nc % nb_trains != 0:
        raise ValueError("`nb_trains` should divide `Nc`.")
    if nb_trains % nb_blades != 0:
        raise ValueError("`nb_trains` should divide `nb_blades`.")
    nb_shot_per_blade = Nc // nb_blades
    nb_spiral_revolutions = max(nb_spiral_revolutions, 1e-5)

    # Initialize trajectory as a stack of single shots
    single_shot = initialize_2D_spiral(
        1, Ns_readouts, in_out=in_out, tilt=0, nb_revolutions=nb_spiral_revolutions
    )
    shot_tilt = 2 * np.pi * nb_blade_revolutions / Nc
    trajectory = stack(single_shot, Nc, z_tilt=shot_tilt)

    # Rotate some shots to separate the blades
    for i_b in range(nb_blades):
        rotation = R2D(
            i_b * initialize_tilt(blade_tilt, nb_partitions=nb_blades) / (1 + in_out)
        ).T
        trajectory[i_b::nb_blades, ..., :2] = (
            trajectory[i_b::nb_blades, ..., :2] @ rotation
        )

    # Re-order the shot to be EPI-compatible
    trajectory = trajectory.reshape((nb_shot_per_blade, nb_blades, Ns_readouts, 3))
    trajectory = np.swapaxes(trajectory, 0, 1)
    trajectory = trajectory.reshape((Nc, Ns_readouts, 3))

    # Merge shots into EPI-like multi-readout trains
    if nb_trains and nb_trains != Nc:
        trajectory = epify(
            trajectory,
            Ns_transitions=Ns_transitions,
            nb_trains=nb_trains,
            reverse_odd_shots=True,
        )

    return trajectory
