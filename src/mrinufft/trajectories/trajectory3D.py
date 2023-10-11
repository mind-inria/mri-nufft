"""3D Trajectory initialization functions."""
import numpy as np
import numpy.linalg as nl

from functools import partial
from scipy.special import ellipj, ellipk

from .expansions import (
    stack_2D_to_3D_expansion,
    rotate_2D_to_3D_expansion,
    cone_2D_to_3D_expansion,
    helix_2D_to_3D_expansion,
)
from .trajectory2D import (
    initialize_2D_radial,
    initialize_2D_spiral,
    initialize_2D_rosette,
    initialize_2D_cones,
)
from .utils import Ry, Rz, Rv, initialize_tilt, initialize_shape_norm, KMAX


############################
# FREEFORM 3D TRAJECTORIES #
############################


def initialize_3D_from_2D_expansion(
    basis, expansion, Nc, Ns, nb_repetitions, basis_kwargs=None, expansion_kwargs=None
):
    """Initialize 3D trajectories from 2D trajectories.

    Parameters
    ----------
    basis : str or array_like
        2D trajectory basis
    expansion : str
        3D trajectory expansion
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    nb_repetitions : int
        Number of repetitions of the 2D trajectory
    basis_kwargs : dict, optional
        Keyword arguments for the 2D trajectory basis, by default {}
    expansion_kwargs : dict, optional
        Keyword arguments for the 3D trajectory expansion, by default {}

    Returns
    -------
    array_like
        3D trajectory
    """
    # Initialization of the keyword arguments
    if basis_kwargs is None:
        basis_kwargs = {}
    if expansion_kwargs is None:
        expansion_kwargs = {}
    # Initialization and warnings for 2D trajectory basis
    bases = {
        "radial": initialize_2D_radial,
        "spiral": initialize_2D_spiral,
        "rosette": initialize_2D_rosette,
        "cones": initialize_2D_cones,
    }
    if isinstance(basis, np.ndarray):
        trajectory2D = basis
    elif basis not in bases.keys():
        raise NotImplementedError(f"Unknown 2D trajectory basis: {basis}")
    else:
        basis_function = bases[basis]
        trajectory2D = basis_function(Nc, Ns, **basis_kwargs)

    # Initialization and warnings for 3D trajectory expansion
    expansions = {
        "stacks": stack_2D_to_3D_expansion,
        "rotations": rotate_2D_to_3D_expansion,
        "cones": cone_2D_to_3D_expansion,
        "helices": helix_2D_to_3D_expansion,
    }
    if expansion not in expansions.keys():
        raise NotImplementedError(f"Unknown 3D expansion: {expansion}")
    expansion_function = expansions[expansion]
    trajectory3D = expansion_function(trajectory2D, nb_repetitions, **expansion_kwargs)
    return trajectory3D.reshape((nb_repetitions * Nc, Ns, 3))


def initialize_3D_cones(Nc, Ns, tilt="golden", in_out=False, nb_zigzags=5, width=1):
    """Initialize 3D trajectories with cones.

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
    width : float, optional
        Width of a cone such that 1 has no redundacy and full coverage, by default 1

    Returns
    -------
    array_like
        3D cones trajectory
    """
    # Initialize first cone characteristics
    radius = np.linspace(-KMAX if (in_out) else 0, KMAX, Ns)
    angles = np.linspace(
        -2 * np.pi * nb_zigzags if (in_out) else 0, 2 * np.pi * nb_zigzags, Ns
    )
    trajectory = np.zeros((Nc, Ns, 3))
    trajectory[:, :, 0] = radius
    trajectory[:, :, 1] = (
        radius * np.cos(angles) * width * 2 * np.pi / Nc ** (2 / 3) / (1 + in_out)
    )
    trajectory[:, :, 2] = (
        radius * np.sin(angles) * width * 2 * np.pi / Nc ** (2 / 3) / (1 + in_out)
    )

    # Determine mostly evenly distributed points on sphere
    points = np.zeros((Nc, 3))
    phi = initialize_tilt(tilt) * np.arange(Nc) / (1 + in_out)
    points[:, 0] = np.linspace(-1, 1, Nc)
    radius = np.sqrt(1 - points[:, 0] ** 2)
    points[:, 1] = np.cos(phi) * radius
    points[:, 2] = np.sin(phi) * radius

    # Rotate initial cone Nc times
    for i in np.arange(1, Nc)[::-1]:
        v1 = np.array((1, 0, 0))
        v2 = points[i]
        rotation = Rv(v1, v2, normalize=False)
        trajectory[i] = (rotation @ trajectory[0].T).T
    return trajectory.reshape((Nc, Ns, 3))


def initialize_3D_wave_caipi(
    Nc,
    Ns,
    nb_revolutions=5,
    width=1,
    packing="triangular",
    shape="square",
    spacing=(1, 1),
):
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
    shape : str or float, optional
        Shape over the 2D kx-ky plane to pack with shots,
        either defined as `str` ("circle", "square", "diamond")
        or as `float` through p-norms following the conventions
        of the `ord` parameter from `numpy.linalg.norm`,
        by default "circle".
    spacing : tuple(int, int)
        Spacing between helices over the 2D kx-ky plane
        normalized similarly to `width` to correspond to
        helix diameters, by default (1, 1).

    Returns
    -------
    array_like
        3D wave-CAIPI trajectory

    References
    ----------
    .. [Bil+15] Bilgic, Berkin, Borjan A. Gagoski, Stephen F. Cauley, Audrey P. Fan,
       Jonathan R. Polimeni, P. Ellen Grant, Lawrence L. Wald, and Kawin Setsompop.
       "Wave‐CAIPI for highly accelerated 3D imaging."
       Magnetic resonance in medicine 73, no. 6 (2015): 2152-2162.
    """
    trajectory = np.zeros((Nc, Ns, 3))
    spacing = np.array(spacing)

    # Initialize first shot
    angles = nb_revolutions * 2 * np.pi * np.arange(0, Ns) / Ns
    trajectory[0, :, 0] = width * np.cos(angles)
    trajectory[0, :, 1] = width * np.sin(angles)
    trajectory[0, :, 2] = np.linspace(-1, 1, Ns)

    # Packing
    side = 2 * int(np.ceil(np.sqrt(Nc))) * np.max(spacing)
    if packing in ["random", "uniform"]:
        positions = 2 * side * (np.random.random((side * side, 2)) - 0.5)
    elif packing in ["circle", "circular"]:
        # Circle packing
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
    elif packing in ["square", "triangle", "triangular", "hexagon", "hexagonal"]:
        # Square packing or initialize hexagonal/triangular packing
        px, py = np.meshgrid(
            np.arange(-side + 1, side, 2), np.arange(-side + 1, side, 2)
        )
        positions = np.stack([px.flatten(), py.flatten()], axis=-1).astype(float)

    if packing in ["triangle", "triangular", "hexagon", "hexagonal"]:
        # Hexagonal/triangular packing based on square packing
        positions[::2, 1] += 1 / 2
        positions[1::2, 1] -= 1 / 2
        ratio = nl.norm(np.diff(positions[:2], axis=-1))
        positions[:, 0] /= ratio / 2

    # Remove points by distance to fit both shape and Nc
    main_order = initialize_shape_norm(shape)
    tie_order = 2 if (main_order != 2) else np.inf  # breaking ties
    positions = np.array(positions) * spacing
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
    Nc, Ns, curve_index=0.2, nb_revolutions=1, tilt="golden", in_out=False
):
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
    tilt : str, float, optional
        Angle between shots around z-axis over precession, by default "golden"
    in_out : bool
        Whether the curves are going in-and-out or start from the center,
        by default False

    Returns
    -------
    array_like
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
    trajectory = np.zeros((Nc, Ns // (1 + in_out), 3))
    period = 4 * ellipk(curve_index**2)
    times = np.linspace(0, nb_revolutions * period, Ns // (1 + in_out), endpoint=False)

    # Initialize first shot
    jacobi = ellipj(times, curve_index**2)
    trajectory[0, :, 0] = jacobi[0] * np.cos(curve_index * times)
    trajectory[0, :, 1] = jacobi[0] * np.sin(curve_index * times)
    trajectory[0, :, 2] = jacobi[1]

    # Make it volumetric instead of just a sphere surface
    magnitudes = np.sqrt(np.linspace(0, 1, Ns // (1 + in_out)))
    trajectory = magnitudes.reshape((1, -1, 1)) * trajectory

    # Determine mostly evenly distributed points on sphere
    points = np.zeros((Nc, 3))
    phi = initialize_tilt(tilt) * np.arange(Nc)
    points[:, 0] = np.linspace(-1, 1, Nc)
    radius = np.sqrt(1 - points[:, 0] ** 2)
    points[:, 1] = np.cos(phi) * radius
    points[:, 2] = np.sin(phi) * radius

    # Rotate initial shot Nc times
    for i in np.arange(1, Nc)[::-1]:
        v1 = np.array((1, 0, 0))
        v2 = points[i]
        rotation = Rv(v1, v2, normalize=False)
        trajectory[i] = (rotation @ trajectory[0].T).T

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
    Nc, Ns, nb_shells, spiral_reduction=1, shell_tilt="intergaps", shot_tilt="uniform"
):
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
    spiral_reduction : float
        Factor used to reduce the automatic spiral curvature, by default 1
    shell_tilt : str, float, optional
        Angle between consecutive shells along z-axis, by default "intergaps"
    shot_tilt : str, float, optional
        Angle between shots over a shell surface along z-axis, by default "uniform"

    Returns
    -------
    array_like
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
    Nc, Ns, nb_shells, shell_tilt=np.pi, ring_tilt=np.pi / 2
):
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
    array_like
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
    Nc,
    Ns,
    nb_shells,
    curve_index=0.5,
    nb_revolutions=1,
    shell_tilt="uniform",
    shot_tilt="uniform",
):
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
        Angle between consecutive shells along z-axis, by default "intergaps"
    shot_tilt : str, float, optional
        Angle between shots over a shell surface along z-axis, by default "uniform"

    Returns
    -------
    array_like
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


#########
# UTILS #
#########


def duplicate_per_axes(trajectory, axes=(0, 1, 2)):
    """
    Duplicate a trajectory along the specified axes.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to duplicate.
    axes : tuple, optional
        Axes along which to duplicate the trajectory, by default (0, 1, 2)

    Returns
    -------
    array_like
        Duplicated trajectory along the specified axes.
    """
    # Copy input trajectory along other axes
    new_trajectory = []
    if 0 in axes:
        new_trajectory.append(trajectory)
    if 1 in axes:
        dp_trajectory = np.copy(trajectory)
        dp_trajectory[..., [1, 2]] = dp_trajectory[..., [2, 1]]
        new_trajectory.append(dp_trajectory)
    if 2 in axes:
        dp_trajectory = np.copy(trajectory)
        dp_trajectory[..., [2, 0]] = dp_trajectory[..., [0, 2]]
        new_trajectory.append(dp_trajectory)
    new_trajectory = np.concatenate(new_trajectory, axis=0)
    return new_trajectory


def _radialize_center_out(trajectory, nb_samples):
    """Radialize a trajectory from the center to the outside."""
    Nc, Ns = trajectory.shape[:2]
    new_trajectory = np.copy(trajectory)
    for i in range(Nc):
        point = trajectory[i, nb_samples]
        new_trajectory[i, :nb_samples, 0] = np.linspace(0, point[0], nb_samples)
        new_trajectory[i, :nb_samples, 1] = np.linspace(0, point[1], nb_samples)
        new_trajectory[i, :nb_samples, 2] = np.linspace(0, point[2], nb_samples)
    return new_trajectory


def _radialize_in_out(trajectory, nb_samples):
    """Radialize a trajectory from the inside to the outside."""
    Nc, Ns = trajectory.shape[:2]
    new_trajectory = np.copy(trajectory)
    first, half, second = (Ns - nb_samples) // 2, Ns // 2, (Ns + nb_samples) // 2
    for i in range(Nc):
        p1 = trajectory[i, first]
        new_trajectory[i, first:half, 0] = np.linspace(0, p1[0], nb_samples // 2)
        new_trajectory[i, first:half, 1] = np.linspace(0, p1[1], nb_samples // 2)
        new_trajectory[i, first:half, 2] = np.linspace(0, p1[2], nb_samples // 2)
        p2 = trajectory[i, second]
        new_trajectory[i, half:second, 0] = np.linspace(0, p2[0], nb_samples // 2)
        new_trajectory[i, half:second, 1] = np.linspace(0, p2[1], nb_samples // 2)
        new_trajectory[i, half:second, 2] = np.linspace(0, p2[2], nb_samples // 2)
    return new_trajectory


def radialize_center(trajectory, nb_samples, in_out=False):
    """Radialize a trajectory.

    Parameters
    ----------
    trajectory : array_like
        Trajectory to radialize.
    nb_samples : int
        Number of samples to keep.
    in_out : bool, optional
        Whether the radialization is from the inside to the outside, by default False
    """
    # Make nb_samples into straight lines around the center
    if in_out:
        return _radialize_in_out(trajectory, nb_samples)
    else:
        return _radialize_center_out(trajectory, nb_samples)
