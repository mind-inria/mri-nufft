"""Helical and annular shell 3D trajectory initializations."""

import numpy as np
import numpy.linalg as nl
from numpy.typing import NDArray

from mrinufft.trajectories.maths import Ry, Rz
from mrinufft.trajectories.utils import KMAX, initialize_tilt


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
       "Three-dimensional MRI with an undersampled spherical shells trajectory."
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
    It is similar to other trajectories based on concentric rings but
    rings are split into halves and rotated to be recombined with
    halves from other rings, in order to better balance the shot lengths
    between longer and shorter rings from a same shell.

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
