"""2D trajectory initializations."""

import numpy as np

from .utils import (
    KMAX,
    R2D,
    initialize_tilt,
    initialize_spiral,
    compute_coprime_factors,
)
from .tools import rotate


#####################
# CIRCULAR PATTERNS #
#####################


def initialize_2D_radial(Nc, Ns, tilt="uniform", in_out=False):
    """Initialize a 2D radial trajectory.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    tilt : str, float, optional
        Tilt of the shots, by default "uniform"
    in_out : bool, optional
        Whether to start from the center or not, by default False
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


def initialize_2D_spiral(
    Nc, Ns, tilt="uniform", in_out=False, nb_revolutions=1, spiral="archimedes"
):
    """Initialize a 2D spiral trajectory.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    tilt : str, float, optional
        Tilt of the shots, by default "uniform"
    in_out : bool, optional
        Whether to start from the center or not, by default False
    nb_revolutions : int, optional
        Number of revolutions, by default 1
    spiral : str, float, optional
        Spiral type, by default "archimedes"

    Returns
    -------
    array_like
        2D spiral trajectory
    """
    # Initialize a first shot in polar coordinates
    segment = np.linspace(-1 if (in_out) else 0, 1, Ns)
    radius = KMAX * segment
    angles = 2 * np.pi * nb_revolutions * (np.abs(segment) ** initialize_spiral(spiral))

    # Convert the first shot to Cartesian coordinates
    trajectory = np.zeros((Nc, Ns, 2))
    trajectory[0, :, 0] = radius * np.cos(angles)
    trajectory[0, :, 1] = radius * np.sin(angles)

    # Rotate the first shot Nc times
    rotation = R2D(initialize_tilt(tilt, Nc) / (1 + in_out)).T
    for i in range(1, Nc):
        trajectory[i] = trajectory[i - 1] @ rotation
    return trajectory


def initialize_2D_cones(Nc, Ns, tilt="uniform", in_out=False, nb_zigzags=5, width=1):
    """Initialize a 2D cone trajectory.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    tilt : str, optional
        Tilt of the shots, by default "uniform"
    in_out : bool, optional
        Whether to start from the center or not, by default False
    nb_zigzags : float, optional
        Number of zigzags, by default 5
    width : float, optional
        Width of the cone, by default 1

    Returns
    -------
    array_like
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


def initialize_2D_sinusoide(
    Nc, Ns, tilt="uniform", in_out=False, nb_zigzags=5, width=1
):
    """Initialize a 2D sinusoide trajectory.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    tilt : str, float, optional
        Tilt of the shots, by default "uniform"
    in_out : bool, optional
        Whether to start from the center or not, by default False
    nb_zigzags : float, optional
        Number of zigzags, by default 5
    width : float, optional
        Width of the sinusoide, by default 1

    Returns
    -------
    array_like
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


def initialize_2D_propeller(Nc, Ns, nb_strips):
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
       application to head motion and free‐breathing cardiac imaging."
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


def initialize_2D_rings(Nc, Ns, nb_rings):
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
    array_like
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


def initialize_2D_rosette(Nc, Ns, in_out=False, coprime_index=0):
    """Initialize a 2D rosette trajectory.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    in_out : bool, optional
        Whether to start from the center or not, by default False
    coprime_index : int, optional
        Index of the coprime factor, by default 0

    Returns
    -------
    array_like
        2D rosette trajectory

    """
    # Prepare to parametrize with coprime factor according to Nc parity
    odd = Nc % 2
    coprime = compute_coprime_factors(
        Nc // (2 - odd),
        coprime_index + 1,
        start=1 if odd else (Nc // 2) % 2 + 1,
        update=2,
    )[-1]

    # Define the whole curve in polar coordinates
    angles = np.pi * np.linspace(-1, 1, Nc * Ns) / (1 + odd)
    shift = np.pi * (odd - in_out) / 2
    radius = KMAX * np.sin(Nc / (2 - odd) * angles + shift)

    # Convert polar to Cartesian coordinates
    trajectory = np.zeros((Nc, Ns, 2))
    trajectory[:, :, 0] = (radius * np.cos(angles * coprime)).reshape((Nc, Ns))
    trajectory[:, :, 1] = (radius * np.sin(angles * coprime)).reshape((Nc, Ns))
    return trajectory


def initialize_2D_polar_lissajous(Nc, Ns, in_out=False, nb_segments=1, coprime_index=0):
    """Initialize a 2D polar Lissajous trajectory.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    in_out : bool, optional
        Whether to start from the center or not, by default False
    nb_segments : int, optional
        Number of segments, by default 1
    coprime_index : int, optional
        Index of the coprime factor, by default 0

    Returns
    -------
    array_like
        2D polar Lissajous trajectory
    """
    # Adapt the parameters to subcases
    nb_segments = nb_segments * (2 - in_out)
    Nc = Nc // nb_segments

    # Define the whole curve in polar coordinates
    segment = np.pi / 2 * np.linspace(-1, 1, Nc * Ns)
    shift = np.pi * (Nc % 2 - in_out) / 2
    radius = KMAX * np.sin(Nc * segment + shift)
    coprime_factors = compute_coprime_factors(Nc, coprime_index + 1, start=Nc % 2 + 1)
    angles = (
        np.pi
        / (1 + in_out)
        / nb_segments
        * np.sin((Nc - coprime_factors[-1]) * segment)
    )

    # Convert polar to Cartesian coordinates for one segment
    trajectory = np.zeros((Nc * nb_segments, Ns, 2))
    trajectory[:Nc, :, 0] = (radius * np.cos(angles)).reshape((Nc, Ns))
    trajectory[:Nc, :, 1] = (radius * np.sin(angles)).reshape((Nc, Ns))

    # Duplicate and rotate each segment
    rotation = R2D(initialize_tilt("uniform", (1 + in_out) * nb_segments))
    for i in range(Nc, Nc * nb_segments):
        trajectory[i] = trajectory[i - Nc] @ rotation
    return trajectory


#########################
# NON-CIRCULAR PATTERNS #
#########################


def initialize_2D_lissajous(Nc, Ns, density=1):
    """Initialize a 2D Lissajous trajectory.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    density : float, optional
        Density of the trajectory, by default 1

    Returns
    -------
    array_like
        2D Lissajous trajectory
    """
    # Define the whole curve in Cartesian coordinates
    segment = np.linspace(-1, 1, Ns)
    angles = np.pi / 2 * np.sign(segment) * np.abs(segment)

    # Define each shot independenty
    trajectory = np.zeros((Nc, Ns, 2))
    tilt = initialize_tilt("uniform", Nc)
    for i in range(Nc):
        trajectory[i, :, 0] = KMAX * np.sin(angles)
        trajectory[i, :, 1] = KMAX * np.sin(angles * density + i * tilt)
    return trajectory


def initialize_2D_waves(Nc, Ns, nb_zigzags=5, width=1):
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
    array_like
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
