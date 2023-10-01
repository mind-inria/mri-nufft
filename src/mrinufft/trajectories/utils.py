"""Utility functions for the trajectory design."""
import numpy as np


#############
# CONSTANTS #
#############

KMAX = 0.5

DEFAULT_CONE_ANGLE = np.pi / 2  # rad
DEFAULT_HELIX_ANGLE = np.pi  # rad

CARBON_GYROMAGNETIC_RATIO = 10708.4  # kHz/T
HYDROGEN_GYROMAGNETIC_RATIO = 42576.384  # kHz/T
PHOSPHOROUS_GYROMAGNETIC_RATIO = 17235  # kHz/T
SODIUM_GYROMAGNETIC_RATIO = 11262  # kHz/T

DEFAULT_RESOLUTION = 6e-4  # m, i.e. 0.6 mm isotropic
DEFAULT_RASTER_TIME = 10e-3  # ms

DEFAULT_GMAX = 0.04  # T/m
DEFAULT_SMAX = 0.1  # T/m/ms


###############
# CONSTRAINTS #
###############

def normalize_trajectory(
    trajectory,
    norm_factor=KMAX,
    resolution=DEFAULT_RESOLUTION,
):
    return trajectory * norm_factor * (2 * resolution)


def unnormalize_trajectory(
    trajectory,
    norm_factor=KMAX,
    resolution=DEFAULT_RESOLUTION,
):
    return trajectory / norm_factor / (2 * resolution)


def convert_trajectory_to_gradients(
    trajectory,
    norm_factor=KMAX,
    resolution=DEFAULT_RESOLUTION,
    raster_time=DEFAULT_RASTER_TIME,
    gamma=HYDROGEN_GYROMAGNETIC_RATIO,
):
    # Un-normalize the trajectory from NUFFT usage
    trajectory = unnormalize_trajectory(trajectory, norm_factor, resolution)

    # Compute gradients and starting positions
    gradients = np.diff(trajectory, axis=1) / gamma / raster_time
    initial_positions = trajectory[:, 0, :]
    return gradients, initial_positions


def convert_gradients_to_trajectory(
    gradients,
    initial_positions=None,
    norm_factor=KMAX,
    resolution=DEFAULT_RESOLUTION,
    raster_time=DEFAULT_RASTER_TIME,
    gamma=HYDROGEN_GYROMAGNETIC_RATIO,
):
    # Handle no initial positions
    if (initial_positions is None):
        initial_positions = np.zeros((gradients.shape[0], 1, gradients.shape[-1]))

    # Prepare and integrate gradients
    trajectory = gradients * gamma * raster_time
    trajectory = np.concatenate([initial_positions, trajectory])
    trajectory = np.cumsum(trajectory, axis=1)

    # Normalize the trajectory for NUFFT usage
    trajectory = normalize_trajectory(trajectory, norm_factor, resolution)
    return trajectory


def convert_gradients_to_slew_rates(
    gradients,
    raster_time=DEFAULT_RASTER_TIME,
):
    # Compute slew rates and starting gradients
    slewrates = np.diff(gradients, axis=1) / raster_time
    initial_gradients = gradients[:, 0, :]
    return slewrates, initial_gradients


def convert_slew_rates_to_gradients(
    slewrates,
    initial_gradients=None,
    raster_time=DEFAULT_RASTER_TIME,
):
    # Handle no initial gradients
    if (initial_gradients is None):
        initial_gradients = np.zeros((slewrates.shape[0], 1, slewrates.shape[-1]))

    # Prepare and integrate slew rates
    gradients = slewrates * raster_time
    gradients = np.concatenate([initial_gradients, gradients])
    gradients = np.cumsum(gradients, axis=1)
    return gradients


def compute_gradients_and_slew_rates(
    trajectory,
    norm_factor=KMAX,
    resolution=DEFAULT_RESOLUTION,
    raster_time=DEFAULT_RASTER_TIME,
    gamma=HYDROGEN_GYROMAGNETIC_RATIO,
):
    # Convert normalized trajectory to gradients
    gradients, _ = convert_trajectory_to_gradients(
        trajectory,
        norm_factor=norm_factor,
        resolution=resolution,
        raster_time=raster_time,
        gamma=gamma,
    )

    # Convert gradients to slew rates
    slewrates, _ = convert_gradients_to_slew_rates(gradients, raster_time)

    return gradients, slewrates


def check_hardware_constraints(
    gradients,
    slewrates,
    gmax=DEFAULT_GMAX,
    smax=DEFAULT_SMAX,
    order=None
):
    """Check if a trajectory satisfies the gradient constraints.

    Parameters
    ----------
    gradients : np.ndarray
        Gradients to check
    slewrates: np.ndarray
        Slewrates to check
    gmax : float, optional
        Maximum gradient amplitude in T/m. The default is DEFAULT_GMAX.
    smax : float, optional
        Maximum slew rate in T/m/ms. The default is DEFAULT_SMAX.
    order : int or str, optional
        Norm order, following the numpy.linalg.norm `ord` convention.
        The default is None.

    Returns
    -------
    bool
        True if the trajectory satisfies the constraints, False otherwise.
    float
        Maximum gradient amplitude in T/m.
    float
        Maximum slew rate in T/m/ms.
    """
    max_grad = np.max(np.linalg.norm(gradients, axis=-1, ord=order))
    max_slew = np.max(np.linalg.norm(slewrates, axis=-1, ord=order))
    return (max_grad < gmax) and (max_slew < smax), max_grad, max_slew


###############
# MATHEMATICS #
###############

def compute_greatest_common_divider(p, q):
    """Compute the greatest common divider of two integers p and q.

    Parameters
    ----------
    p : int
        First integer.
    q : int
        Second integer.

    Returns
    -------
    int
        The greatest common divider of p and q.
    """
    while q != 0:
        p, q = q, p % q
    return p


def compute_coprime_factors(Nc, length, start=1, update=1):
    """Compute a list of coprime factors of Nc.

    Parameters
    ----------
    Nc : int
        Number to factorize.
    length : int
        Number of coprime factors to compute.
    start : int, optional
        First number to check. The default is 1.
    update : int, optional
        Increment between two numbers to check. The default is 1.

    Returns
    -------
    list
        List of coprime factors of Nc.
    """
    count = start
    coprimes = []
    while len(coprimes) < length:
        if compute_greatest_common_divider(Nc, count) == 1:
            coprimes.append(count)
        count += update
    return coprimes


#############
# ROTATIONS #
#############

def R2D(theta):
    """Initialize 2D rotation matrix.

    Parameters
    ----------
    theta : float
        Rotation angle in rad.

    Returns
    -------
    np.ndarray
        2D rotation matrix.
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def Rx(theta):
    """Initialize 3D rotation matrix around x axis.

    Parameters
    ----------
    theta : float
        Rotation angle in rad.

    Returns
    -------
    np.ndarray
        2D rotation matrix.
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def Ry(theta):
    """Initialize 3D rotation matrix around y axis.

    Parameters
    ----------
    theta : float
        Rotation angle in rad.

    Returns
    -------
    np.ndarray
        2D rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def Rz(theta):
    """Initialize 3D rotation matrix around z axis.

    Parameters
    ----------
    theta : float
        Rotation angle in rad.

    Returns
    -------
    np.ndarray
        2D rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def Rv(v1, v2, normalize=True):
    """Initialize 3D rotation matrix from two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.
    normalize : bool, optional
        Normalize the vectors. The default is True.

    Returns
    -------
    np.ndarray
        3D rotation matrix.
    """
    if normalize:
        v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2)
    v3 = np.cross(v1, v2)
    cross_matrix = np.cross(v3, np.identity(v3.shape[0]) * -1)
    return np.identity(3) + cross_matrix + cross_matrix @ cross_matrix / (1 + cos_theta)


###########
# OPTIONS #
###########

def initialize_tilt(tilt, nb_partitions=1):
    r"""Initialize the tilt angle.

    Parameters
    ----------
    tilt : str or float
        Tilt angle in rad or name of the tilt.
    nb_partitions : int, optional
        Number of partitions. The default is 1.

    Returns
    -------
    float
        Tilt angle in rad.

    Raises
    ------
    NotImplementedError
        If the tilt name is unknown.

    Notes
    -----
    The following values are accepted for the tilt name, with :math:`N` the number of
    partitions:

    - "none": no tilt
    - "uniform": uniform tilt: 2:math:`\pi / N`
    - "intergaps": :math:`\pi/2N`
    - "inverted": inverted tilt :math:`\pi/N + \pi`
    - "golden": tilt of the golden angle :math:`\pi(3-\sqrt{5})`
    - "mri golden": tilt of the golden angle used in MRI :math:`\pi(\sqrt{5}-1)/2`

    """
    if not isinstance(tilt, str):
        return tilt
    elif tilt == "none":
        return 0
    elif tilt == "uniform":
        return 2 * np.pi / nb_partitions
    elif tilt == "intergaps":
        return np.pi / nb_partitions / 2
    elif tilt == "inverted":
        return np.pi / nb_partitions + np.pi
    elif tilt == "golden":
        return np.pi * (3 - np.sqrt(5))
    elif tilt == "mri golden":
        return np.pi * (np.sqrt(5) - 1) / 2
    else:
        raise NotImplementedError(f"Unknown tilt name: {tilt}")


def initialize_spiral(spiral):
    """Initialize the spiral type.

    Parameters
    ----------
    spiral : str or int
        Spiral type or number of interleaves.

    Returns
    -------
    int
        Spiral type.
    """
    if not isinstance(spiral, str):
        return spiral
    elif spiral == "archimedes":
        return 1
    elif spiral == "fermat":
        return 2
    else:
        raise NotImplementedError(f"Unknown spiral name: {spiral}")
