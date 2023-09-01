"""Utility functions for the trajectory design."""
import warnings
import numpy as np

#############
# CONSTANTS #
#############

KMAX = 0.5

DEFAULT_RESOLUTION = 6e-4  # 0.6 mm isotropic
DEFAULT_CONE_ANGLE = np.pi / 2  # rad
DEFAULT_HELIX_ANGLE = np.pi  # rad

DEFAULT_RASTER_TIME_MS = 10e-3  # ms
DEFAULT_GYROMAGNETIC_RATIO = 42.576e3  # MHz/T

DEFAULT_GMAX = 0.04  # T/m
DEFAULT_SMAX = 100.0e-3  # mT/m/s


###############
# CONSTRAINTS #
###############


def _check_gradient_constraints(
    gradients,
    slews,
    gmax=DEFAULT_GMAX,
    smax=DEFAULT_SMAX,
):
    """Check if a trajectory satisfies the gradient constraints.

    Parameters
    ----------
    gradients : np.ndarray
        Gradients to check.
    slews: np.ndarray
        Slews to check
    gmax : float, optional
        Maximum gradient amplitude in T/m. The default is DEFAULT_GMAX.
    smax : float, optional
        Maximum slew rate in mT/m/s. The default is DEFAULT_SMAX.

    Returns
    -------
    bool
        True if the trajectory satisfies the constraints, False otherwise.
    float
        Maximum gradient amplitude in T/m.
    float
        Maximum slew rate in mT/m/s.
    """
    max_grad = np.max(np.linalg.norm(gradients, axis=-1))
    max_slew = np.max(np.linalg.norm(slews, axis=-1))
    return (max_grad < gmax) and (max_slew < smax), max_grad, max_slew


def compute_gradients(
    trajectory,
    traj_norm_factor=KMAX,
    resolution=DEFAULT_RESOLUTION,
    raster_time=DEFAULT_RASTER_TIME_MS,
    gamma=DEFAULT_GYROMAGNETIC_RATIO,
    check_constraints=False,
    smax=DEFAULT_SMAX,
    gmax=DEFAULT_GMAX,
):
    """Compute Gradient and Slew rate from a trajectory.

    Also check for constraints violations if check_constraints is True.


    Parameters
    ----------
    trajectory : np.ndarray
        array of trajectory points
    traj_norm_factor : float, optional
        Normalization factor for trajectory points.
        The default is KMAX.
    resolution : float, optional
        Resolution of the trajectory in mm. The default is DEFAULT_RESOLUTION.
    raster_time : float, optional
        Duration of each point in the trajectory in ms.
        The default is DEFAULT_RASTER_TIME_MS.
    gamma : float, optional
        Gyromagnetic ratio of the particle. The default is DEFAULT_GYROMAGNETIC_RATIO.
    check_constraints : bool, optional
        If True, also check for constraints violations.
        The default is False.
    smax : float, optional
        Maximum Slew rate. The default is DEFAULT_SMAX.
    gmax : float, optional
        Maximum Gradient magnitude. The default is DEFAULT_GMAX.


    Returns
    -------
    gradients : np.ndarray
        array of gradients (x,y,z) in a 3D space
    slews : np.ndarray
        array of slew rates (x,y,z) in a 3D space
    start_positions : np.ndarray
        array of start positions (x,y,z) in a 3D space
    """
    # normalize the trajectory
    trajectory = trajectory / traj_norm_factor / (2 * resolution)

    # compute gradients and slew rates
    gradients = np.diff(trajectory, axis=1) / raster_time / gamma
    slews = np.diff(gradients, axis=1) / raster_time

    # compute the start position
    start_positions = trajectory[:, 0, :]
    if check_constraints:
        violation, maxG, maxS = _check_gradient_constraints(
            gradients=gradients,
            slews=slews,
            gmax=gmax,
            smax=smax,
        )
        if violation:
            warnings.warn(
                "Hard constraints violated! "
                f"Max Gradient magnitude: {maxG:.2f} > {gmax:.2f}"
                f"Max Slew rate: {maxS:.2f} > {smax:.2f}"
            )
    return gradients, start_positions, slews


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
