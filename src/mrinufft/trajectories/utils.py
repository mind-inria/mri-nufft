"""Utility functions for the trajectory design."""
from typing import Tuple
import warnings
import numpy as np

from .io import write_gradient_file, KMAX

#############
# CONSTANTS #
#############

DEFAULT_CONE_ANGLE = np.pi / 2  # rad
DEFAULT_HELIX_ANGLE = np.pi  # rad

DEFAULT_RASTER_TIME = 10e-3  # ms
DEFAULT_GYROMAGNETIC_RATIO = 42.576e3  # MHz/T

DEFAULT_GMAX = 0.04  # T/m
DEFAULT_SMAX = 100.0e-3  # mT/m/s


###############
# CONSTRAINTS #
###############

def get_grads_from_kspace_points(
    trajectory: np.ndarray,
    FOV: Tuple[float, ...],
    img_size: Tuple[int, ...],
    trajectory_normalization_factor: float = KMAX,
    gyromagnetic_constant: float = DEFAULT_GYROMAGNETIC_RATIO,
    gradient_raster_time: float = DEFAULT_RASTER_TIME,
    check_constraints: bool = True,
    gradient_mag_max: float = DEFAULT_GMAX,
    slew_rate_max: float = DEFAULT_SMAX,
    grad_filename: str = None,
    write_kwargs: dict = {},
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate gradients from k-space points. Also returns start positions, slew rates and 
    allows for checking of scanner constraints.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory in k-space points. Shape (num_shots, num_samples_per_shot, dimension).
    FOV : tuple
        Field of view
    img_size : tuple
        Image size
    trajectory_normalization_factor : float, optional
        Trajectory normalization factor, by default 0.5
    gyromagnetic_constant : float, optional
        Gyromagnetic constant, by default 42.576e3
    gradient_raster_time : float, optional
        Gradient raster time, by default 0.01
    check_constraints : bool, optional
        Check scanner constraints, by default True
    gradient_mag_max : float, optional
        Maximum gradient magnitude, by default 40e-3
    slew_rate_max : float, optional
        Maximum slew rate, by default 100e-3
    grad_filename : str, optional
        Gradient filename, by default None. If none gradient file is not written
    write_kwargs : dict, optional
        Keyword arguments for writing gradients. See io.py for details.
        
    Returns
    -------
    gradients : np.ndarray
        Gradients. Shape (num_shots-1, num_samples_per_shot, dimension).
    start_positions : np.ndarray
        Start positions. Shape (num_shots, dimension).
    slew_rate : np.ndarray
        Slew rates. Shape (num_shots-2, num_samples_per_shot, dimension).
    """
    # normalize trajectory by image size
    if trajectory_normalization_factor:
        trajectory = trajectory * np.array(img_size) / (
            2 * np.array(FOV)
        ) / trajectory_normalization_factor

    # calculate gradients and slew
    gradients = np.diff(trajectory, axis=1) / gyromagnetic_constant / gradient_raster_time
    start_positions = trajectory[:, 0, :]
    slew_rate = np.diff(gradients, axis=1) / gradient_raster_time

    # check constraints
    if check_constraints:
        if np.max(gradients) > gradient_mag_max:
            warnings.warn(
                "Gradient Maximum Maginitude overflow from Machine capabilities"
            )
        if np.max(slew_rate) > slew_rate_max:
            occurences = np.where(slew_rate > slew_rate_max)
            warnings.warn(
                "Slew Rate overflow from Machine capabilities!\n"
                "Occurences per shot : "
                + str(len(occurences[0]) / trajectory.shape[0])
                + "\n"
                "Max Value : "
                + str(np.max(np.abs(slew_rate)))
            )
    if grad_filename is not None:
        write_gradient_file(
            gradients=gradients,
            start_positions=start_positions,
            grad_filename=grad_filename,
            img_size=img_size, 
            FOV=FOV,
            gyromagnetic_constant=gyromagnetic_constant,
            **write_kwargs)
    return gradients, start_positions, slew_rate


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
