"""Utility functions for mathematical operations."""

import numpy as np
import numpy.linalg as nl

CIRCLE_PACKING_DENSITY = np.pi / (2 * np.sqrt(3))
EIGENVECTOR_2D_FIBONACCI = (0.4656, 0.6823, 1)


##########
# PRIMES #
##########


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
        # Check greatest common divider (gcd)
        if np.gcd(Nc, count) == 1:
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
        3D rotation matrix.
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
        3D rotation matrix.
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
        3D rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def Rv(v1, v2, normalize=True, eps=1e-8):
    """Initialize 3D rotation matrix from two vectors.

    Initialize a 3D rotation matrix from two vectors using Rodrigues's rotation
    formula. Note that the rotation is carried around the axis orthogonal to both
    vectors from the origin, and therefore is undetermined when both vectors
    are colinear. While this case is handled manually, close cases might result
    in approximative behaviors.

    Parameters
    ----------
    v1 : np.ndarray
        Source vector.
    v2 : np.ndarray
        Target vector.
    normalize : bool, optional
        Normalize the vectors. The default is True.

    Returns
    -------
    np.ndarray
        3D rotation matrix.
    """
    # Check for colinearity, not handled by Rodrigues' coefficients
    if nl.norm(np.cross(v1, v2)) < eps:
        sign = np.sign(np.dot(v1, v2))
        return sign * np.identity(3)

    if normalize:
        v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2)
    v3 = np.cross(v1, v2)
    cross_matrix = np.cross(v3, np.identity(v3.shape[0]) * -1)
    return np.identity(3) + cross_matrix + cross_matrix @ cross_matrix / (1 + cos_theta)


def Ra(vector, theta):
    """Initialize 3D rotation matrix around an arbitrary vector.

    Initialize a 3D rotation matrix to rotate around `vector` by an angle `theta`.
    It corresponds to a generalized formula with `Rx`, `Ry` and `Rz` as subcases.

    Parameters
    ----------
    vector : np.ndarray
        Vector defining the rotation axis, automatically normalized.
    theta : float
        Angle in radians defining the rotation around `vector`.

    Returns
    -------
    np.ndarray
        3D rotation matrix.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    v_x, v_y, v_z = vector / np.linalg.norm(vector)
    return np.array(
        [
            [
                cos_t + v_x**2 * (1 - cos_t),
                v_x * v_y * (1 - cos_t) + v_z * sin_t,
                v_x * v_z * (1 - cos_t) - v_y * sin_t,
            ],
            [
                v_y * v_x * (1 - cos_t) - v_z * sin_t,
                cos_t + v_y**2 * (1 - cos_t),
                v_y * v_z * (1 - cos_t) + v_x * sin_t,
            ],
            [
                v_z * v_x * (1 - cos_t) + v_y * sin_t,
                v_z * v_y * (1 - cos_t) - v_x * sin_t,
                cos_t + v_z**2 * (1 - cos_t),
            ],
        ]
    )


#############
# FIBONACCI #
#############


def is_from_fibonacci_sequence(n):
    """Check if an integer belongs to the Fibonacci sequence.

    An integer belongs to the Fibonacci sequence if either
    :math:`5*n²+4` or :math:`5*n²-4` is a perfect square
    (`Wikipedia <https://en.wikipedia.org/wiki/Fibonacci_sequence#Recognizing_Fibonacci_numbers>`_).

    Parameters
    ----------
    n : int
        Integer to check.

    Returns
    -------
    bool
        Whether or not ``n`` belongs to the Fibonacci sequence.
    """

    def _is_perfect_square(n):
        r = int(np.sqrt(n))
        return r * r == n

    return _is_perfect_square(5 * n**2 + 4) or _is_perfect_square(5 * n**2 - 4)


def get_closest_fibonacci_number(x):
    """Provide the closest Fibonacci number.

    Parameters
    ----------
    x : float
        Value to match.

    Returns
    -------
    int
        Closest number from the Fibonacci sequence.
    """
    # Find the power such that x ~= phi ** power
    phi = (1 + np.sqrt(5)) / 2
    power = np.ceil(np.log(x) / np.log(phi)) + 1

    # Check closest between the ones below and above n
    lower_xf = int(np.around(phi ** (power) / np.sqrt(5)))
    upper_xf = int(np.around(phi ** (power + 1) / np.sqrt(5)))
    xf = lower_xf if (x - lower_xf) < (upper_xf - x) else upper_xf
    return xf


def generate_fibonacci_lattice(nb_points, epsilon=0.25):
    """Generate 2D Cartesian coordinates using the Fibonacci lattice.

    Place 2D points over a 1x1 square following the Fibonacci lattice.

    Parameters
    ----------
    nb_points : int
        Number of 2D points to generate.
    epsilon : float
        Continuous offset used to reduce initially wrong lattice behavior.

    Returns
    -------
    np.ndarray
        Array of 2D Cartesian coordinates covering a 1x1 square.
    """
    angle = (1 + np.sqrt(5)) / 2
    fibonacci_square = np.zeros((nb_points, 2))
    fibonacci_square[:, 0] = (np.arange(nb_points) / angle) % 1
    fibonacci_square[:, 1] = (np.arange(nb_points) + epsilon) / (
        nb_points - 1 + 2 * epsilon
    )
    return fibonacci_square


def generate_fibonacci_circle(nb_points, epsilon=0.25):
    """Generate 2D Cartesian coordinates shaped as Fibonacci spirals.

    Place 2D points structured as Fibonacci spirals by distorting
    a square Fibonacci lattice into a circle of radius 1.

    Parameters
    ----------
    nb_points : int
        Number of 2D points to generate.
    epsilon : float
        Continuous offset used to reduce initially wrong lattice behavior.

    Returns
    -------
    np.ndarray
        Array of 2D Cartesian coordinates covering a circle of radius 1.
    """
    fibonacci_square = generate_fibonacci_lattice(nb_points, epsilon)
    radius = np.sqrt(fibonacci_square[:, 1])
    angles = 2 * np.pi * fibonacci_square[:, 0]

    fibonacci_circle = np.zeros((nb_points, 2))
    fibonacci_circle[:, 0] = radius * np.cos(angles)
    fibonacci_circle[:, 1] = radius * np.sin(angles)
    return fibonacci_circle


def generate_fibonacci_sphere(nb_points, epsilon=0.25):
    """Generate 3D Cartesian coordinates as a Fibonacci sphere.

    Place 3D points almost evenly over a sphere surface of radius
    1 by distorting a square Fibonacci lattice into a sphere.

    Parameters
    ----------
    nb_points : int
        Number of 3D points to generate.
    epsilon : float
        Continuous offset used to reduce initially wrong lattice behavior.

    Returns
    -------
    np.ndarray
        Array of 3D Cartesian coordinates covering a sphere of radius 1.
    """
    fibonacci_square = generate_fibonacci_lattice(nb_points, epsilon)
    theta = 2 * np.pi * fibonacci_square[:, 0]
    phi = np.arccos(1 - 2 * fibonacci_square[:, 1])

    fibonacci_sphere = np.zeros((nb_points, 3))
    fibonacci_sphere[:, 0] = np.cos(theta) * np.sin(phi)
    fibonacci_sphere[:, 1] = np.sin(theta) * np.sin(phi)
    fibonacci_sphere[:, 2] = np.cos(phi)
    return fibonacci_sphere
