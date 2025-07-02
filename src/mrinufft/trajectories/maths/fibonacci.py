"""Fibonacci-related functions."""

import numpy as np


def is_from_fibonacci_sequence(n: int) -> bool:
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

    def _is_perfect_square(n: int) -> bool:
        r = int(np.sqrt(n))
        return r * r == n

    return _is_perfect_square(5 * n**2 + 4) or _is_perfect_square(5 * n**2 - 4)


def get_closest_fibonacci_number(x: float) -> int:
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


def generate_fibonacci_lattice(nb_points: int, epsilon: float = 0.25) -> np.ndarray:
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


def generate_fibonacci_circle(nb_points: int, epsilon: float = 0.25) -> np.ndarray:
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


def generate_fibonacci_sphere(nb_points: int, epsilon: float = 0.25) -> np.ndarray:
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
