"""Utility module for mathematical operations."""

# Constants
import numpy as np

from .fibonacci import (
    generate_fibonacci_circle,
    generate_fibonacci_lattice,
    generate_fibonacci_sphere,
    get_closest_fibonacci_number,
    is_from_fibonacci_sequence,
)
from .primes import compute_coprime_factors
from .rotations import R2D, Ra, Rv, Rx, Ry, Rz
from .tsp_solver import solve_tsp_with_2opt

CIRCLE_PACKING_DENSITY = np.pi / (2 * np.sqrt(3))
EIGENVECTOR_2D_FIBONACCI = (0.4656, 0.6823, 1)
