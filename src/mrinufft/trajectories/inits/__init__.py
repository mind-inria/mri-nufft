"""Module containing all trajectories."""

from .random_walk import initialize_2D_random_walk, initialize_3D_random_walk
from .travelling_salesman import (
    initialize_2D_travelling_salesman,
    initialize_3D_travelling_salesman,
)

__all__ = [
    "initialize_2D_random_walk",
    "initialize_3D_random_walk",
    "initialize_2D_travelling_salesman",
    "initialize_3D_travelling_salesman",
]
