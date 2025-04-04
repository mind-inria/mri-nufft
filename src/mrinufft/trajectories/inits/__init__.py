"""Module containing all trajectories."""

from .eccentric import initialize_2D_eccentric, initialize_3D_eccentric
from .random_walk import initialize_2D_random_walk, initialize_3D_random_walk
from .travelling_salesman import (
    initialize_2D_travelling_salesman,
    initialize_3D_travelling_salesman,
)

__all__ = [
    # eccentric
    "initialize_2D_eccentric",
    "initialize_3D_eccentric",
    # random walk
    "initialize_2D_random_walk",
    "initialize_3D_random_walk",
    # travelling salesman
    "initialize_2D_travelling_salesman",
    "initialize_3D_travelling_salesman",
]
