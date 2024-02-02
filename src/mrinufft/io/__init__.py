"""Input/Output module for trajectories and data."""

from .cfl import traj2cfl, cfl2traj
from .nsp import read_trajectory, write_trajectory


__all__ = [
    "traj2cfl",
    "cfl2traj",
    "read_trajectory",
    "write_trajectory",
]
