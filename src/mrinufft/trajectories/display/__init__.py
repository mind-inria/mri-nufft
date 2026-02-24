"""Display functions for trajectories."""

from mrinufft.trajectories.display.config import displayConfig
from mrinufft.trajectories.display.trajectories import (
    display_2D_trajectory,
    display_3D_trajectory,
    _setup_2D_ticks,
    _setup_3D_ticks,
)
from mrinufft.trajectories.display.gradients import (
    display_gradients_simply,
    display_gradients,
)

__all__ = [
    "displayConfig",
    "display_2D_trajectory",
    "display_3D_trajectory",
    "display_gradients_simply",
    "display_gradients",
    "get_gridded_trajectory",
]


def __getattr__(name):
    if name == "get_gridded_trajectory":
        from mrinufft.trajectories.display.advanced import get_gridded_trajectory
        return get_gridded_trajectory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
