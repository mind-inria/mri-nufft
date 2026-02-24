"""Trajectory manipulation tools."""

from mrinufft.trajectories.tools.transforms import (
    stack,
    rotate,
    precess,
    conify,
    oversample,
    duplicate_along_axes,
    radialize_center,
    _radialize_center_out,
    _radialize_in_out,
    _flip2center,
)
from mrinufft.trajectories.tools.epi import epify, unepify
from mrinufft.trajectories.tools.winding import prewind, rewind
from mrinufft.trajectories.tools.functional import stack_spherically, shellify
from mrinufft.trajectories.tools.random import get_random_loc_1d, stack_random
from mrinufft.trajectories.tools.caipi import (
    get_grappa_caipi_positions,
    get_packing_spacing_positions,
)

__all__ = [
    # transforms
    "stack",
    "rotate",
    "precess",
    "conify",
    "oversample",
    "duplicate_along_axes",
    "radialize_center",
    # epi
    "epify",
    "unepify",
    # winding
    "prewind",
    "rewind",
    # functional
    "stack_spherically",
    "shellify",
    # random
    "get_random_loc_1d",
    "stack_random",
    # caipi
    "get_grappa_caipi_positions",
    "get_packing_spacing_positions",
]
