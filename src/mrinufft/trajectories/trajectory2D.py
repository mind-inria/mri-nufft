"""Backward-compatibility shim for trajectory2D functions.

All 2D trajectory initializations have moved to the inits/ submodule.
"""

from mrinufft.trajectories.inits.cones import initialize_2D_cones
from mrinufft.trajectories.inits.lissajous import (
    initialize_2D_lissajous,
    initialize_2D_polar_lissajous,
)
from mrinufft.trajectories.inits.propeller import initialize_2D_propeller
from mrinufft.trajectories.inits.radial import initialize_2D_radial
from mrinufft.trajectories.inits.rings import initialize_2D_rings
from mrinufft.trajectories.inits.rosette import initialize_2D_rosette
from mrinufft.trajectories.inits.spiral import (
    initialize_2D_fibonacci_spiral,
    initialize_2D_spiral,
    initialize_2D_vds_spiral,
)
from mrinufft.trajectories.inits.waves import (
    initialize_2D_sinusoide,
    initialize_2D_waves,
)
