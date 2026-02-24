"""Module containing all trajectories."""

from mrinufft.trajectories.inits.eccentric import (
    initialize_2D_eccentric,
    initialize_3D_eccentric,
)
from mrinufft.trajectories.inits.random_walk import (
    initialize_2D_random_walk,
    initialize_3D_random_walk,
)
from mrinufft.trajectories.inits.travelling_salesman import (
    initialize_2D_travelling_salesman,
    initialize_3D_travelling_salesman,
)
from mrinufft.trajectories.inits.radial import (
    initialize_2D_radial,
    initialize_3D_phyllotaxis_radial,
    initialize_3D_golden_means_radial,
    initialize_3D_wong_radial,
    initialize_3D_park_radial,
)
from mrinufft.trajectories.inits.propeller import initialize_2D_propeller
from mrinufft.trajectories.inits.rings import initialize_2D_rings
from mrinufft.trajectories.inits.spiral import (
    initialize_2D_spiral,
    initialize_2D_fibonacci_spiral,
    initialize_2D_vds_spiral,
)
from mrinufft.trajectories.inits.seiffert import (
    initialize_3D_seiffert_spiral,
    initialize_3D_seiffert_shells,
)
from mrinufft.trajectories.inits.cones import (
    initialize_2D_cones,
    initialize_3D_cones,
)
from mrinufft.trajectories.inits.waves import (
    initialize_2D_sinusoide,
    initialize_2D_waves,
)
from mrinufft.trajectories.inits.rosette import initialize_2D_rosette
from mrinufft.trajectories.inits.lissajous import (
    initialize_2D_lissajous,
    initialize_2D_polar_lissajous,
)
from mrinufft.trajectories.inits.floret import initialize_3D_floret
from mrinufft.trajectories.inits.wave_caipi import initialize_3D_wave_caipi
from mrinufft.trajectories.inits.shells import (
    initialize_3D_helical_shells,
    initialize_3D_annular_shells,
)
from mrinufft.trajectories.inits.fmri import (
    initialize_3D_turbine,
    initialize_3D_repi,
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
    # radial
    "initialize_2D_radial",
    "initialize_3D_phyllotaxis_radial",
    "initialize_3D_golden_means_radial",
    "initialize_3D_wong_radial",
    "initialize_3D_park_radial",
    # propeller
    "initialize_2D_propeller",
    # rings
    "initialize_2D_rings",
    # spiral
    "initialize_2D_spiral",
    "initialize_2D_fibonacci_spiral",
    "initialize_2D_vds_spiral",
    # seiffert
    "initialize_3D_seiffert_spiral",
    "initialize_3D_seiffert_shells",
    # cones
    "initialize_2D_cones",
    "initialize_3D_cones",
    # waves
    "initialize_2D_sinusoide",
    "initialize_2D_waves",
    # rosette
    "initialize_2D_rosette",
    # lissajous
    "initialize_2D_lissajous",
    "initialize_2D_polar_lissajous",
    # floret
    "initialize_3D_floret",
    # wave_caipi
    "initialize_3D_wave_caipi",
    # shells
    "initialize_3D_helical_shells",
    "initialize_3D_annular_shells",
    # fmri
    "initialize_3D_turbine",
    "initialize_3D_repi",
]
