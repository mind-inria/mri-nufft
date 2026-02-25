"""Backward-compatibility shim for trajectory3D functions.

All 3D trajectory initializations have moved to the inits/ submodule.
"""

from mrinufft.trajectories.inits.cones import initialize_3D_cones
from mrinufft.trajectories.inits.floret import initialize_3D_floret
from mrinufft.trajectories.inits.fmri import initialize_3D_repi, initialize_3D_turbine
from mrinufft.trajectories.inits.radial import (
    initialize_3D_golden_means_radial,
    initialize_3D_park_radial,
    initialize_3D_phyllotaxis_radial,
    initialize_3D_wong_radial,
)
from mrinufft.trajectories.inits.seiffert import (
    initialize_3D_seiffert_shells,
    initialize_3D_seiffert_spiral,
)
from mrinufft.trajectories.inits.shells import (
    initialize_3D_annular_shells,
    initialize_3D_helical_shells,
)
from mrinufft.trajectories.inits.wave_caipi import initialize_3D_wave_caipi
