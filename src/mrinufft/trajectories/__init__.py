"""Collection of trajectories and tools used for non-Cartesian MRI.

See also the trajectories examples: :ref:`sphx_glr_generated_autoexamples_trajectories`
"""

##########
# maths  #
##########

from mrinufft.trajectories.maths.fibonacci import get_closest_fibonacci_number
from mrinufft.trajectories.maths.rotations import R2D, Ra, Rv, Rx, Ry, Rz

__all__ = [
    "R2D",
    "Ra",
    "Rv",
    "Rx",
    "Ry",
    "Rz",
    "compute_coprime_factors",
    "get_closest_fibonacci_number",
]


#########
# inits #
#########

from .inits.cones import initialize_2D_cones, initialize_3D_cones
from .inits.eccentric import initialize_2D_eccentric, initialize_3D_eccentric
from .inits.floret import initialize_3D_floret
from .inits.fmri import initialize_3D_repi, initialize_3D_turbine
from .inits.lissajous import initialize_2D_lissajous, initialize_2D_polar_lissajous
from .inits.propeller import initialize_2D_propeller
from .inits.radial import (
    initialize_2D_radial,
    initialize_3D_golden_means_radial,
    initialize_3D_park_radial,
    initialize_3D_phyllotaxis_radial,
    initialize_3D_wong_radial,
)
from .inits.random_walk import initialize_2D_random_walk, initialize_3D_random_walk
from .inits.rings import initialize_2D_rings
from .inits.rosette import initialize_2D_rosette
from .inits.seiffert import initialize_3D_seiffert_shells, initialize_3D_seiffert_spiral
from .inits.shells import initialize_3D_annular_shells, initialize_3D_helical_shells
from .inits.spiral import (
    initialize_2D_fibonacci_spiral,
    initialize_2D_spiral,
    initialize_2D_vds_spiral,
)
from .inits.travelling_salesman import (
    initialize_2D_travelling_salesman,
    initialize_3D_travelling_salesman,
)
from .inits.wave_caipi import initialize_3D_wave_caipi
from .inits.waves import initialize_2D_sinusoide, initialize_2D_waves

__all__ += [
    "initialize_2D_cones",
    "initialize_2D_eccentric",
    "initialize_2D_fibonacci_spiral",
    "initialize_2D_lissajous",
    "initialize_2D_polar_lissajous",
    "initialize_2D_propeller",
    "initialize_2D_radial",
    "initialize_2D_random_walk",
    "initialize_2D_rings",
    "initialize_2D_rosette",
    "initialize_2D_sinusoide",
    "initialize_2D_spiral",
    "initialize_2D_travelling_salesman",
    "initialize_2D_vds_spiral",
    "initialize_2D_waves",
    "initialize_3D_annular_shells",
    "initialize_3D_cones",
    "initialize_3D_eccentric",
    "initialize_3D_floret",
    "initialize_3D_golden_means_radial",
    "initialize_3D_helical_shells",
    "initialize_3D_park_radial",
    "initialize_3D_phyllotaxis_radial",
    "initialize_3D_random_walk",
    "initialize_3D_repi",
    "initialize_3D_seiffert_shells",
    "initialize_3D_seiffert_spiral",
    "initialize_3D_travelling_salesman",
    "initialize_3D_turbine",
    "initialize_3D_wave_caipi",
    "initialize_3D_wong_radial",
]

#########
# tools #
#########

from .tools.caipi import (
    get_grappa_caipi_positions,
    get_packing_spacing_positions,
)
from .tools.epi import epify, unepify
from .tools.functional import shellify, stack_spherically
from .tools.random import get_random_loc_1d, stack_random
from .tools.transforms import (
    conify,
    duplicate_along_axes,
    oversample,
    precess,
    radialize_center,
    rotate,
    stack,
)
from .tools.winding import prewind, rewind

__all__ += [
    "conify",
    "duplicate_along_axes",
    "epify",
    "get_grappa_caipi_positions",
    "get_packing_spacing_positions",
    "get_random_loc_1d",
    "oversample",
    "precess",
    "prewind",
    "radialize_center",
    "rewind",
    "rotate",
    "shellify",
    "stack",
    "stack_random",
    "stack_spherically",
    "unepify",
]


from .gradients import (
    connect_gradient,
    get_prephasors_and_spoilers,
    min_length_connection,
    patch_center_anomaly,
)
from .projection import (
    GradientLinearProjection,
    GroupL2SoftThresholding,
    parameterize_by_arc_length,
    project_trajectory,
)
from .sampling import (
    create_chauffert_density,
    create_cutoff_decay_density,
    create_energy_density,
    create_fast_chauffert_density,
    create_polynomial_density,
    sample_from_density,
)
from .utils import (
    Acquisition,
    FloatEnum,
    Gammas,
    Hardware,
    NormShapes,
    Packings,
    SiemensGradient,
    Spirals,
    StrEnum,
    Tilts,
    VDSorder,
    VDSpdf,
    check_hardware_constraints,
    compute_gradients_and_slew_rates,
    convert_gradients_to_slew_rates,
    convert_gradients_to_trajectory,
    convert_slew_rates_to_gradients,
    convert_trajectory_to_gradients,
    initialize_algebraic_spiral,
    initialize_shape_norm,
    initialize_tilt,
    normalize_trajectory,
    unnormalize_trajectory,
)

__all__ += [
    "Acquisition",
    "FloatEnum",
    "Gammas",
    "GradientLinearProjection",
    "GroupL2SoftThresholding",
    "Hardware",
    "KMAX",
    "NormShapes",
    "Packings",
    "SiemensGradient",
    "Spirals",
    "StrEnum",
    "Tilts",
    "VDSorder",
    "VDSpdf",
    "check_hardware_constraints",
    "compute_gradients_and_slew_rates",
    "connect_gradient",
    "convert_gradients_to_slew_rates",
    "convert_gradients_to_trajectory",
    "convert_slew_rates_to_gradients",
    "convert_trajectory_to_gradients",
    "create_chauffert_density",
    "create_cutoff_decay_density",
    "create_energy_density",
    "create_fast_chauffert_density",
    "create_polynomial_density",
    "get_prephasors_and_spoilers",
    "initialize_algebraic_spiral",
    "initialize_shape_norm",
    "initialize_tilt",
    "linear_projection",
    "min_length_connection",
    "normalize_trajectory",
    "parameterize_by_arc_length",
    "patch_center_anomaly",
    "project_trajectory",
    "sample_from_density",
    "unnormalize_trajectory",
]
