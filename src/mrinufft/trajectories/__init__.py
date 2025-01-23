"""Collection of trajectories and tools used for non-Cartesian MRI."""

from .display import display_2D_trajectory, display_3D_trajectory, displayConfig
from .gradients import patch_center_anomaly
from .inits import (
    initialize_2D_random_walk,
    initialize_2D_travelling_salesman,
    initialize_3D_random_walk,
    initialize_3D_travelling_salesman,
)
from .sampling import (
    create_chauffert_density,
    create_cutoff_decay_density,
    create_energy_density,
    create_fast_chauffert_density,
    create_polynomial_density,
    sample_from_density,
)
from .tools import (
    conify,
    duplicate_along_axes,
    oversample,
    precess,
    radialize_center,
    rotate,
    shellify,
    stack,
    stack_spherically,
)
from .trajectory2D import (
    initialize_2D_cones,
    initialize_2D_fibonacci_spiral,
    initialize_2D_lissajous,
    initialize_2D_polar_lissajous,
    initialize_2D_propeller,
    initialize_2D_radial,
    initialize_2D_rings,
    initialize_2D_rosette,
    initialize_2D_sinusoide,
    initialize_2D_spiral,
    initialize_2D_waves,
)
from .trajectory3D import (
    initialize_3D_annular_shells,
    initialize_3D_cones,
    initialize_3D_floret,
    initialize_3D_golden_means_radial,
    initialize_3D_helical_shells,
    initialize_3D_park_radial,
    initialize_3D_phyllotaxis_radial,
    initialize_3D_repi,
    initialize_3D_seiffert_shells,
    initialize_3D_seiffert_spiral,
    initialize_3D_turbine,
    initialize_3D_wave_caipi,
    initialize_3D_wong_radial,
)

from .tools import (
    stack_random,
    get_random_loc_1d,
)


__all__ = [
    # trajectories
    "initialize_2D_radial",
    "initialize_2D_spiral",
    "initialize_2D_fibonacci_spiral",
    "initialize_2D_cones",
    "initialize_2D_sinusoide",
    "initialize_2D_propeller",
    "initialize_2D_rosette",
    "initialize_2D_rings",
    "initialize_2D_polar_lissajous",
    "initialize_2D_lissajous",
    "initialize_2D_waves",
    "initialize_2D_random_walk",
    "initialize_2D_travelling_salesman",
    "initialize_3D_phyllotaxis_radial",
    "initialize_3D_golden_means_radial",
    "initialize_3D_wong_radial",
    "initialize_3D_park_radial",
    "initialize_3D_from_2D_expansion",
    "initialize_3D_cones",
    "initialize_3D_floret",
    "initialize_3D_wave_caipi",
    "initialize_3D_seiffert_spiral",
    "initialize_3D_helical_shells",
    "initialize_3D_annular_shells",
    "initialize_3D_seiffert_shells",
    "initialize_3D_turbine",
    "initialize_3D_repi",
    "initialize_3D_random_walk",
    "initialize_3D_travelling_salesman",
    # tools
    "get_random_loc_1d",
    "stack",
    "stack_random",
    "rotate",
    "precess",
    "conify",
    "stack_spherically",
    "shellify",
    "duplicate_along_axes",
    "radialize_center",
    "displayConfig",
    "display_2D_trajectory",
    "display_3D_trajectory",
    "patch_center_anomaly",
    "oversample",
    # densities
    "sample_from_density",
    "create_cutoff_decay_density",
    "create_polynomial_density",
    "create_energy_density",
    "create_chauffert_density",
    "create_fast_chauffert_density",
    # display
    "displayConfig",
    "display_2D_trajectory",
    "display_3D_trajectory",
]
