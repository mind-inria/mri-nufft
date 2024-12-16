"""MRI-NUFFT.

MRI-NUFFT provides an easy to use Fourier operator for non-Cartesian
reconstruction.

Doing non-Cartesian MRI has never been so easy.
"""

from .operators import (
    get_operator,
    check_backend,
    list_backends,
    get_interpolators_from_fieldmap,
)

from .trajectories import (
    # trajectories
    initialize_2D_radial,
    initialize_2D_spiral,
    initialize_2D_fibonacci_spiral,
    initialize_2D_cones,
    initialize_2D_sinusoide,
    initialize_2D_propeller,
    initialize_2D_rosette,
    initialize_2D_rings,
    initialize_2D_polar_lissajous,
    initialize_2D_lissajous,
    initialize_2D_waves,
    initialize_2D_random_walk,
    initialize_2D_travelling_salesman,
    initialize_3D_phyllotaxis_radial,
    initialize_3D_golden_means_radial,
    initialize_3D_wong_radial,
    initialize_3D_park_radial,
    initialize_3D_cones,
    initialize_3D_floret,
    initialize_3D_wave_caipi,
    initialize_3D_seiffert_spiral,
    initialize_3D_helical_shells,
    initialize_3D_annular_shells,
    initialize_3D_seiffert_shells,
    initialize_3D_turbine,
    initialize_3D_repi,
    initialize_3D_random_walk,
    initialize_3D_travelling_salesman,
    # tools
    stack,
    rotate,
    precess,
    conify,
    stack_spherically,
    shellify,
    duplicate_along_axes,
    radialize_center,
    oversample,
    # densities
    sample_from_density,
    create_cutoff_decay_density,
    create_polynomial_density,
    create_energy_density,
    create_chauffert_density,
    create_fast_chauffert_density,
    # display
    displayConfig,
    display_2D_trajectory,
    display_3D_trajectory,
)

from .density import voronoi, cell_count, pipe, get_density

__all__ = [
    # nufft
    "get_operator",
    "check_backend",
    "list_backends",
    "get_interpolators_from_fieldmap",
    "voronoi",
    "cell_count",
    "pipe",
    "get_density",
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
    "stack",
    "rotate",
    "precess",
    "conify",
    "stack_spherically",
    "shellify",
    "duplicate_along_axes",
    "radialize_center",
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

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
