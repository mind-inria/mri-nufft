"""Utility functions in general."""

from enum import Enum, EnumMeta
from numbers import Real
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

#############
# CONSTANTS #
#############

KMAX = 0.5

DEFAULT_RESOLUTION = 6e-4  # m, i.e. 0.6 mm isotropic
DEFAULT_RASTER_TIME = 10e-3  # ms

DEFAULT_GMAX = 0.04  # T/m
DEFAULT_SMAX = 0.1  # T/m/ms


#########
# ENUMS #
#########


class CaseInsensitiveEnumMeta(EnumMeta):
    """A case-insensitive EnumMeta."""

    def __getitem__(self, name: str) -> Enum:
        """Allow ``MyEnum['Member'] == MyEnum['MEMBER']`` ."""
        return super().__getitem__(name.upper())

    def __getattr__(self, name: str) -> Any:  # noqa ANN401
        """Allow ``MyEnum.Member == MyEnum.MEMBER`` ."""
        return super().__getattr__(name.upper())


class FloatEnum(float, Enum, metaclass=CaseInsensitiveEnumMeta):
    """An Enum for float that is case insensitive for ist attributes."""

    pass


class StrEnum(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """An Enum for str that is case insensitive for its attributes."""

    pass


class Gammas(FloatEnum):
    """Enumerate gyromagnetic ratios for common nuclei in MR."""

    # Values in kHz/T
    HYDROGEN = 42576
    HELIUM = 32434
    CARBON = 10708
    OXYGEN = 5772
    FLUORINE = 40078
    SODIUM = 11262
    PHOSPHOROUS = 17235
    XENON = 11777

    # Aliases
    H = H1 = PROTON = HYDROGEN
    He = He3 = HELIUM
    C = C13 = CARBON
    O = O17 = OXYGEN  # noqa: E741
    F = F19 = FLUORINE
    Na = Na23 = SODIUM
    P = P31 = PHOSPHOROUS
    X = X129 = XENON


class Spirals(FloatEnum):
    """Enumerate algebraic spiral types."""

    ARCHIMEDES = 1
    ARITHMETIC = ARCHIMEDES
    GALILEAN = 2
    GALILEO = GALILEAN
    FERMAT = 0.5
    PARABOLIC = FERMAT
    HYPERBOLIC = -1
    LITHUUS = -0.5


class NormShapes(FloatEnum):
    """Enumerate shape norms."""

    L1 = 1
    L2 = 2
    LINF = np.inf
    SQUARE = LINF
    CUBE = LINF
    CIRCLE = L2
    SPHERE = L2
    DIAMOND = L1
    OCTAHEDRON = L1


class Tilts(StrEnum):
    r"""Enumerate available tilts.

    Notes
    -----
    The following values are accepted for the tilt name, with :math:`N` the number of
    partitions:

    - "none": no tilt
    - "uniform": uniform tilt: 2:math:`\pi / N`
    - "intergaps": :math:`\pi/2N`
    - "inverted": inverted tilt :math:`\pi/N + \pi`
    - "golden": tilt of the golden angle :math:`\pi(3-\sqrt{5})`
    - "mri golden": tilt of the golden angle used in MRI :math:`\pi(\sqrt{5}-1)/2`

    """

    UNIFORM = "uniform"
    NONE = "none"
    INTERGAPS = "intergaps"
    INVERTED = "inverted"
    GOLDEN = "golden"
    MRI_GOLDEN = "mri-golden"
    MRI = MRI_GOLDEN


class Packings(StrEnum):
    """Enumerate available packing method for shots.

    It is mostly used for wave-CAIPI trajectory

    See Also
    --------
    mrinufft.trajectories.trajectories3D.initialize_3D_wave_caipi

    """

    RANDOM = "random"
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    HEXAGON = "hexagon"
    SQUARE = "square"
    FIBONACCI = "fibonacci"

    # Aliases
    CIRCULAR = CIRCLE
    TRIANGULAR = TRIANGLE
    HEXAGONAL = HEXAGON
    UNIFORM = RANDOM
    SPIRAL = FIBONACCI


#############################
# Variable Density Sampling #
#############################


class VDSorder(StrEnum):
    """Available ordering for variable density sampling."""

    CENTER_OUT = "center-out"
    RANDOM = "random"
    TOP_DOWN = "top-down"


class VDSpdf(StrEnum):
    """Available law for variable density sampling."""

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    EQUISPACED = "equispaced"


###############
# CONSTRAINTS #
###############


def normalize_trajectory(
    trajectory: NDArray,
    norm_factor: float = KMAX,
    resolution: float | NDArray = DEFAULT_RESOLUTION,
) -> NDArray:
    """Normalize an un-normalized/natural trajectory for NUFFT use.

    Parameters
    ----------
    trajectory : NDArray
        Un-normalized trajectory consisting of k-space coordinates in 2D or 3D.
    norm_factor : float, optional
        Trajectory normalization factor, by default KMAX.
    resolution : float, np.array, optional
        Resolution of MR image in meters, isotropic as `int`
        or anisotropic as `np.array`.
        The default is DEFAULT_RESOLUTION.

    Returns
    -------
    trajectory : NDArray
        Normalized trajectory corresponding to `trajectory` input.
    """
    return trajectory * norm_factor * (2 * resolution)


def unnormalize_trajectory(
    trajectory: NDArray,
    norm_factor: float = KMAX,
    resolution: float | NDArray = DEFAULT_RESOLUTION,
) -> NDArray:
    """Un-normalize a NUFFT-normalized trajectory.

    Parameters
    ----------
    trajectory : NDArray
        Normalized trajectory consisting of k-space coordinates in 2D or 3D.
    norm_factor : float, optional
        Trajectory normalization factor, by default KMAX.
    resolution : float, np.array, optional
        Resolution of MR image in meters, isotropic as `int`
        or anisotropic as `np.array`.
        The default is DEFAULT_RESOLUTION.

    Returns
    -------
    trajectory : NDArray
        Un-normalized trajectory corresponding to `trajectory` input.
    """
    return trajectory / norm_factor / (2 * resolution)


def convert_trajectory_to_gradients(
    trajectory: NDArray,
    norm_factor: float = KMAX,
    resolution: float | NDArray = DEFAULT_RESOLUTION,
    raster_time: float = DEFAULT_RASTER_TIME,
    gamma: float = Gammas.HYDROGEN,
    get_final_positions: bool = False,
) -> tuple[NDArray, ...]:
    """Derive a normalized trajectory over time to provide gradients.

    Parameters
    ----------
    trajectory : NDArray
        Normalized trajectory consisting of k-space coordinates in 2D or 3D.
    norm_factor : float, optional
        Trajectory normalization factor, by default KMAX.
    resolution : float, np.array, optional
        Resolution of MR image in meters, isotropic as `int`
        or anisotropic as `np.array`.
        The default is DEFAULT_RESOLUTION.
    raster_time : float, optional
        Amount of time between the acquisition of two
        consecutive samples in ms.
        The default is `DEFAULT_RASTER_TIME`.
    gamma : float, optional
        Gyromagnetic ratio of the selected nucleus in kHz/T
        The default is Gammas.HYDROGEN.
    get_final_positions : bool, optional
        If `True`, return the final positions in k-space.
        The default is `False`.

    Returns
    -------
    gradients : NDArray
        Gradients corresponding to `trajectory`.
    """
    # Un-normalize the trajectory from NUFFT usage
    trajectory = unnormalize_trajectory(trajectory, norm_factor, resolution)

    # Compute gradients and starting positions
    gradients = np.diff(trajectory, axis=1) / gamma / raster_time
    initial_positions = trajectory[:, 0, :]
    if get_final_positions:
        return gradients, initial_positions, trajectory[:, -1, :]
    return gradients, initial_positions


def convert_gradients_to_trajectory(
    gradients: NDArray,
    initial_positions: NDArray | None = None,
    norm_factor: float = KMAX,
    resolution: float | NDArray = DEFAULT_RESOLUTION,
    raster_time: float = DEFAULT_RASTER_TIME,
    gamma: float = Gammas.HYDROGEN,
) -> NDArray:
    """Integrate gradients over time to provide a normalized trajectory.

    Parameters
    ----------
    gradients : NDArray
        Gradients over 2 or 3 directions.
    initial_positions: NDArray, optional
        Positions in k-space at the beginning of the readout window.
        The default is `None`.
    norm_factor : float, optional
        Trajectory normalization factor, by default KMAX.
    resolution : float, np.array, optional
        Resolution of MR image in meters, isotropic as `int`
        or anisotropic as `np.array`.
        The default is DEFAULT_RESOLUTION.
    raster_time : float, optional
        Amount of time between the acquisition of two
        consecutive samples in ms.
        The default is `DEFAULT_RASTER_TIME`.
    gamma : float, optional
        Gyromagnetic ratio of the selected nucleus in kHz/T
        The default is Gammas.HYDROGEN.

    Returns
    -------
    trajectory : NDArray
        Normalized trajectory corresponding to `gradients`.
    """
    # Handle no initial positions
    if initial_positions is None:
        initial_positions = np.zeros((gradients.shape[0], 1, gradients.shape[-1]))

    # Prepare and integrate gradients
    trajectory = gradients * gamma * raster_time
    trajectory = np.concatenate([initial_positions[:, None, :], trajectory], axis=1)
    trajectory = np.cumsum(trajectory, axis=1)

    # Normalize the trajectory for NUFFT usage
    trajectory = normalize_trajectory(trajectory, norm_factor, resolution)
    return trajectory


def convert_gradients_to_slew_rates(
    gradients: NDArray,
    raster_time: float = DEFAULT_RASTER_TIME,
) -> tuple[NDArray, NDArray]:
    """Derive the gradients over time to provide slew rates.

    Parameters
    ----------
    gradients : NDArray
        Gradients over 2 or 3 directions.
    raster_time : float, optional
        Amount of time between the acquisition of two
        consecutive samples in ms.
        The default is `DEFAULT_RASTER_TIME`.

    Returns
    -------
    slewrates : NDArray
        Slew rates corresponding to `gradients`.
    initial_gradients : NDArray
        Gradients at the beginning of the readout window.
    """
    # Compute slew rates and starting gradients
    slewrates = np.diff(gradients, axis=1) / raster_time
    initial_gradients = gradients[:, 0, :]
    return slewrates, initial_gradients


def convert_slew_rates_to_gradients(
    slewrates: NDArray,
    initial_gradients: NDArray | None = None,
    raster_time: float = DEFAULT_RASTER_TIME,
) -> NDArray:
    """Integrate slew rates over time to provide gradients.

    Parameters
    ----------
    slewrates : NDArray
        Slew rates over 2 or 3 directions.
    initial_gradients: NDArray, optional
        Gradients at the beginning of the readout window.
        The default is `None`.
    raster_time : float, optional
        Amount of time between the acquisition of two
        consecutive samples in ms.
        The default is `DEFAULT_RASTER_TIME`.

    Returns
    -------
    gradients : NDArray
        Gradients corresponding to `slewrates`.
    """
    # Handle no initial gradients
    if initial_gradients is None:
        initial_gradients = np.zeros((slewrates.shape[0], 1, slewrates.shape[-1]))

    # Prepare and integrate slew rates
    gradients = slewrates * raster_time
    gradients = np.concatenate([initial_gradients[:, None, :], gradients], axis=1)
    gradients = np.cumsum(gradients, axis=1)
    return gradients


def compute_gradients_and_slew_rates(
    trajectory: NDArray,
    norm_factor: float = KMAX,
    resolution: float | NDArray = DEFAULT_RESOLUTION,
    raster_time: float = DEFAULT_RASTER_TIME,
    gamma: float = Gammas.HYDROGEN,
) -> tuple[NDArray, NDArray]:
    """Compute the gradients and slew rates from a normalized trajectory.

    Parameters
    ----------
    trajectory : NDArray
        Normalized trajectory consisting of k-space coordinates in 2D or 3D.
    norm_factor : float, optional
        Trajectory normalization factor, by default KMAX.
    resolution : float, np.array, optional
        Resolution of MR image in meters, isotropic as `int`
        or anisotropic as `np.array`.
        The default is DEFAULT_RESOLUTION.
    raster_time : float, optional
        Amount of time between the acquisition of two
        consecutive samples in ms.
        The default is `DEFAULT_RASTER_TIME`.
    gamma : float, optional
        Gyromagnetic ratio of the selected nucleus in kHz/T
        The default is Gammas.HYDROGEN.

    Returns
    -------
    gradients : NDArray
        Gradients corresponding to `trajectory`.
    slewrates : NDArray
        Slew rates corresponding to `trajectory` gradients.
    """
    # Convert normalized trajectory to gradients
    gradients, _ = convert_trajectory_to_gradients(
        trajectory,
        norm_factor=norm_factor,
        resolution=resolution,
        raster_time=raster_time,
        gamma=gamma,
    )

    # Convert gradients to slew rates
    slewrates, _ = convert_gradients_to_slew_rates(gradients, raster_time)

    return gradients, slewrates


def check_hardware_constraints(
    gradients: NDArray,
    slewrates: NDArray,
    gmax: float = DEFAULT_GMAX,
    smax: float = DEFAULT_SMAX,
    order: int | str | None = None,
) -> tuple[bool, float, float]:
    """Check if a trajectory satisfies the gradient hardware constraints.

    Parameters
    ----------
    gradients : NDArray
        Gradients to check
    slewrates: NDArray
        Slewrates to check
    gmax : float, optional
        Maximum gradient amplitude in T/m. The default is DEFAULT_GMAX.
    smax : float, optional
        Maximum slew rate in T/m/ms. The default is DEFAULT_SMAX.
    order : int or str, optional
        Norm order defining how the constraints are checked,
        typically 2 or `np.inf`, following the `numpy.linalg.norm`
        conventions on parameter `ord`.
        The default is None.

    Returns
    -------
    bool
        True if the trajectory satisfies the constraints, False otherwise.
    float
        Maximum gradient amplitude in T/m.
    float
        Maximum slew rate in T/m/ms.
    """
    max_grad = np.max(np.linalg.norm(gradients, axis=-1, ord=order))
    max_slew = np.max(np.linalg.norm(slewrates, axis=-1, ord=order))
    return (max_grad < gmax) and (max_slew < smax), max_grad, max_slew


###########
# OPTIONS #
###########


def initialize_tilt(tilt: str | float | None, nb_partitions: int = 1) -> float:
    r"""Initialize the tilt angle.

    Parameters
    ----------
    tilt : str | float | None
        Tilt angle in rad or name of the tilt.
    nb_partitions : int, optional
        Number of partitions. The default is 1.

    Returns
    -------
    float
        Tilt angle in rad.

    Raises
    ------
    NotImplementedError
        If the tilt name is unknown.

    See Also
    --------
    Tilts

    """
    if isinstance(tilt, Real):
        return tilt
    elif tilt is None or tilt == Tilts.NONE:
        return 0
    elif tilt == Tilts.UNIFORM:
        return 2 * np.pi / nb_partitions
    elif tilt == Tilts.INTERGAPS:
        return np.pi / nb_partitions / 2
    elif tilt == Tilts.INVERTED:
        return np.pi / nb_partitions + np.pi
    elif tilt == Tilts.GOLDEN:
        return np.pi * (3 - np.sqrt(5))
    elif tilt == Tilts.MRI_GOLDEN:
        return np.pi * (np.sqrt(5) - 1) / 2
    else:
        raise NotImplementedError(f"Unknown tilt name: {tilt}")


def initialize_algebraic_spiral(spiral: str | float) -> float:
    """Initialize the algebraic spiral type.

    Parameters
    ----------
    spiral : str | float
        Spiral type or spiral power value.

    Returns
    -------
    float
        Spiral power value.
    """
    if isinstance(spiral, Real):
        return float(spiral)
    return Spirals[str(spiral)]


def initialize_shape_norm(shape: str | float) -> float:
    """Initialize the norm for a given shape.

    Parameters
    ----------
    shape : str | float
        Shape name or p-norm value.

    Returns
    -------
    float
        Shape p-norm value.
    """
    if isinstance(shape, Real):
        return float(shape)
    return NormShapes[str(shape)]
