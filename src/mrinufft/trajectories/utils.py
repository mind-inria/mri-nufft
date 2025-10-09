"""Utility functions in general."""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from numbers import Real
from typing import Any, ClassVar
from typing import Literal

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
        return super().__getattribute__(name.upper())


class FloatEnum(float, Enum, metaclass=CaseInsensitiveEnumMeta):
    """An Enum for float that is case insensitive for ist attributes."""

    pass


class StrEnum(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """An Enum for str that is case insensitive for its attributes."""

    pass


class Gammas(FloatEnum):
    """Enumerate gyromagnetic ratios for common nuclei in MR."""

    # Values in Hz/T
    HYDROGEN = 42_576_000
    HELIUM = 32_434_000
    CARBON = 10_708_000
    OXYGEN = 5_772_000
    FLUORINE = 40_078_000
    SODIUM = 11_262_000
    PHOSPHOROUS = 17_235_000
    XENON = 11_777_000

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


#############################
# Hardware and Acquisition  #
#############################


class SI:
    """SI prefixes."""

    giga = 1e9
    mega = 1e6
    kilo = 1e3
    hecto = 100
    deca = 10
    deci = 0.1
    centi = 0.01
    milli = 0.001
    micro = 1e-6
    nano = 1e-9

    gauss = 1e-4  # T to Gauss conversion factor


@dataclass(frozen=True)
class Hardware:
    """
    Hardware configuration for MRI sequences.

    Parameters
    ----------
    gmax : float
        Maximum gradient amplitude in T/m. Defaults to 40 mT/m.
    smax : float
        Maximum slew rate in T/m/s. Defaults to 200 T/m/s.
    n_coils : int
        Number of coils used in the MRI system. Defaults to 8.
    min_dwell_time : float
        Minimum ADC dwell time in seconds. Defaults to 1 ns.
    grad_raster_time : float
        Gradient raster time in seconds. Defaults to 10 us.
    field_strength : float
        Magnetic field strength in Tesla. Defaults to 3.0 T.

    Attributes
    ----------
    raster_time : float
        The gradient raster time in seconds, which is the same as `grad_raster_time`.

    Notes
    -----
    The `Hardware` class encapsulates the hardware constraints for MRI sequences,
    including the maximum gradient amplitude, slew rate, number of coils, dwell time,
    gradient raster time, and magnetic field strength.
    It is designed to be used in conjunction with the `Acquisition` class, which defines
    acquisition parameters such as field of view, image size, and gyromagnetic ratio.

    The default values for the parameters are set to typical values used in MRI systems.
    These values can be adjusted based on the specific hardware being used for MRI
    acquisition.

    For convenience, we define several common hardware configurations,

    """

    gmax: float = 40 * SI.milli  # Maximum gradient amplitude in T/m
    smax: float = 200  # T/m/s
    n_coils: int = 32
    min_dwell_time: float = 1 * SI.nano  # s
    grad_raster_time: float = 10 * SI.micro  # s
    field_strength: float = 3.0  # Tesla

    @property
    def raster_time(self) -> float:
        """Alias of grad_raster_time."""
        return self.grad_raster_time


# fmt: off
class SIEMENS:
    """Common hardware configurations for Siemens MRI systems."""

    TERRA          = Hardware(gmax=80*SI.milli, smax=200, field_strength=7)
    TERRAX         = Hardware(gmax=200*SI.milli, smax=200, field_strength=7)
    TERRAX_IMPULSE = Hardware(gmax=200*SI.milli, smax=900, field_strength=7)
    PRISMA         = Hardware(gmax=40*SI.milli, smax=200, field_strength=3)
    CIMA           = Hardware(gmax=200*SI.milli, smax=200, field_strength=3)
    CIMAX          = Hardware(gmax=200*SI.milli, smax=200, field_strength=3)

# fmt: on


@dataclass(frozen=True)
class Acquisition:
    """
    Acquisition configuration for MRI sequences.

    Parameters
    ----------
    fov : tuple[float, float, float]
        Field of View in meters (x, y, z).
    img_size : tuple[int, int, int]
        Image size in pixels (x, y, z).
    hardware : Hardware
        Hardware configuration for the acquisition.
    gamma : Gammas, optional
        Gyromagnetic ratio in Hz/T for the nucleus being imaged.
        Defaults to `Gammas.HYDROGEN`.
    adc_dwell_time: float
        Time resolution for the ADC, in seconds. default to 5us.
    norm_factor : float, optional
        Normalization factor for the trajectory. Defaults to 0.5.

    Attributes
    ----------
    default : ClassVar[Acquisition]
        The default acquisition configuration used if none is specified.
        You can set it using the `set_default` class method.

    Notes
    -----
    The `Acquisition` class encapsulates the parameters needed for MRI acquisition,
    including the field of view, image size, hardware specifications, and gyromagnetic
    ratio.

    It is designed to be used in conjunction with the `Hardware` class, which defines
    the hardware constraints such as maximum gradient amplitude and slew rate.
    The `default` class variable holds the default acquisition configuration, which can
    be set using the `set_default` method. This allows for easy access to a standard
    acquisition configuration without needing to instantiate a new `Acquisition` object
    each time.

    """

    fov: tuple[float, float, float]  # Field of View in m
    img_size: tuple[int, int, int]  # Image size in pixels
    hardware: Hardware = SIEMENS.TERRA  # Hardware configuration
    gamma: Gammas = Gammas.HYDROGEN  # Hz/T
    adc_dwell_time: float = 5 * SI.micro  # us
    norm_factor: float = 0.5

    default: ClassVar[Acquisition]
    _old_default: Acquisition | None = field(default=None, init=False)

    def __post_init__(self):
        """Validate parameters after initialization."""
        if isinstance(self.fov, Real):
            self.fov = (self.fov, self.fov, self.fov)
        if isinstance(self.img_size, int):
            self.img_size = (self.img_size, self.img_size, self.img_size)

        if not (2 <= len(self.fov) <= 3):
            raise ValueError("fov must be a tuple of 3 elements (x, y, z).")
        if not (2 <= len(self.img_size) <= 3):
            raise ValueError("img_size must be a tuple of 3 elements (x, y, z).")
        if any(s <= 0 for s in self.img_size):
            raise ValueError("img_size must contain positive integers.")
        if any(f <= 0 for f in self.fov):
            raise ValueError("fov must contain positive values.")
        if self.adc_dwell_time < self.hardware.min_dwell_time:
            raise ValueError(
                f"adc_dwell_time ({self.adc_dwell_time}s) "
                f"must be >= hardware.min_dwell_time ({self.hardware.min_dwell_time}s)."
            )

    def set_default(self) -> Acquisition:
        """Make the current acquisition configuration the default."""
        Acquisition.default = self
        return self

    def __getattr__(self, name):
        """Allow access to hardware attributes directly from Acquisition."""
        return getattr(self.hardware, name)

    @property
    def res(self) -> tuple[float, ...]:
        """Resolution in meters."""
        return tuple(
            fov / size for fov, size in zip(self.fov, self.img_size, strict=True)
        )

    @property
    def kmax(self) -> tuple[float, ...]:
        """Maximum k-space value in 1/m."""
        return tuple(
            0.5 * size / fov for fov, size in zip(self.fov, self.img_size, strict=True)
        )

    # Context Manager to use temporary new default.
    def __enter__(self) -> Acquisition:
        """Enter Context Manager with new default."""
        object.__setattr__(
            self, "_old_default", deepcopy(Acquisition.default)
        )  # bypass frozen
        self.set_default()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit Context Manager and reset default."""
        self._old_default.set_default()
        object.__setattr__(self, "_old_default", None)  # bypass frozen


# Create a default acquisition.
Acquisition.default = Acquisition(
    fov=(0.256, 0.256, 0.256), img_size=(256, 256, 256), hardware=Hardware()
)

###############
# CONSTRAINTS #
###############


def normalize_trajectory(
    trajectory: NDArray,
    acq: Acquisition | None = None,
) -> NDArray:
    """Normalize an un-normalized/natural trajectory for NUFFT use.

    Parameters
    ----------
    trajectory : NDArray
        Un-normalized trajectory consisting of k-space coordinates in 2D or 3D.
    acq : Acquisition, optional
        Acquisition configuration to use.
        If `None`, the default acquisition is used.

    Returns
    -------
    trajectory : NDArray
        Normalized trajectory corresponding to `trajectory` input.
    """
    acq = acq or Acquisition.default
    return trajectory * acq.norm_factor * (2 * np.array(acq.res))


def unnormalize_trajectory(
    trajectory: NDArray,
    acq: Acquisition | None = None,
) -> NDArray:
    """Un-normalize a NUFFT-normalized trajectory.

    Parameters
    ----------
    trajectory : NDArray
        Normalized trajectory consisting of k-space coordinates in 2D or 3D.
    acq : Acquisition, optional
        Acquisition configuration to use.
        If `None`, the default acquisition is used.

    Returns
    -------
    trajectory : NDArray
        Un-normalized trajectory corresponding to `trajectory` input.
    """
    acq = acq or Acquisition.default
    return (
        trajectory / acq.norm_factor / (2 * np.array(acq.res)[: trajectory.shape[-1]])
    )


def convert_trajectory_to_gradients(
    trajectory: NDArray,
    acq: Acquisition | None = None,
    get_final_positions: bool = False,
) -> tuple[NDArray, ...]:
    """Derive a normalized trajectory over time to provide gradients.

    Parameters
    ----------
    trajectory : NDArray
        Normalized trajectory consisting of k-space coordinates in 2D or 3D.
    acq : Acquisition, optional
        Acquisition configuration to use for normalization.
        If `None`, the default acquisition is used.
    get_final_positions : bool, optional
        If `True`, return the final positions in k-space.
        The default is `False`.

    Returns
    -------
    gradients : NDArray
        Gradients corresponding to `trajectory` in T/m
    """
    acq = acq or Acquisition.default
    # Un-normalize the trajectory from NUFFT usage
    trajectory = unnormalize_trajectory(trajectory, acq)

    # Compute gradients and starting positions
    gradients = np.diff(trajectory, axis=1) / acq.gamma / acq.raster_time
    initial_positions = trajectory[:, 0, :]
    if get_final_positions:
        return gradients, initial_positions, trajectory[:, -1, :]
    return gradients, initial_positions


def convert_gradients_to_trajectory(
    gradients: NDArray,
    initial_positions: NDArray | None = None,
    acq: Acquisition | None = None,
) -> NDArray:
    """Integrate gradients over time to provide a normalized trajectory.

    Parameters
    ----------
    gradients : NDArray
        Gradients over 2 or 3 directions in T/m
    initial_positions: NDArray, optional
        Positions in k-space at the beginning of the readout window.
        The default is `None`.
    acq : Acquisition, optional
        Acquisition configuration to use for normalization.
        If `None`, the default acquisition is used.

    Returns
    -------
    trajectory : NDArray
        Normalized trajectory corresponding to `gradients`.
    """
    # Handle no initial positions
    acq = acq or Acquisition.default
    if initial_positions is None:
        initial_positions = np.zeros((gradients.shape[0], gradients.shape[-1]))

    # Prepare and integrate gradients
    trajectory = gradients * acq.gamma * acq.raster_time
    trajectory = np.concatenate([initial_positions[:, None, :], trajectory], axis=1)
    trajectory = np.cumsum(trajectory, axis=1)

    # Normalize the trajectory for NUFFT usage
    trajectory = normalize_trajectory(trajectory, acq)
    return trajectory


def convert_gradients_to_slew_rates(
    gradients: NDArray, acq: Acquisition | None = None, raster_time: float | None = None
) -> tuple[NDArray, NDArray]:
    """Derive the gradients over time to provide slew rates.

    Parameters
    ----------
    gradients : NDArray
        Gradients over 2 or 3 directions.
    acq : Acquisition, optional
        Acquisition configuration to use.
        If `None`, the default acquisition is used.
    raster_time: float
        Raster time in seconds

    Returns
    -------
    slewrates : NDArray
        Slew rates corresponding to `gradients`.
    initial_gradients : NDArray
        Gradients at the beginning of the readout window.
    """
    # Compute slew rates and starting gradients
    if isinstance(acq, float) and raster_time is None:
        raster_time = acq
    elif isinstance(acq, Acquisition) and raster_time is None:
        raster_time = acq.raster_time
    elif raster_time is None and acq is None:
        raster_time = Acquisition.default.raster_time
    else:
        raise ValueError("incompatible acquisition and raster_time")
    slewrates = np.diff(gradients, axis=1) / raster_time
    initial_gradients = gradients[:, 0, :]
    return slewrates, initial_gradients


def convert_slew_rates_to_gradients(
    slewrates: NDArray,
    initial_gradients: NDArray | None = None,
    acq: Acquisition | None = None,
) -> NDArray:
    """Integrate slew rates over time to provide gradients.

    Parameters
    ----------
    slewrates : NDArray
        Slew rates over 2 or 3 directions.
    initial_gradients: NDArray, optional
        Gradients at the beginning of the readout window.
        The default is `None`.
    acq : Acquisition, optional
        Acquisition configuration to use for normalization.
        If `None`, the default acquisition is used.

    Returns
    -------
    gradients : NDArray
        Gradients corresponding to `slewrates`.
    """
    # Handle no initial gradients
    acq = acq or Acquisition.default
    if initial_gradients is None:
        initial_gradients = np.zeros((slewrates.shape[0], 1, slewrates.shape[-1]))

    # Prepare and integrate slew rates
    gradients = slewrates * acq.raster_time
    gradients = np.concatenate([initial_gradients[:, None, :], gradients], axis=1)
    gradients = np.cumsum(gradients, axis=1)
    return gradients


def compute_gradients_and_slew_rates(
    trajectory: NDArray,
    acq: Acquisition | None = None,
) -> tuple[NDArray, NDArray]:
    """Compute the gradients and slew rates from a normalized trajectory.

    Parameters
    ----------
    trajectory : NDArray
        Normalized trajectory consisting of k-space coordinates in 2D or 3D.
    acq : Acquisition, optional
        Acquisition configuration to use for normalization.
        If `None`, the default acquisition is used.

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
        acq,
    )

    # Convert gradients to slew rates
    slewrates, _ = convert_gradients_to_slew_rates(gradients, acq)

    return gradients, slewrates


def check_hardware_constraints(
    gradients: NDArray,
    slewrates: NDArray,
    acq: Acquisition | None = None,
    order: float | Literal["fro", "nuc"] | None = None,
) -> tuple[bool, float, float]:
    """Check if a trajectory satisfies the gradient hardware constraints.

    Parameters
    ----------
    gradients : NDArray
        Gradients to check
    slewrates: NDArray
        Slewrates to check
    acq : Acquisition, optional
        Acquisition configuration to use for checking.
        If `None`, the default acquisition is used.

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
    acq = acq or Acquisition.default

    max_grad = np.max(np.linalg.norm(gradients, axis=-1, ord=order))
    max_slew = np.max(np.linalg.norm(slewrates, axis=-1, ord=order))
    return (max_grad < acq.gmax) and (max_slew < acq.smax), max_grad, max_slew


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
