"""Utility functions in general."""

from __future__ import annotations

import re
from typing import ClassVar, Any
from dataclasses import dataclass
from enum import Enum, EnumMeta
from numbers import Real
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
        return super().__getattr__(name.upper())


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
    gmax: float = 40 * SI.milli  # Maximum gradient amplitude in T/m
    smax: float = 200  # T/m/s
    n_coils: int = 8
    dwell_time: float = 1 * SI.nano  # s
    grad_raster_time: float = 5 * SI.micro  # s
    field_strength: float = 3.0  # Tesla

    @property
    def raster_time(self) -> float:
        return self.grad_raster_time


class SIEMENS_HARDWARE(object):
    TERRAX = Hardware()
    PRISMA = ...
    CIMA = ...
    CIMAX = ...
    ISEULT = ...


@dataclass(frozen=True)
class Acquisition:
    default: ClassVar[Acquisition]

    fov: tuple[float, float, float]  # Field of View in m
    img_size: tuple[int, int, int]  # Image size in pixels
    hardware: Hardware
    gamma: Gammas = Gammas.HYDROGEN  # Hz/T
    oversampling: int = 1  # Oversampling factor for the ADC
    norm_factor: float = 0.5

    def set_default(self) -> Acquisition:
        """Make the current acquisition configuration the default."""
        Acquisition.default = self
        return self

    def __getattr__(self, name):
        # pass through attributes to the hardware object
        return getattr(self.hardware, name)

    @classmethod
    def __getattr__(cls, name):
        return getattr(cls.default, name)

    @property
    def res(self) -> tuple[float, ...]:
        """Resolution in meters."""
        return tuple(fov / size for fov, size in zip(self.fov, self.img_size))


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
        Acquisition configuration to use for normalization.
        If `None`, the default acquisition is used.

    Returns
    -------
    trajectory : NDArray
        Normalized trajectory corresponding to `trajectory` input.
    """
    acq = acq or Acquisition.default
    return trajectory * acq.norm_factor * (2 * acq.res)


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
        Acquisition configuration to use for un-normalization.
        If `None`, the default acquisition is used.
    Returns
    -------
    trajectory : NDArray
        Un-normalized trajectory corresponding to `trajectory` input.
    """
    acq = acq or Acquisition.default
    return trajectory / acq.norm_factor / (2 * acq.resolution)


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
        Gradients corresponding to `trajectory`.
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
        Gradients over 2 or 3 directions.
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
        initial_positions = np.zeros((gradients.shape[0], 1, gradients.shape[-1]))

    # Prepare and integrate gradients
    trajectory = gradients * acq.gamma * acq.raster_time
    trajectory = np.concatenate([initial_positions[:, None, :], trajectory], axis=1)
    trajectory = np.cumsum(trajectory, axis=1)

    # Normalize the trajectory for NUFFT usage
    trajectory = normalize_trajectory(trajectory, acq)
    return trajectory


def convert_gradients_to_slew_rates(
    gradients: NDArray,
    acq: Acquisition | None = None,
) -> tuple[NDArray, NDArray]:
    """Derive the gradients over time to provide slew rates.

    Parameters
    ----------
    gradients : NDArray
        Gradients over 2 or 3 directions.
    acq : Acquisition, optional
        Acquisition configuration to use.
        If `None`, the default acquisition is used.
    Returns
    -------
    slewrates : NDArray
        Slew rates corresponding to `gradients`.
    initial_gradients : NDArray
        Gradients at the beginning of the readout window.
    """
    # Compute slew rates and starting gradients
    acq = acq or Acquisition.default
    slewrates = np.diff(gradients, axis=1) / acq.raster_time
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
