"""Utility functions for the trajectory design."""

import numpy as np

from enum import Enum, EnumMeta


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

    def __getitem__(self, name):
        """Allow ``MyEnum['Member'] == MyEnum['MEMBER']`` ."""
        return super().__getitem__(name.upper())

    def __getattr__(self, name):
        """Allow ``MyEnum.Member == MyEnum.MEMBER`` ."""
        return super().__getattr__(name.upper())


class FloatEnum(float, Enum, metaclass=CaseInsensitiveEnumMeta):
    """An Enum for float that is case insensitive for ist attributes."""

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
    """Enumerate spiral types."""

    ARCHIMEDES = 1
    ARITHMETIC = ARCHIMEDES
    FERMAT = 2
    PARABOLIC = FERMAT
    HYPERBOLIC = -1
    LITHUUS = -2


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


class Tilts(str, Enum):
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


class Packings(str, Enum, metaclass=CaseInsensitiveEnumMeta):
    """Enumerate available packing method for shots.

    It is mostly use for wave-CAIPI trajectory

    See Also
    --------
    mrinufft.trajectories.trajectories3D.initialize_3D_wave_caipi

    """

    RANDOM = "random"
    CIRCLE = "circle"
    TRIANGLE = "triangle"
    HEXAGON = "hexagon"
    SQUARE = "square"

    # Aliases
    CIRCULAR = CIRCLE
    TRIANGULAR = TRIANGLE
    HEXAGONAL = HEXAGON
    UNIFORM = RANDOM


###############
# CONSTRAINTS #
###############


def normalize_trajectory(
    trajectory,
    norm_factor=KMAX,
    resolution=DEFAULT_RESOLUTION,
):
    """Normalize an un-normalized/natural trajectory for NUFFT use.

    Parameters
    ----------
    trajectory : np.ndarray
        Un-normalized trajectory consisting of k-space coordinates in 2D or 3D.
    norm_factor : float, optional
        Trajectory normalization factor, by default KMAX.
    resolution : float, np.array, optional
        Resolution of MR image in meters, isotropic as `int`
        or anisotropic as `np.array`.
        The default is DEFAULT_RESOLUTION.

    Returns
    -------
    trajectory : np.ndarray
        Normalized trajectory corresponding to `trajectory` input.
    """
    return trajectory * norm_factor * (2 * resolution)


def unnormalize_trajectory(
    trajectory,
    norm_factor=KMAX,
    resolution=DEFAULT_RESOLUTION,
):
    """Un-normalize a NUFFT-normalized trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        Normalized trajectory consisting of k-space coordinates in 2D or 3D.
    norm_factor : float, optional
        Trajectory normalization factor, by default KMAX.
    resolution : float, np.array, optional
        Resolution of MR image in meters, isotropic as `int`
        or anisotropic as `np.array`.
        The default is DEFAULT_RESOLUTION.

    Returns
    -------
    trajectory : np.ndarray
        Un-normalized trajectory corresponding to `trajectory` input.
    """
    return trajectory / norm_factor / (2 * resolution)


def convert_trajectory_to_gradients(
    trajectory,
    norm_factor=KMAX,
    resolution=DEFAULT_RESOLUTION,
    raster_time=DEFAULT_RASTER_TIME,
    gamma=Gammas.HYDROGEN,
):
    """Derive a normalized trajectory over time to provide gradients.

    Parameters
    ----------
    trajectory : np.ndarray
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
    gradients : np.ndarray
        Gradients corresponding to `trajectory`.
    """
    # Un-normalize the trajectory from NUFFT usage
    trajectory = unnormalize_trajectory(trajectory, norm_factor, resolution)

    # Compute gradients and starting positions
    gradients = np.diff(trajectory, axis=1) / gamma / raster_time
    initial_positions = trajectory[:, 0, :]
    return gradients, initial_positions


def convert_gradients_to_trajectory(
    gradients,
    initial_positions=None,
    norm_factor=KMAX,
    resolution=DEFAULT_RESOLUTION,
    raster_time=DEFAULT_RASTER_TIME,
    gamma=Gammas.HYDROGEN,
):
    """Integrate gradients over time to provide a normalized trajectory.

    Parameters
    ----------
    gradients : np.ndarray
        Gradients over 2 or 3 directions.
    initial_positions: np.ndarray, optional
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
    trajectory : np.ndarray
        Normalized trajectory corresponding to `gradients`.
    """
    # Handle no initial positions
    if initial_positions is None:
        initial_positions = np.zeros((gradients.shape[0], 1, gradients.shape[-1]))

    # Prepare and integrate gradients
    trajectory = gradients * gamma * raster_time
    trajectory = np.concatenate([initial_positions, trajectory])
    trajectory = np.cumsum(trajectory, axis=1)

    # Normalize the trajectory for NUFFT usage
    trajectory = normalize_trajectory(trajectory, norm_factor, resolution)
    return trajectory


def convert_gradients_to_slew_rates(
    gradients,
    raster_time=DEFAULT_RASTER_TIME,
):
    """Derive the gradients over time to provide slew rates.

    Parameters
    ----------
    gradients : np.ndarray
        Gradients over 2 or 3 directions.
    raster_time : float, optional
        Amount of time between the acquisition of two
        consecutive samples in ms.
        The default is `DEFAULT_RASTER_TIME`.

    Returns
    -------
    slewrates : np.ndarray
        Slew rates corresponding to `gradients`.
    initial_gradients : np.ndarray
        Gradients at the beginning of the readout window.
    """
    # Compute slew rates and starting gradients
    slewrates = np.diff(gradients, axis=1) / raster_time
    initial_gradients = gradients[:, 0, :]
    return slewrates, initial_gradients


def convert_slew_rates_to_gradients(
    slewrates,
    initial_gradients=None,
    raster_time=DEFAULT_RASTER_TIME,
):
    """Integrate slew rates over time to provide gradients.

    Parameters
    ----------
    slewrates : np.ndarray
        Slew rates over 2 or 3 directions.
    initial_gradients: np.ndarray, optional
        Gradients at the beginning of the readout window.
        The default is `None`.
    raster_time : float, optional
        Amount of time between the acquisition of two
        consecutive samples in ms.
        The default is `DEFAULT_RASTER_TIME`.

    Returns
    -------
    gradients : np.ndarray
        Gradients corresponding to `slewrates`.
    """
    # Handle no initial gradients
    if initial_gradients is None:
        initial_gradients = np.zeros((slewrates.shape[0], 1, slewrates.shape[-1]))

    # Prepare and integrate slew rates
    gradients = slewrates * raster_time
    gradients = np.concatenate([initial_gradients, gradients])
    gradients = np.cumsum(gradients, axis=1)
    return gradients


def compute_gradients_and_slew_rates(
    trajectory,
    norm_factor=KMAX,
    resolution=DEFAULT_RESOLUTION,
    raster_time=DEFAULT_RASTER_TIME,
    gamma=Gammas.HYDROGEN,
):
    """Compute the gradients and slew rates from a normalized trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
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
    gradients : np.ndarray
        Gradients corresponding to `trajectory`.
    slewrates : np.ndarray
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
    gradients, slewrates, gmax=DEFAULT_GMAX, smax=DEFAULT_SMAX, order=None
):
    """Check if a trajectory satisfies the gradient hardware constraints.

    Parameters
    ----------
    gradients : np.ndarray
        Gradients to check
    slewrates: np.ndarray
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


###############
# MATHEMATICS #
###############


def compute_greatest_common_divider(p, q):
    """Compute the greatest common divider of two integers p and q.

    Parameters
    ----------
    p : int
        First integer.
    q : int
        Second integer.

    Returns
    -------
    int
        The greatest common divider of p and q.
    """
    while q != 0:
        p, q = q, p % q
    return p


def compute_coprime_factors(Nc, length, start=1, update=1):
    """Compute a list of coprime factors of Nc.

    Parameters
    ----------
    Nc : int
        Number to factorize.
    length : int
        Number of coprime factors to compute.
    start : int, optional
        First number to check. The default is 1.
    update : int, optional
        Increment between two numbers to check. The default is 1.

    Returns
    -------
    list
        List of coprime factors of Nc.
    """
    count = start
    coprimes = []
    while len(coprimes) < length:
        if compute_greatest_common_divider(Nc, count) == 1:
            coprimes.append(count)
        count += update
    return coprimes


#############
# ROTATIONS #
#############


def R2D(theta):
    """Initialize 2D rotation matrix.

    Parameters
    ----------
    theta : float
        Rotation angle in rad.

    Returns
    -------
    np.ndarray
        2D rotation matrix.
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def Rx(theta):
    """Initialize 3D rotation matrix around x axis.

    Parameters
    ----------
    theta : float
        Rotation angle in rad.

    Returns
    -------
    np.ndarray
        2D rotation matrix.
    """
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def Ry(theta):
    """Initialize 3D rotation matrix around y axis.

    Parameters
    ----------
    theta : float
        Rotation angle in rad.

    Returns
    -------
    np.ndarray
        2D rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def Rz(theta):
    """Initialize 3D rotation matrix around z axis.

    Parameters
    ----------
    theta : float
        Rotation angle in rad.

    Returns
    -------
    np.ndarray
        2D rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def Rv(v1, v2, normalize=True):
    """Initialize 3D rotation matrix from two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.
    normalize : bool, optional
        Normalize the vectors. The default is True.

    Returns
    -------
    np.ndarray
        3D rotation matrix.
    """
    if normalize:
        v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    cos_theta = np.dot(v1, v2)
    v3 = np.cross(v1, v2)
    cross_matrix = np.cross(v3, np.identity(v3.shape[0]) * -1)
    return np.identity(3) + cross_matrix + cross_matrix @ cross_matrix / (1 + cos_theta)


###########
# OPTIONS #
###########


def initialize_tilt(tilt, nb_partitions=1):
    r"""Initialize the tilt angle.

    Parameters
    ----------
    tilt : str or float
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
    if not (isinstance(tilt, str) or tilt is None):
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


def initialize_spiral(spiral):
    """Initialize the spiral type.

    Parameters
    ----------
    spiral : str or float
        Spiral type or spiral power value.

    Returns
    -------
    float
        Spiral power value.
    """
    if isinstance(spiral, float):
        return spiral
    return Spirals[spiral]


def initialize_shape_norm(shape):
    """Initialize the norm for a given shape.

    Parameters
    ----------
    shape : str or float
        Shape name or p-norm value.

    Returns
    -------
    float
        Shape p-norm value.
    """
    if isinstance(shape, float):
        return shape
    return NormShapes[shape]
