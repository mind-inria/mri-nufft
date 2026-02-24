"""Wave-CAIPI 3D trajectory initialization."""

import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.utils import KMAX, Acquisition, convert_gradients_to_trajectory
from mrinufft.trajectories.tools import get_grappa_caipi_positions, get_packing_spacing_positions


def initialize_3D_wave_caipi(
    Nc_or_R: int | tuple,
    Ns: int,
    nb_revolutions: float = 5,
    width: float = 1,
    packing: str = "triangular",
    shape: str | float = "square",
    spacing: tuple = (1, 1),
    readout_axis: int = 0,
    wavegrad: float | tuple | None = None,
    caipi_delta: int = 0,
    acq: Acquisition | None = None,
) -> NDArray:
    """Initialize 3D trajectories with Wave-CAIPI.

    This implementation is based on the work from [Bil+15]_.

    Parameters
    ----------
    Nc_or_R : int or tuple[int, int]
        Number of shots `Nc` or GRAPPA `R` factors along the two
        phase-encoding directions.
        - If an **int** is provided, it is interpreted as `Nc` (number of shots).
        - If a **tuple[int, int]** is provided, it is interpreted as
          `R` (GRAPPA factors).
    Ns : int
        Number of samples per shot
    nb_revolutions : float, optional
        Number of revolutions, by default 5
    width : float, optional
        Diameter of the helices normalized such that `width=1`
        densely covers the k-space without overlap for square packing,
        by default 1.
    packing : str, optional
        Packing method used to position the helices:
        "triangular"/"hexagonal", "square", "circular"
        or "random"/"uniform", by default "triangular".
    shape : str | float, optional
        Shape over the 2D :math:`k_x-k_y` plane to pack with shots,
        either defined as `str` ("circle", "square", "diamond")
        or as `float` through p-norms following the conventions
        of the `ord` parameter from `numpy.linalg.norm`,
        by default "circle".
    spacing : tuple[int, int]
        Spacing between helices over the 2D :math:`k_x-k_y` plane
        normalized similarly to `width` to correspond to
        helix diameters, by default (1, 1).
    readout_axis : int, optional
        Axis along which the readout is performed,
        either 0, 1 or 2.
    wavegrad : float, optional
        Wave gradient amplitude in T/m, by default None
        If None, the value of `width` is used to estimate
        the wave gradient amplitude. `acq` must be provided
        if used.
    acq : Acquisition, optional
        Acquisition parameters, by default None.

    Returns
    -------
    NDArray
        3D wave-CAIPI trajectory

    References
    ----------
    .. [Bil+15] Bilgic, Berkin, Borjan A. Gagoski, Stephen F. Cauley, Audrey P. Fan,
       Jonathan R. Polimeni, P. Ellen Grant, Lawrence L. Wald, and Kawin Setsompop.
       "Wave-CAIPI for highly accelerated 3D imaging."
       Magnetic resonance in medicine 73, no. 6 (2015): 2152-2162.
    """
    acq = acq if acq is not None else Acquisition.default
    if not np.isscalar(Nc_or_R):
        R = Nc_or_R
        sample_axis = tuple(
            im for i, im in enumerate(acq.img_size) if i != readout_axis
        )
        positions = (
            get_grappa_caipi_positions(sample_axis, R, caipi_delta) / acq.norm_factor
        )
        wavegrad = np.array(
            [[[wavegrad]]] if np.isscalar(wavegrad) else [[list(wavegrad)]], np.float32
        )
        # Get the trajectory for the gradient wave.
        # Normalize back to -1, 1 as thats how we start defining trajectory
        width = (
            (
                np.squeeze(convert_gradients_to_trajectory(wavegrad, acq=acq))[-1]
                / acq.norm_factor
            )
            * Ns
            / 2
            / np.pi
            / nb_revolutions
        )  # Extra factor from angles
        if np.isscalar(width):
            width = (width, width, width)
        width = tuple(w for i, w in enumerate(width) if i != readout_axis)
    else:
        width = (width, width) if np.isscalar(width) else width
        positions = get_packing_spacing_positions(Nc_or_R, packing, shape, spacing)
    # Initialize first shot
    angles = nb_revolutions * 2 * np.pi * np.arange(0, Ns) / Ns
    initial_shot = np.stack(
        [width[0] * np.cos(angles), width[1] * np.sin(angles), np.linspace(-1, 1, Ns)],
        axis=-1,
    )
    # reorder based on readout axis
    perm = [[2, 0, 1], [1, 2, 0], [0, 1, 2]][readout_axis]
    initial_shot = initial_shot[..., perm]

    # Shifting copies of the initial shot to all positions
    positions = np.insert(positions, readout_axis, 0, axis=-1)
    trajectory = initial_shot[None] + positions[:, None]

    axes = [[1, 2], [0, 2], [0, 1]][readout_axis]
    if np.isscalar(Nc_or_R):
        trajectory[..., axes] /= np.max(np.abs(trajectory))
    return KMAX * trajectory
