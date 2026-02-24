"""fMRI-oriented 3D trajectory initializations (TURBINE, REPI)."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.maths import R2D
from mrinufft.trajectories.utils import initialize_tilt
from mrinufft.trajectories.tools import stack, epify
from mrinufft.trajectories.inits.radial import initialize_2D_radial
from mrinufft.trajectories.inits.spiral import initialize_2D_spiral


def initialize_3D_turbine(
    Nc: int,
    Ns_readouts: int,
    Ns_transitions: int,
    nb_blades: int,
    blade_tilt: str = "uniform",
    nb_trains: int | Literal["auto"] = "auto",
    skip_factor: int = 1,
    in_out: bool = True,
) -> NDArray:
    """Initialize 3D TURBINE trajectory.

    This is an implementation of the TURBINE (Trajectory Using Radially
    Batched Internal Navigator Echoes) trajectory proposed in [MGM10]_.
    It consists of EPI-like multi-echo planes rotated around any axis
    (here :math:`k_z`-axis) in a radial fashion.

    Note that our implementation also proposes to segment the planes
    into several shots instead of just one, and includes the proposition
    from [GMC22]_ to also accelerate within the blades by skipping lines
    but while alternating them between blades.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns_readouts : int
        Number of samples per readout
    Ns_transitions : int
        Number of samples per transition between two readouts
    nb_blades : int
        Number of line stacks over the :math:`k_z`-axis axis
    blade_tilt : str, float, optional
        Tilt between individual blades, by default "uniform"
    nb_trains : int, Literal["auto"], optional
        Number of resulting shots, or readout trains, such that each of
        them will be composed of :math:`n` readouts with
        ``Nc = n * nb_trains``. If ``"auto"`` then ``nb_trains`` is set
        to ``nb_blades``.
    skip_factor : int, optional
        Factor defining the way different blades alternate to skip lines,
        forming groups of `skip_factor` non-redundant blades.
    in_out : bool, optional
        Whether the curves are going in-and-out or start from the center

    Returns
    -------
    NDArray
        3D TURBINE trajectory

    References
    ----------
    .. [MGM10] McNab, Jennifer A., Daniel Gallichan, and Karla L. Miller.
       "3D steady-state diffusion-weighted imaging with trajectory using
       radially batched internal navigator echoes (TURBINE)."
       Magnetic Resonance in Medicine 63, no. 1 (2010): 235-242.
    .. [GMC22] Graedel, Nadine N., Karla L. Miller, and Mark Chiew.
       "Ultrahigh resolution fMRI at 7T using radial-cartesian TURBINE sampling."
       Magnetic Resonance in Medicine 88, no. 5 (2022): 2058-2073.
    """
    # Assess arguments validity
    if nb_trains == "auto":
        nb_trains = nb_blades
    if not isinstance(skip_factor, int):
        raise ValueError("`skip_factor` must be an integer.")
    if Nc % nb_blades != 0:
        raise ValueError("`nb_blades` should divide `Nc`.")
    if Nc % nb_trains != 0:
        raise ValueError("`nb_trains` should divide `Nc`.")
    if nb_trains and (Nc % nb_trains != 0):
        raise ValueError("`nb_trains` should divide `Nc`.")
    nb_shot_per_blade = Nc // nb_blades

    # Initialize the first shot of each blade on a plane
    single_plane = initialize_2D_radial(
        nb_blades, Ns_readouts, in_out=in_out, tilt=blade_tilt
    )

    # Stack the blades first shots with tilt to create each full blade
    trajectory = stack(single_plane, nb_shot_per_blade * skip_factor, z_tilt=0)

    # Re-order the shot to be EPI-compatible
    trajectory = trajectory.reshape(
        (nb_shot_per_blade * skip_factor, nb_blades, Ns_readouts, 3)
    )
    trajectory = np.swapaxes(trajectory, 0, 1)
    trajectory = trajectory.reshape((Nc * skip_factor, Ns_readouts, 3))

    # Skip lines but alternate which ones between blades
    skip_mask = (
        np.arange(nb_blades)[:, None]
        + np.arange(nb_shot_per_blade * skip_factor)[None, :]
    )
    skip_mask = ((skip_mask % skip_factor) == 0).astype(bool).flatten()
    trajectory = trajectory[skip_mask]

    # Merge shots into EPI-like multi-readout trains
    if nb_trains and nb_trains != Nc:
        trajectory = epify(
            trajectory,
            Ns_transitions=Ns_transitions,
            nb_trains=nb_trains,
            reverse_odd_shots=True,
        )

    return trajectory


def initialize_3D_repi(
    Nc: int,
    Ns_readouts: int,
    Ns_transitions: int,
    nb_blades: int,
    nb_blade_revolutions: float = 0,
    blade_tilt: str = "uniform",
    nb_spiral_revolutions: float = 0,
    spiral: str = "archimedes",
    nb_trains: int | Literal["auto"] = "auto",
    in_out: bool = True,
) -> NDArray:
    """Initialize 3D REPI trajectory.

    This is an implementation of the REPI (Radial Echo Planar Imaging)
    trajectory proposed in [RMS22]_ and officially inspired
    from TURBINE proposed in [MGM10]_.
    It consists of multi-echo stacks of lines or spirals rotated around any axis
    (here :math:`k_z`-axis) in a radial fashion, but each stack is also slightly
    shifted along the rotation axis in order to be entangled with the others
    without redundancy. This feature is similar to choosing ``skip_factor``
    equal to ``nb_blades`` in TURBINE.

    Note that our implementation also proposes to segment the planes/stacks
    into several shots, instead of just one. Spirals can also be customized
    beyond the classic Archimedean spiral.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns_readouts : int
        Number of samples per readout
    Ns_transitions : int
        Number of samples per transition between two readouts
    nb_blades : int
        Number of line stacks over the kz axis
    nb_blade_revolutions : float
        Number of revolutions over lines/spirals within a blade
        over the kz axis.
    blade_tilt : str, float, optional
        Tilt between individual blades, by default "uniform"
    nb_spiral_revolutions : float, optional
        Number of revolutions of the spirals over the readouts, by default 0
    spiral : str, float, optional
        Spiral type, by default "archimedes"
    nb_trains : int, Literal["auto"], optional
        Number of trains dividing the readouts, such that each
        shot will be composed of `n` readouts with `Nc = n * nb_trains`.
        If "auto" then `nb_trains` is set to `nb_blades`.
    in_out : bool, optional
        Whether the curves are going in-and-out or start from the center

    Returns
    -------
    NDArray
        3D REPI trajectory

    References
    ----------
    .. [RMS22] Rettenmeier, Christoph A., Danilo Maziero, and V. Andrew Stenger.
       "Three dimensional radial echo planar imaging for functional MRI."
       Magnetic Resonance in Medicine 87, no. 1 (2022): 193-206.
    """
    # Assess arguments validity
    if nb_trains == "auto":
        nb_trains = nb_blades
    if Nc % nb_blades != 0:
        raise ValueError("`nb_blades` should divide `Nc`.")
    if Nc % nb_trains != 0:
        raise ValueError("`nb_trains` should divide `Nc`.")
    if nb_trains % nb_blades != 0:
        raise ValueError("`nb_trains` should divide `nb_blades`.")
    nb_shot_per_blade = Nc // nb_blades
    nb_spiral_revolutions = max(nb_spiral_revolutions, 1e-5)

    # Initialize trajectory as a stack of single shots
    single_shot = initialize_2D_spiral(
        1, Ns_readouts, in_out=in_out, tilt=0, nb_revolutions=nb_spiral_revolutions
    )
    shot_tilt = 2 * np.pi * nb_blade_revolutions / Nc
    trajectory = stack(single_shot, Nc, z_tilt=shot_tilt)

    # Rotate some shots to separate the blades
    for i_b in range(nb_blades):
        rotation = R2D(
            i_b * initialize_tilt(blade_tilt, nb_partitions=nb_blades) / (1 + in_out)
        ).T
        trajectory[i_b::nb_blades, ..., :2] = (
            trajectory[i_b::nb_blades, ..., :2] @ rotation
        )

    # Re-order the shot to be EPI-compatible
    trajectory = trajectory.reshape((nb_shot_per_blade, nb_blades, Ns_readouts, 3))
    trajectory = np.swapaxes(trajectory, 0, 1)
    trajectory = trajectory.reshape((Nc, Ns_readouts, 3))

    # Merge shots into EPI-like multi-readout trains
    if nb_trains and nb_trains != Nc:
        trajectory = epify(
            trajectory,
            Ns_transitions=Ns_transitions,
            nb_trains=nb_trains,
            reverse_odd_shots=True,
        )

    return trajectory
