"""Trajectories based on the ECCENTRIC formulation."""

import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.maths import R2D
from mrinufft.trajectories.utils import KMAX


def _initialize_2D_eccentric(
    Nc: int,
    Ns: int,
    radius_ratio: float,
    center_ratio: float = 0.0,
    nb_revolutions: float = 1.0,
    min_distance: float = 0.0,
    max_radius: float = 1.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Initialize a 2D ECCENTRIC trajectory.

    This function offers an additional degree of customization
    with `max_radius` used to generate 3D stacks following the authors
    specifications.
    It is made private to avoid giving users access to
    the `max_radius` argument as it is irrelevant for them.

    Parameters
    ----------
    Nc : int
        Number of shots/circles.
    Ns : int
        Number of samples per shot/circle.
    radius_ratio : float
        Radius of each circle relatively to the k-space radius,
        between 0 and 0.5.
    center_ratio : float, optional
        Proportion between 0 and 1 of shots positioned around
        the center into a pseudo-rosette pattern. Default to 0.
    nb_revolutions : float, optional
        Number of revolutions per circle. Defaults to 1.
    min_distance : float, optional
        Minimum allowed distance between consecutive circles relatively
        to the k-space radius between 0 and 0.5. Defaults to 0.
    max_radius : float, optional
        Maximum radius for circle placement relative to the k-space radius,
        between 0 and 1. Defaults to 1.
    seed : int | None, optional
        Random seed for reproducibility, used only to draw the circle centers.
        Defaults to None.

    Returns
    -------
    NDArray[np.float64]
        The generated 2D trajectory with shape (Nc, Ns, 2).
    """
    # Check arguments validity
    if not (0 < radius_ratio <= 0.5):
        raise ValueError("The `radius_ratio` should be strictly between 0 and 0.5.")
    if not (0 <= center_ratio <= 1):
        raise ValueError("The `center_ratio` should be between 0 and 1.")
    if not (0 <= min_distance <= 0.5):
        raise ValueError("The `min_distance` should be between 0 and 0.5.")
    if not (0 <= max_radius <= 1):
        raise ValueError("The `max_radius` should be between 0 and 1.")

    # Define a single circle
    circle_angles = np.linspace(0, 2 * np.pi * nb_revolutions, Ns, endpoint=False)
    circle = np.zeros((Ns, 2))
    circle[:, 0] = radius_ratio * np.cos(circle_angles)
    circle[:, 1] = radius_ratio * np.sin(circle_angles)

    # Draw random positions for each circle until consecutive
    # circles are not too close
    rng = np.random.default_rng(seed=seed)
    distances, angles = np.zeros(Nc), np.zeros(Nc)
    close_mask = np.ones(Nc).astype(bool)
    while close_mask.any():
        # Draw new points only where too close
        nb_close = np.sum(close_mask)
        distances[close_mask] = (
            rng.random(size=nb_close) * max_radius * (1 - radius_ratio)
        )
        angles[close_mask] = rng.random(size=nb_close) * 2 * np.pi

        # Update positions
        positions = np.zeros((Nc, 2))
        positions[:, 0] = distances * np.cos(angles)
        positions[:, 1] = distances * np.sin(angles)

        # Check again for closeness
        close_mask = (
            np.linalg.norm(positions - np.roll(positions, shift=1, axis=0), axis=-1)
            < min_distance
        )
        close_mask[0] = False
        # Break the rolling closeness and guarantee to find a solution

    # Enforce some central positions to be in pseudo-rosette style
    Nc_center = round(Nc * center_ratio)
    angles[:Nc_center] = np.linspace(0, 2 * np.pi, Nc_center, endpoint=False)
    positions[:Nc_center, 0] = radius_ratio * np.cos(angles[:Nc_center])
    positions[:Nc_center, 1] = radius_ratio * np.sin(angles[:Nc_center])

    # Assemble trajectory
    trajectory = np.zeros((Nc, Ns, 2))
    for i in range(Nc):
        # Apply rotation so the circle's first point is the
        # closest to the center
        rotation = R2D(np.pi + angles[i]).T
        trajectory[i] = positions[i, None, :] + circle[None, :, :] @ rotation

    return KMAX * trajectory


def initialize_2D_eccentric(
    Nc: int,
    Ns: int,
    radius_ratio: float,
    center_ratio: float = 0.0,
    nb_revolutions: float = 1.0,
    min_distance: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Initialize a 2D ECCENTRIC trajectory.

    This is a reproduction of the proposition from [Kla+24]_.
    It creates trajectories as uniformly distributed circles,
    with a pseudo rosette-like structure at the center to ensure
    its coverage. ECCENTRIC stands for ECcentric Circle ENcoding
    TRajectorIes for Compressed sensing.

    Notes
    -----
    This implementation follows the original propositions but
    decisions were made about missing details and additional
    features are proposed:
    - circles are oriented such that their starting points are the closest
    to the center. It is chosen to avoid sampling the center at
    radically different times, which would cause contrast discrepancies
    and signal loss due to dephasing.
    - the number of circle revolutions is an input instead of sticking to 1,
    to handle multi-echo sequences or simply benefit from a higher duty cycle.

    Parameters
    ----------
    Nc : int
        Number of shots/circles.
    Ns : int
        Number of samples per shot/circle.
    radius_ratio : float
        Radius of each circle relatively to the k-space radius,
        between 0 and 0.5.
    center_ratio : float, optional
        Proportion between 0 and 1 of shots positioned around
        the center into a pseudo-rosette pattern. Default to 0.
    nb_revolutions : float, optional
        Number of revolutions per circle. Defaults to 1.
    min_distance : float, optional
        Minimum allowed distance between consecutive circles relatively
        to the k-space radius between 0 and 0.5. Defaults to 0.
    seed : int | None, optional
        Random seed for reproducibility, used only to draw the circle centers.
        Defaults to None.

    Returns
    -------
    NDArray[np.float64]
        The generated 2D trajectory with shape (Nc, Ns, 2).

    References
    ----------
    .. [Kla+24] Klauser, Antoine, Bernhard Strasser, Wolfgang Bogner,
       Lukas Hingerl, Sebastien Courvoisier, Claudiu Schirda,
       Bruce R. Rosen, Francois Lazeyras, and Ovidiu C. Andronesi.
       "ECCENTRIC: a fast and unrestrained approach for high-resolution
       in vivo metabolic imaging at ultra-high field MR".
       Imaging Neuroscience 2 (2024): 1-20.
    """
    return _initialize_2D_eccentric(
        Nc=Nc,
        Ns=Ns,
        radius_ratio=radius_ratio,
        center_ratio=center_ratio,
        nb_revolutions=nb_revolutions,
        min_distance=min_distance,
        max_radius=1,
        seed=seed,
    )


def initialize_3D_eccentric(
    Nc: int,
    Ns: int,
    nb_stacks: int,
    radius_ratio: float,
    center_ratio: float = 0.0,
    nb_revolutions: float = 1.0,
    min_distance: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Initialize a 3D ECCENTRIC trajectory.

    This is a reproduction of the proposition from [Kla+24]_.
    It creates trajectories as uniformly distributed circles
    stacked spherically over the :math:`k_z`-axis, with a pseudo
    rosette-like structure at the center to ensure its coverage.
    ECCENTRIC stands for ECcentric Circle ENcoding TRajectorIes
    for Compressed sensing.

    Notes
    -----
    This implementation follows the original propositions but
    decisions were made about missing details and additional
    features are proposed:
    - circles are oriented such that their starting points are the closest
    to the center. It is chosen to avoid sampling the center at
    radically different times, which would cause contrast discrepancies
    and signal loss due to dephasing.
    - the number of circle revolutions is an input instead of sticking to 1,
    to handle multi-echo sequences or simply benefit from a higher duty cycle.

    Parameters
    ----------
    Nc : int
        Number of shots/circles.
    Ns : int
        Number of samples per shot/circle.
    nb_stacks : int
        Number of stack layers along the :math:`k_z`-axis.
    radius_ratio : float
        Radius of each circle relatively to the k-space radius,
        between 0 and 0.5.
    center_ratio : float, optional
        Proportion between 0 and 1 of shots positioned around
        the center into a pseudo-rosette pattern. Default to 0.
    nb_revolutions : float, optional
        Number of revolutions per circle. Defaults to 1.
    min_distance : float, optional
        Minimum allowed distance between consecutive circles relatively
        to the k-space radius between 0 and 0.5. Defaults to 0.
    max_radius : float, optional
        Maximum radius for circle placement relative to the k-space radius,
        between 0 and 1. Defaults to 1.
    seed : int | None, optional
        Random seed for reproducibility, used only to draw the circle centers.
        Defaults to None.

    Returns
    -------
    NDArray[np.float64]
        The generated 3D trajectory with shape (Nc, Ns, 3).

    References
    ----------
    .. [Kla+24] Klauser, Antoine, Bernhard Strasser, Wolfgang Bogner,
       Lukas Hingerl, Sebastien Courvoisier, Claudiu Schirda,
       Bruce R. Rosen, Francois Lazeyras, and Ovidiu C. Andronesi.
       "ECCENTRIC: a fast and unrestrained approach for high-resolution
       in vivo metabolic imaging at ultra-high field MR".
       Imaging Neuroscience 2 (2024): 1-20.
    """
    trajectory = np.zeros((Nc, Ns, 3))

    # Attribute shots to stacks following a prescribed density
    Nc_per_stack = np.ones(nb_stacks).astype(int)
    stack_positions = np.linspace(-1, 1, nb_stacks)
    density = np.sqrt(
        1 - stack_positions**2
    )  # same as the paper but simpler formulation
    for _ in range(Nc - nb_stacks):
        idx = np.argmax(density / Nc_per_stack)
        Nc_per_stack[idx] += 1

    # Generate each stack
    counter = 0
    for i in range(nb_stacks):
        # Set indices and update counter for next round
        id_start = counter
        id_end = counter + Nc_per_stack[i]
        counter += Nc_per_stack[i]

        # Generate the stack but change the maximum radius
        stack = _initialize_2D_eccentric(
            Nc=Nc_per_stack[i],
            Ns=Ns,
            radius_ratio=radius_ratio,
            center_ratio=center_ratio,
            nb_revolutions=nb_revolutions,
            min_distance=min_distance,
            max_radius=density[i],
            seed=seed,
        )
        trajectory[id_start:id_end, :, :2] = stack
        trajectory[id_start:id_end, :, 2] = KMAX * stack_positions[i]

    return trajectory
