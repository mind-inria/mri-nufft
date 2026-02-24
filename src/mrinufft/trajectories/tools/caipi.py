"""CAIPI trajectory tools: get_grappa_caipi_positions, get_packing_spacing_positions."""

from functools import partial

import numpy as np
from numpy.typing import NDArray
import numpy.linalg as nl

from mrinufft.trajectories.utils import KMAX, Packings, initialize_shape_norm
from mrinufft.trajectories.maths import CIRCLE_PACKING_DENSITY, generate_fibonacci_circle


def get_grappa_caipi_positions(
    img_size: tuple,
    grappa_factors: tuple,
    caipi_delta: int = 0,
    acs_region: tuple | None = None,
) -> NDArray | tuple:
    """
    Generate a Cartesian k-space sampling mask for GRAPPA with optional CAIPI shifts.

    This function computes the k-space sampling positions for a GRAPPA (GeneRalized
    Autocalibrating Partial Parallel Acquisition) pattern, optionally incorporating
    CAIPI (Controlled Aliasing in Parallel Imaging) shifts. The sampling points are
    distributed over a Cartesian grid based on the specified GRAPPA acceleration
    factors and image size.

    Parameters
    ----------
    img_size : array_like of int, shape (2,)
        The size of the k-space grid (number of samples) along each dimension,
        typically corresponding to the phase-encoding and frequency-encoding
        directions.

    grappa_factors : array_like of int, shape (2,)
        The GRAPPA acceleration factors along each axis. For example, a factor
        of ``[2, 1]`` means every second line is sampled along the first axis,
        while every line is sampled along the second axis.

    caipi_delta : float, optional
        The CAIPI phase shift (in units of k-space fraction) to apply along the
        second dimension. A nonzero value introduces controlled aliasing between
        slices or coil elements. Default is ``0`` (no shift).

    acs_region : array_like of int, shape (2,)
        The size of the Auto-Calibration Signal (ACS) region along each dimension.
        This region is fully sampled to allow for accurate GRAPPA kernel estimation.

    Returns
    -------
    positions : ndarray of shape (N, 2)
        The Cartesian coordinates of the sampled k-space points in 2D plane.
    acs_positions : ndarray of shape (M, 2)
        The Cartesian coordinates of the ACS region k-space points in 2D plane.

    Notes
    -----
    - This function merely gives the mask positions, not the entire trajectory.
    """
    Nc_per_axis = np.array(img_size) // np.array(grappa_factors)
    positions = (
        np.asarray(
            np.meshgrid(
                *[
                    np.linspace(-KMAX, KMAX, num, endpoint=num % 2)
                    for num in Nc_per_axis
                ],
                indexing="ij",
            )
        )
        .T.astype(np.float32)
        .reshape(-1, 2)
    )
    if caipi_delta > 0:
        positions[::2, 1] += caipi_delta / img_size[1]
    if acs_region is not None:
        acs_max_loc = np.array(acs_region) / np.array(img_size) * KMAX
        acs_positions = (
            np.asarray(
                np.meshgrid(
                    *[
                        np.linspace(-max_loc, max_loc, num, endpoint=num % 2)  # type: ignore
                        for num, max_loc in zip(acs_region, acs_max_loc)
                    ],
                    indexing="ij",
                )
            )
            .T.astype(np.float32)
            .reshape(-1, 2)
        )
        return positions, acs_positions
    return positions


def get_packing_spacing_positions(
    Nc: int,
    packing: str = "triangular",
    shape: str | float = "square",
    spacing: tuple = (1, 1),
):
    """
    Generate a k-space positions for a fixed spacing and packing.

    Parameters
    ----------
    Nc : int
        Number of positions
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

    Returns
    -------
    positions : ndarray of shape (N, 2)
        The Non-Cartesian coordinates of the sampled k-space points in 2D plane.
    """
    # Choose the helix positions according to packing
    packing_enum = Packings[packing]
    side = 2 * int(np.ceil(np.sqrt(Nc))) * np.max(spacing)
    if packing_enum == Packings.RANDOM:
        positions = 2 * side * (np.random.random((side * side, 2)) - 0.5)
    elif packing_enum == Packings.CIRCLE:
        positions = np.zeros((1, 2))
        counter = 0
        while len(positions) < side**2:
            counter += 1
            perimeter = 2 * np.pi * counter
            nb_shots = int(np.trunc(perimeter))
            # Add the full circle
            radius = 2 * counter
            angles = 2 * np.pi * np.arange(nb_shots) / nb_shots
            circle = radius * np.exp(1j * angles)
            positions = np.concatenate(
                [positions, np.array([circle.real, circle.imag]).T], axis=0
            )
    elif packing_enum in [Packings.SQUARE, Packings.TRIANGLE, Packings.HEXAGONAL]:
        # Square packing or initialize hexagonal/triangular packing
        px, py = np.meshgrid(
            np.arange(-side + 1, side, 2), np.arange(-side + 1, side, 2)
        )
        positions = np.stack([px.flatten(), py.flatten()], axis=-1).astype(float)

    if packing_enum in [Packings.HEXAGON, Packings.TRIANGLE]:
        # Hexagonal/triangular packing based on square packing
        positions[::2, 1] += 1 / 2
        positions[1::2, 1] -= 1 / 2
        ratio = nl.norm(np.diff(positions[:2], axis=-1))
        positions[:, 0] /= ratio / 2
    if packing_enum == Packings.FIBONACCI:
        # Estimate helix width based on the k-space 2D surface
        # and an optimal circle packing
        positions = np.sqrt(
            Nc * 2 / CIRCLE_PACKING_DENSITY
        ) * generate_fibonacci_circle(Nc * 2)
    # Remove points by distance to fit both shape and Nc
    main_order = initialize_shape_norm(shape)
    tie_order = 2 if (main_order != 2) else np.inf  # breaking ties
    positions = np.array(positions) * np.array(spacing)
    positions = sorted(positions, key=partial(nl.norm, ord=tie_order))
    positions = sorted(positions, key=partial(nl.norm, ord=main_order))
    positions = positions[:Nc]
    return positions
