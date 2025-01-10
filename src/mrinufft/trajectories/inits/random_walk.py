"""Trajectories based on random walks."""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..sampling import sample_from_density
from ..utils import KMAX


def _get_adjacent_neighbors_offsets(nb_dims: int) -> NDArray:
    return np.concatenate([np.eye(nb_dims), -np.eye(nb_dims)], axis=0).astype(int)


def _get_neighbors_offsets(nb_dims: int) -> NDArray:
    neighbors = (np.indices([3] * nb_dims) - 1).reshape((nb_dims, -1)).T
    nb_half = neighbors.shape[0] // 2
    # Remove full zero entry
    neighbors = np.concatenate([neighbors[:nb_half], neighbors[-nb_half:]], axis=0)
    return neighbors


def _initialize_ND_random_walk(
    Nc: int,
    Ns: int,
    density: NDArray,
    *,
    diagonals: bool = True,
    pseudo_random: bool = True,
    **sampling_kwargs: Any,  # noqa ANN401
) -> NDArray:
    density = density / np.sum(density)
    flat_density = np.copy(density.flatten())
    shape = density.shape
    nb_dims = len(shape)
    mask = np.ones_like(flat_density)

    # Prepare neighbor offsets once
    offsets = (
        _get_neighbors_offsets(nb_dims)
        if diagonals
        else _get_adjacent_neighbors_offsets(nb_dims)
    )

    # Make all random draws at once for performance
    draws = np.random.random((Ns, Nc))  # inverted shape for convenience

    # Initialize shot starting points
    locations = sample_from_density(Nc, density, **sampling_kwargs)
    choices = np.around((locations + KMAX) * (np.array(shape) - 1)).astype(int)
    choices = np.ravel_multi_index(choices.T, shape)
    routes = [choices]

    # Walk
    for i in range(1, Ns):
        # Express indices back to multi-dim coordinates to check bounds
        neighbors = np.array(np.unravel_index(choices, shape))
        neighbors = neighbors[:, None] + offsets.T[..., None]

        # Find out-of-bound neighbors and ignore them
        invalids = (neighbors < 0).any(axis=0) | (
            neighbors >= np.array(shape)[:, None, None]
        ).any(axis=0)
        neighbors[:, invalids] = 0
        invalids = invalids.T

        # Flatten indices to use np.random.choice
        neighbors = np.ravel_multi_index(neighbors, shape).T

        # Set walk probabilities
        walk_probs = flat_density[neighbors]
        walk_probs[invalids] = 0
        walk_probs = walk_probs / np.sum(walk_probs, axis=-1, keepdims=True)
        cum_walk_probs = np.cumsum(walk_probs, axis=-1)

        # Select next walk steps
        indices = np.argmax(draws[i][:, None] < cum_walk_probs, axis=-1)
        choices = neighbors[np.arange(Nc), indices]
        routes.append(choices)

        # Update density to account for already drawn positions
        if pseudo_random:
            flat_density[choices] = (
                mask[choices] * flat_density[choices] / (mask[choices] + 1)
            )
            mask[choices] += 1

    # Create trajectory from routes
    locations = np.indices(shape)
    locations = locations.reshape((nb_dims, -1))
    trajectory = np.array([locations[:, r].T for r in np.array(routes).T])
    trajectory = 2 * KMAX * trajectory / (np.array(shape) - 1) - KMAX
    return trajectory


def initialize_2D_random_walk(
    Nc: int,
    Ns: int,
    density: NDArray,
    *,
    diagonals: bool = True,
    pseudo_random: bool = True,
    **sampling_kwargs: Any,  # noqa ANN401
) -> NDArray:
    """Initialize a 2D random walk trajectory.

    This is an adaptation of the proposition from [Cha+14]_.
    It creates a trajectory by walking randomly to neighboring points
    following a provided sampling density.

    This implementation is different from the original proposition:
    trajectories are continuous with a fixed length instead of
    making random jumps to other locations, and an option
    is provided to have pseudo-random walks to improve coverage.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    density : NDArray
        Sampling density used to determine the walk probabilities,
        normalized automatically by its sum during the call for convenience.
    diagonals : bool, optional
        Whether to draw the next walk step from the diagonal neighbors
        on top of the adjacent ones. Default to ``True``.
    pseudo_random : bool, optional
        Whether to adapt the density dynamically to reduce areas
        already covered. The density is still statistically followed
        for undersampled acquisitions. Default to ``True``.
    **sampling_kwargs : Any
        Sampling parameters in
        ``mrinufft.trajectories.sampling.sample_from_density`` used for the
        shot starting positions.

    Returns
    -------
    NDArray
        2D random walk trajectory

    References
    ----------
    .. [Cha+14] Chauffert, Nicolas, Philippe Ciuciu,
       Jonas Kahn, and Pierre Weiss.
       "Variable density sampling with continuous trajectories"
       SIAM Journal on Imaging Sciences 7, no. 4 (2014): 1962-1992.
    """
    if len(density.shape) != 2:
        raise ValueError("`density` is expected to be 2-dimensional.")
    return _initialize_ND_random_walk(
        Nc,
        Ns,
        density,
        diagonals=diagonals,
        pseudo_random=pseudo_random,
        **sampling_kwargs,
    )


def initialize_3D_random_walk(
    Nc: int,
    Ns: int,
    density: NDArray,
    *,
    diagonals: bool = True,
    pseudo_random: bool = True,
    **sampling_kwargs: Any,  # noqa ANN401
) -> NDArray:
    """Initialize a 3D random walk trajectory.

    This is an adaptation of the proposition from [Cha+14]_.
    It creates a trajectory by walking randomly to neighboring points
    following a provided sampling density.

    This implementation is different from the original proposition:
    trajectories are continuous with a fixed length instead of
    making random jumps to other locations, and an option
    is provided to have pseudo-random walks to improve coverage.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    density : NDArray
        Sampling density used to determine the walk probabilities,
        normalized automatically by its sum during the call for convenience.
    diagonals : bool, optional
        Whether to draw the next walk step from the diagonal neighbors
        on top of the adjacent ones. Default to ``True``.
    pseudo_random : bool, optional
        Whether to adapt the density dynamically to reduce areas
        already covered. The density is still statistically followed
        for undersampled acquisitions. Default to ``True``.
    **sampling_kwargs : Any
        Sampling parameters in
        ``mrinufft.trajectories.sampling.sample_from_density`` used for the
        shot starting positions.

    Returns
    -------
    NDArray
        3D random walk trajectory

    References
    ----------
    .. [Cha+14] Chauffert, Nicolas, Philippe Ciuciu,
       Jonas Kahn, and Pierre Weiss.
       "Variable density sampling with continuous trajectories"
       SIAM Journal on Imaging Sciences 7, no. 4 (2014): 1962-1992.
    """
    if len(density.shape) != 3:
        raise ValueError("`density` is expected to be 3-dimensional.")
    return _initialize_ND_random_walk(
        Nc,
        Ns,
        density,
        diagonals=diagonals,
        pseudo_random=pseudo_random,
        **sampling_kwargs,
    )
