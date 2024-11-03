"""Trajectories based on random walks."""

import numpy as np

from ..utils import KMAX


def _get_adjacent_neighbors_flat_offsets(shape):
    nb_dims = len(shape)
    neighborhood = np.indices([3] * nb_dims) - 1
    distances = np.sum(np.abs(neighborhood), axis=0)

    center = np.ravel_multi_index([1] * nb_dims, dims=shape)
    neighbors = np.ravel_multi_index(np.where(distances == 1), dims=shape) - center
    return neighbors


def _get_diagonal_neighbors_flat_offsets(shape):
    nb_dims = len(shape)
    neighborhood = np.indices([3] * nb_dims) - 1
    distances = np.sum(np.abs(neighborhood), axis=0)

    center = np.ravel_multi_index([1] * nb_dims, dims=shape)
    neighbors = np.ravel_multi_index(np.where(distances > 1), dims=shape) - center
    return neighbors


def _initialize_ND_random_walk(Nc, Ns, density, *, diagonals=True, pseudo_random=True):
    flat_density = np.copy(density.flatten())
    max_id = np.prod(density.shape)
    mask = np.ones_like(flat_density)

    # Prepare neighbor offsets once
    offsets = _get_adjacent_neighbors_flat_offsets(density.shape)
    if diagonals:
        offsets = np.concatenate(
            [offsets, _get_diagonal_neighbors_flat_offsets(density.shape)]
        )

    # Make all random draws at once for performance
    draws = np.random.random((Ns, Nc))  # inverted shape for convenience

    # Initialize shot starting points
    choices = np.random.choice(np.arange(len(flat_density)), size=Nc, p=flat_density)
    routes = [choices]

    # Walk
    for i in range(1, Ns):
        neighbors = choices[:, None] + offsets[None]

        # Find out-of-bound neighbors and ignore them
        invalids = (neighbors < 0) | (neighbors > max_id)
        neighbors[invalids] = 0

        # Set walk probabilities
        walk_probs = flat_density[neighbors]
        walk_probs[invalids] = 0
        walk_probs = walk_probs / np.sum(walk_probs, axis=-1, keepdims=True)
        cum_walk_probs = np.cumsum(walk_probs, axis=-1)

        # Select next walk steps
        indices = np.argmax(draws[i][:, None] < cum_walk_probs, axis=-1)
        choices = neighbors[np.arange(Nc), indices]
        routes.append(choices)

        # Update density to account for already drawed positions
        if pseudo_random:
            flat_density[choices] = (
                mask[choices] * flat_density[choices] / (mask[choices] + 1)
            )
            mask[choices] += 1
    routes = np.array(routes).T

    # Create trajectory from routes
    locations = np.indices(density.shape)
    locations = locations.reshape((len(density.shape), -1))
    trajectory = np.array([locations[:, r].T for r in routes])
    trajectory = 2 * KMAX * trajectory / density.shape - KMAX
    return trajectory


def initialize_2D_random_walk(Nc, Ns, density, *, diagonals=True, pseudo_random=True):
    """Initialize a 2D random walk trajectory.

    It creates a trajectory by walking randomly to neighboring points
    following a provided sampling density.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    density : array_like
        Sampling density used to determine the walk probabilities.
    diagonals : bool, optional
        Whether to draw the next walk step from the diagional neighbors
        on top of the adjacent ones. Default to True.
    pseudo_random : bool, optional
        Whether to adapt the density dynamically to reduce areas
        already covered. The density is still statistically followed
        for undersampled acquisitions. Default to True.

    Returns
    -------
    array_like
        2D random walk trajectory
    """
    if len(density.shape) != 2:
        raise ValueError("`density` is expected to be 2-dimensional.")
    return _initialize_ND_random_walk(
        Nc, Ns, density, diagonals=diagonals, pseudo_random=pseudo_random
    )


def initialize_3D_random_walk(Nc, Ns, density, *, diagonals=True, pseudo_random=True):
    """Initialize a 3D random walk trajectory.

    It creates a trajectory by walking randomly to neighboring points
    following a provided sampling density.

    Parameters
    ----------
    Nc : int
        Number of shots
    Ns : int
        Number of samples per shot
    density : array_like
        Sampling density used to determine the walk probabilities.
    diagonals : bool, optional
        Whether to draw the next walk step from the diagional neighbors
        on top of the adjacent ones. Default to True.
    pseudo_random : bool, optional
        Whether to adapt the density dynamically to reduce areas
        already covered. The density is still statistically followed
        for undersampled acquisitions. Default to True.

    Returns
    -------
    array_like
        3D random walk trajectory
    """
    if len(density.shape) != 3:
        raise ValueError("`density` is expected to be 3-dimensional.")
    return _initialize_ND_random_walk(
        Nc, Ns, density, diagonals=diagonals, pseudo_random=pseudo_random
    )
