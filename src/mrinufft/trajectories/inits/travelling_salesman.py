"""Trajectories based on the Travelig Salesman Problem."""

import numpy as np
import numpy.linalg as nl
from scipy.interpolate import CubicSpline
from tqdm.auto import tqdm

from ..densities import sample_from_density
from ..maths import solve_tsp_with_2opt


def _get_approx_cluster_sizes(nb_total, nb_clusters):
    cluster_sizes = round(nb_total / nb_clusters) * np.ones(nb_clusters).astype(int)
    delta_sum = nb_total - np.sum(cluster_sizes)
    cluster_sizes[: int(np.abs(delta_sum))] += np.sign(delta_sum)
    return cluster_sizes


def _sort_by_coordinate(array, coord):
    if array.shape[-1] < 3 and coord.lower() in ["z", "theta"]:
        raise ValueError(
            f"Invalid `coord`='{coord}' for arrays with less than 3 dimensions."
        )

    match coord.lower():
        case "x":
            coord = array[..., 0]
        case "y":
            coord = array[..., 1]
        case "z":
            coord = array[..., 2]
        case "r":
            coord = np.linalg.norm(array, axis=-1)
        case "phi":
            coord = np.sign(array[..., 1]) * np.arccos(
                array[..., 0] / nl.norm(array[..., :2], axis=-1)
            )
        case "theta":
            coord = np.arccos(array[..., 2] / nl.norm(array, axis=-1))
        case _:
            raise ValueError(f"Unknown coordinate `{coord}`")
    order = np.argsort(coord)
    return array[order]


def _cluster_by_coordinate(
    locations, nb_clusters, cluster_by, second_cluster_by=None, sort_by=None
):
    # Gather dimension variables
    nb_dims = locations.shape[-1]
    locations = locations.reshape((-1, nb_dims))
    nb_locations = locations.shape[0]

    # Check arguments validity
    if nb_locations % nb_clusters:
        raise ValueError("`nb_clusters` should divide the number of locations")
    cluster_size = nb_locations // nb_clusters

    # Create chunks of cluters by a first coordinate
    locations = _sort_by_coordinate(locations, cluster_by)

    if second_cluster_by:
        # Cluster each location within the chunks of clusters by a second coordinate
        chunk_sizes = _get_approx_cluster_sizes(
            nb_clusters, round(np.sqrt(nb_clusters))
        )
        chunk_ranges = np.cumsum([0] + list(chunk_sizes))
        for i in range(len(chunk_sizes)):
            i_s, i_e = (
                chunk_ranges[i] * cluster_size,
                chunk_ranges[i + 1] * cluster_size,
            )
            locations[i_s:i_e] = _sort_by_coordinate(
                locations[i_s:i_e], second_cluster_by
            )
    locations = locations.reshape((nb_clusters, cluster_size, nb_dims))

    # Order locations within each cluster by another coordinate
    if sort_by:
        for i in range(nb_clusters):
            locations[i] = _sort_by_coordinate(locations[i], sort_by)
    return locations


def _initialize_ND_travelling_salesman(
    Nc,
    Ns,
    density,
    first_cluster_by=None,
    second_cluster_by=None,
    sort_by=None,
    nb_tsp_points="auto",
    sampling="random",
    tsp_tol=1e-8,
    verbose=False,
):
    # Handle variable inputs
    nb_tsp_points = Ns if nb_tsp_points == "auto" else nb_tsp_points

    # Check arguments validity
    if Nc * nb_tsp_points > np.prod(density.shape):
        raise ValueError(
            "`density` array not large enough to peak `Nc` * `nb_tsp_points` points."
        )
    Nd = len(density.shape)

    # Select k-space locations
    density = density / np.sum(density)
    locations = sample_from_density(Nc * Ns, density, method=sampling)

    # Re-organise locations into Nc clusters
    if first_cluster_by:
        locations = _cluster_by_coordinate(
            locations,
            Nc,
            cluster_by=first_cluster_by,
            second_cluster_by=second_cluster_by,
            sort_by=sort_by,
        )

        # Compute TSP solution within each cluster/shot
        for i in tqdm(range(Nc), disable=not verbose):
            order = solve_tsp_with_2opt(locations[i], improvement_threshold=tsp_tol)
            locations[i] = locations[i][order]
    else:
        locations = (
            _sort_by_coordinate(locations, coord=sort_by) if sort_by else locations
        )

        # Compute TSP solution over the whole cloud
        order = solve_tsp_with_2opt(locations, improvement_threshold=tsp_tol)
        locations = locations[order]
        locations = locations.reshape((Nc, Ns, Nd))

    # Interpolate shot points up to full length
    trajectory = np.zeros((Nc, Ns, Nd))
    for i in range(Nc):
        cbs = CubicSpline(np.linspace(0, 1, nb_tsp_points), locations[i])
        trajectory[i] = cbs(np.linspace(0, 1, Ns))
    return trajectory


def initialize_2D_travelling_salesman(
    Nc,
    Ns,
    density,
    first_cluster_by=None,
    second_cluster_by=None,
    sort_by=None,
    nb_tsp_points="auto",
    sampling="random",
    tsp_tol=1e-8,
    verbose=False,
):
    if len(density.shape) != 2:
        raise ValueError("`density` is expected to be 2-dimensional.")
    return _initialize_ND_travelling_salesman(
        Nc,
        Ns,
        density,
        first_cluster_by=first_cluster_by,
        second_cluster_by=second_cluster_by,
        sort_by=sort_by,
        nb_tsp_points=nb_tsp_points,
        sampling=sampling,
        tsp_tol=tsp_tol,
        verbose=verbose,
    )


def initialize_3D_travelling_salesman(
    Nc,
    Ns,
    density,
    first_cluster_by=None,
    second_cluster_by=None,
    sort_by=None,
    nb_tsp_points="auto",
    sampling="random",
    tsp_tol=1e-8,
    verbose=False,
):
    if len(density.shape) != 3:
        raise ValueError("`density` is expected to be 3-dimensional.")
    return _initialize_ND_travelling_salesman(
        Nc,
        Ns,
        density,
        first_cluster_by=first_cluster_by,
        second_cluster_by=second_cluster_by,
        sort_by=sort_by,
        nb_tsp_points=nb_tsp_points,
        sampling=sampling,
        tsp_tol=tsp_tol,
        verbose=verbose,
    )
