"""Trajectories based on the Travelling Salesman Problem."""

from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.linalg as nl
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from tqdm.auto import tqdm

from ..maths import solve_tsp_with_2opt
from ..sampling import sample_from_density
from ..tools import oversample

Coordinate: TypeAlias = Literal["x", "y", "z", "r", "phi", "theta"]


def _get_approx_cluster_sizes(nb_total: int, nb_clusters: int) -> NDArray:
    # Give a list of cluster sizes close to sqrt(`nb_total`)
    cluster_sizes = round(nb_total / nb_clusters) * np.ones(nb_clusters).astype(int)
    delta_sum = nb_total - np.sum(cluster_sizes)
    cluster_sizes[: int(np.abs(delta_sum))] += np.sign(delta_sum)
    return cluster_sizes


def _sort_by_coordinate(array: NDArray, coord: Coordinate) -> NDArray:
    # Sort a list of N-D locations by a Cartesian/spherical coordinate
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
    locations: NDArray,
    nb_clusters: int,
    cluster_by: Coordinate,
    second_cluster_by: Coordinate | None = None,
    sort_by: Coordinate | None = None,
) -> NDArray:
    # Cluster approximately a list of N-D locations by Cartesian/spherical coordinates
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
    Nc: int,
    Ns: int,
    density: NDArray,
    first_cluster_by: Coordinate | None = None,
    second_cluster_by: Coordinate | None = None,
    sort_by: Coordinate | None = None,
    tsp_tol: float = 1e-8,
    *,
    verbose: bool = False,
    **sampling_kwargs: Any,  # noqa ANN401
) -> NDArray:
    # Check arguments validity
    if Nc * Ns > np.prod(density.shape):
        raise ValueError("`density` array not large enough to pick `Nc` * `Ns` points.")
    Nd = len(density.shape)

    # Select k-space locations
    trajectory = sample_from_density(Nc * Ns, density, **sampling_kwargs)

    # Re-organise locations into Nc clusters
    if first_cluster_by:
        trajectory = _cluster_by_coordinate(
            trajectory,
            Nc,
            cluster_by=first_cluster_by,
            second_cluster_by=second_cluster_by,
            sort_by=sort_by,
        )

        # Compute TSP solution within each cluster/shot
        for i in tqdm(range(Nc), disable=not verbose):
            order = solve_tsp_with_2opt(trajectory[i], improvement_threshold=tsp_tol)
            trajectory[i] = trajectory[i][order]
    else:
        trajectory = (
            _sort_by_coordinate(trajectory, coord=sort_by) if sort_by else trajectory
        )

        # Compute TSP solution over the whole cloud
        order = solve_tsp_with_2opt(trajectory, improvement_threshold=tsp_tol)
        trajectory = trajectory[order]
        trajectory = trajectory.reshape((Nc, Ns, Nd))

    return trajectory


def initialize_2D_travelling_salesman(
    Nc: int,
    Ns: int,
    density: NDArray,
    first_cluster_by: Coordinate | None = None,
    second_cluster_by: Coordinate | None = None,
    sort_by: Coordinate | None = None,
    tsp_tol: float = 1e-8,
    *,
    verbose: bool = False,
    **sampling_kwargs: Any,  # noqa ANN401
) -> NDArray:
    """
    Initialize a 2D trajectory using a Travelling Salesman Problem (TSP)-based path.

    This is a reproduction of the work from [Cha+14]_. The TSP solution
    is obtained using the 2-opt method in O(n²). An additional option
    is provided to cluster shots before solving the TSP and thus
    reduce drastically the computation time. The initial sampling method
    can also be customized.

    Parameters
    ----------
    Nc : int
        The number of clusters (or shots) to divide the trajectory into.
    Ns : int
        The number of points per cluster.
    density : NDArray
        A 2-dimensional density array from which points are sampled.
    first_cluster_by : {"x", "y", "z", "r", "phi", "theta"}, optional
        The coordinate used to cluster points initially, by default ``None``.
    second_cluster_by : {"x", "y", "z", "r", "phi", "theta"}, optional
        A secondary coordinate used for clustering within primary clusters,
        by default ``None``.
    sort_by : {"x", "y", "z", "r", "phi", "theta"}, optional
        The coordinate by which to order points within each cluster,
        by default ``None``.
    tsp_tol : float, optional
        Convergence tolerance for the TSP solution, by default ``1e-8``.
    verbose : bool, optional
        If ``True``, displays a progress bar, by default ``False``.
    **sampling_kwargs : dict, optional
        Additional arguments to pass to
        ``mrinufft.trajectories.sampling.sample_from_density``.

    Returns
    -------
    NDArray
        A 2D array representing the TSP-ordered trajectory.

    Raises
    ------
    ValueError
        If ``density`` is not a 2-dimensional array.

    References
    ----------
    .. [Cha+14] Chauffert, Nicolas, Philippe Ciuciu,
       Jonas Kahn, and Pierre Weiss.
       "Variable density sampling with continuous trajectories"
       SIAM Journal on Imaging Sciences 7, no. 4 (2014): 1962-1992.
    """
    if len(density.shape) != 2:
        raise ValueError("`density` is expected to be 2-dimensional.")
    return _initialize_ND_travelling_salesman(
        Nc,
        Ns,
        density,
        first_cluster_by=first_cluster_by,
        second_cluster_by=second_cluster_by,
        sort_by=sort_by,
        tsp_tol=tsp_tol,
        verbose=verbose,
        **sampling_kwargs,
    )


def initialize_3D_travelling_salesman(
    Nc: int,
    Ns: int,
    density: NDArray,
    first_cluster_by: Coordinate | None = None,
    second_cluster_by: Coordinate | None = None,
    sort_by: Coordinate | None = None,
    tsp_tol: float = 1e-8,
    *,
    verbose: bool = False,
    **sampling_kwargs: Any,  # noqa ANN401
) -> NDArray:
    """
    Initialize a 3D trajectory using a Travelling Salesman Problem (TSP)-based path.

    This is a reproduction of the work from [Cha+14]_. The TSP solution
    is obtained using the 2-opt method with a complexity in O(n²) in time
    and memory.

    An additional option is provided to cluster shots before solving the
    TSP and thus reduce drastically the computation time. The initial
    sampling method can also be customized.

    Parameters
    ----------
    Nc : int
        The number of clusters (or shots) to divide the trajectory into.
    Ns : int
        The number of points per cluster.
    density : NDArray
        A 3-dimensional density array from which points are sampled.
    first_cluster_by : {"x", "y", "z", "r", "phi", "theta"}, optional
        The coordinate used to cluster points initially, by default ``None``.
    second_cluster_by : {"x", "y", "z", "r", "phi", "theta"}, optional
        A secondary coordinate used for clustering within primary clusters,
        by default ``None``.
    sort_by : {"x", "y", "z", "r", "phi", "theta"}, optional
        The coordinate by which to order points within each cluster,
        by default ``None``.
    tsp_tol : float, optional
        Convergence tolerance for the TSP solution, by default ``1e-8``.
    verbose : bool, optional
        If ``True``, displays a progress bar, by default ``False``.
    **sampling_kwargs : dict, optional
        Additional arguments to pass to
        ``mrinufft.trajectories.sampling.sample_from_density``.

    Returns
    -------
    NDArray
        A 3D array representing the TSP-ordered trajectory.

    Raises
    ------
    ValueError
        If ``density`` is not a 3-dimensional array.

    References
    ----------
    .. [Cha+14] Chauffert, Nicolas, Philippe Ciuciu,
       Jonas Kahn, and Pierre Weiss.
       "Variable density sampling with continuous trajectories."
       SIAM Journal on Imaging Sciences 7, no. 4 (2014): 1962-1992.
    """
    if len(density.shape) != 3:
        raise ValueError("`density` is expected to be 3-dimensional.")
    return _initialize_ND_travelling_salesman(
        Nc,
        Ns,
        density,
        first_cluster_by=first_cluster_by,
        second_cluster_by=second_cluster_by,
        sort_by=sort_by,
        tsp_tol=tsp_tol,
        verbose=verbose,
        **sampling_kwargs,
    )
