"""Solver for the Travelling Salesman Problem."""

import numpy as np
from numpy.typing import NDArray


def solve_tsp_with_2opt(
    locations: NDArray, improvement_threshold: float = 1e-8
) -> NDArray:
    """Solve the TSP problem using a 2-opt approach.

    A sub-optimal solution to the Travelling Salesman Problem (TSP)
    is provided using the 2-opt approach in O(n²) where chunks of
    an initially random route are reversed, and selected if the
    total distance is reduced. As a result, the route solution
    does not cross its own path in 2D.

    Parameters
    ----------
    locations : NDArray
        An array of N points with shape (N, D) where D is the space dimension.
    improvement_threshold : float, optional
        Threshold used as progress criterion to stop the optimization process.
        The default is 1e-8.

    Returns
    -------
    NDArray
        The new positions order of shape (N,).
    """
    route = np.arange(locations.shape[0])
    distances = np.linalg.norm(locations[None] - locations[:, None], axis=-1)
    route_length = np.sum(distances[0])

    improvement_factor = 1
    while improvement_factor >= improvement_threshold:
        old_route_length = route_length
        for i in range(1, len(route) - 2):
            # Check new distance by reversing chunks between i and j
            for j in range(i + 1, len(route) - 1):
                # Compute new route distance variation
                delta_length = (
                    distances[route[i - 1], route[j]]
                    + distances[route[i], route[j + 1]]
                    - distances[route[i - 1], route[i]]
                    - distances[route[j], route[j + 1]]
                )

                if delta_length < 0:
                    # Reverse route chunk
                    route = np.concatenate(
                        [route[:i], route[i : j + 1][::-1], route[j + 1 :]]
                    )
                    route_length += delta_length

        improvement_factor = abs(1 - route_length / old_route_length)
    return route
