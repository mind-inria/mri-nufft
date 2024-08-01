"""Test the density compensation methods."""

import numpy as np
import numpy.testing as npt
from pytest_cases import parametrize, parametrize_with_cases

from case_trajectories import CasesTrajectories
from helpers import assert_correlate
from mrinufft.density import cell_count, voronoi
from mrinufft.density.utils import normalize_weights
from mrinufft._utils import proper_trajectory


def slow_cell_count2D(traj, shape, osf):
    """Perform the cell count but it is slow."""
    traj = proper_trajectory(traj, normalize="unit")
    bins = [np.linspace(-0.5, 0.5, int(osf * s) + 1) for s in shape]

    h, edges = np.histogramdd(
        traj,
        bins,
    )

    weights = np.ones(len(traj))

    bx = bins[0]
    by = bins[1]
    for i, (bxmin, bxmax) in enumerate(zip(bx[:-1], bx[1:])):
        for j, (bymin, bymax) in enumerate(zip(by[:-1], by[1:])):
            weights[
                (bxmin <= traj[:, 0])
                & (traj[:, 0] <= bxmax)
                & (bymin <= traj[:, 1])
                & (traj[:, 1] <= bymax)
            ] = h[i, j]

    return normalize_weights(weights)


def radial_distance(traj, shape):
    """Compute the radial distance of a trajectory."""
    proper_traj = proper_trajectory(traj, normalize="unit")
    weights = np.linalg.norm(proper_traj, axis=-1)
    return weights


@parametrize("osf", [1, 1.25, 2])
@parametrize_with_cases("traj, shape", cases=[CasesTrajectories.case_radial2D])
def test_cell_count2D(traj, shape, osf):
    """Test the cell count method."""
    count_ref = slow_cell_count2D(traj, shape, osf)
    count_real = cell_count(traj, shape, osf)
    npt.assert_allclose(count_real, count_ref, atol=1e-5)


@parametrize_with_cases("traj, shape", cases=[CasesTrajectories.case_radial2D])
def test_voronoi(traj, shape):
    """Test the voronoi method."""
    result = voronoi(traj)
    distance = radial_distance(traj, shape)
    result = result / np.mean(result)
    distance = distance / np.mean(distance)
    assert_correlate(result, distance, slope=1)
