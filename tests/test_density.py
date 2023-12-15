"""Test the density compensation methods."""


import numpy as np
import numpy.testing as npt
from pytest_cases import fixture, parametrize, parametrize_with_cases

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


@fixture(scope="module")
def radial_distance():
    """Compute the radial distance of a trajectory."""
    traj, shape = CasesTrajectories().case_radial2D()

    proper_traj = proper_trajectory(traj, normalize="unit")
    weights = 2 * np.pi * np.sqrt(proper_traj[:, 0] ** 2 + proper_traj[:, 1] ** 2)

    return normalize_weights(weights)


@parametrize("osf", [1, 1.25, 2])
@parametrize_with_cases("traj, shape", cases=[CasesTrajectories.case_radial2D])
def test_cell_count2D(traj, shape, osf):
    """Test the cell count method."""
    count_ref = slow_cell_count2D(traj, shape, osf)
    count_real = cell_count(traj, shape, osf)
    npt.assert_allclose(count_real, count_ref, atol=1e-5)


@parametrize_with_cases("traj, shape", cases=[CasesTrajectories.case_radial2D])
def test_voronoi(traj, shape, radial_distance):
    """Test the voronoi method."""
    result = voronoi(traj)

    assert_correlate(result, radial_distance, slope=2 * np.pi)


def test_pipe():
    """Test the pipe method."""
    pass
