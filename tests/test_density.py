"""Test the density compensation methods."""

import numpy as np
import numpy.testing as npt
from pytest_cases import parametrize, parametrize_with_cases

from case_trajectories import CasesTrajectories
from helpers import assert_correlate
from mrinufft.density import cell_count, radial, voronoi
from mrinufft.density.utils import _normalize_weights
from mrinufft._utils import proper_trajectory
from mrinufft.trajectories import (
    initialize_2D_radial,
    initialize_3D_phyllotaxis_radial,
)


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

    return _normalize_weights(weights)


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


@parametrize("in_out", [False, True])
@parametrize("Ns", [256, 257])
@parametrize(
    "init, dim",
    [(initialize_2D_radial, 2), (initialize_3D_phyllotaxis_radial, 3)],
)
def test_radial(init, dim, in_out, Ns):
    """Test that radial weights follow the |k|^(d-1) isotropic profile."""
    traj = init(64, Ns, in_out=in_out)
    weights = radial(traj)
    r = np.linalg.norm(proper_trajectory(traj, normalize="unit"), axis=-1)

    assert weights.shape == (traj.shape[0] * traj.shape[1],)
    assert np.all(weights > 0)
    npt.assert_allclose(np.sum(weights), 1)
    # Away from the center, the shell volume is proportional to r^(d-1).
    far = r > 0.05
    ratio = weights[far] / r[far] ** (dim - 1)
    npt.assert_allclose(ratio, np.mean(ratio), rtol=1e-3)


def test_radial_in_out_center_out():
    """Test that in-out and center-out sampling of the same points agree."""
    Nc, Ns = 32, 128
    w_co = radial(initialize_2D_radial(2 * Nc, Ns, in_out=False))
    w_io = radial(initialize_2D_radial(Nc, 2 * Ns - 1, in_out=True))
    # Same non-center samples, the center weight is split among Nc or 2Nc samples.
    w_co, w_io = np.sort(w_co)[2 * Nc :], np.sort(w_io)[Nc:]
    npt.assert_allclose(w_co, w_io, rtol=1e-6)


def test_voronoi_radial3D_dense():
    """Test that voronoi matches the radial weights on a dense 3D radial."""
    traj = initialize_3D_phyllotaxis_radial(1024, 1024)
    w_vor = voronoi(traj)
    w_rad = radial(traj)
    rel = (w_vor / np.sum(w_vor)) / w_rad - 1
    assert np.all(w_vor > 0)
    assert abs(np.median(rel)) < 5e-3
    assert np.percentile(np.abs(rel), 95) < 3e-2
