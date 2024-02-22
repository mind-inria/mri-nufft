"""Specific test for gpunufft."""

import numpy as np
import numpy.testing as npt
from pytest_cases import parametrize, parametrize_with_cases

from case_trajectories import CasesTrajectories
from helpers import assert_correlate
from mrinufft.density import cell_count, voronoi, pipe
from mrinufft.density.utils import normalize_weights
from mrinufft._utils import proper_trajectory


def radial_distance(traj, shape):
    """Compute the radial distance of a trajectory."""
    proper_traj = proper_trajectory(traj, normalize="unit")
    weights = np.linalg.norm(proper_traj, axis=-1)
    return weights


@parametrize("osf", [1, 1.25, 2])
@parametrize_with_cases(
    "traj, shape",
    cases=[
        CasesTrajectories.case_nyquist_radial2D,
        CasesTrajectories.case_nyquist_radial3D,
    ],
)
@parametrize(backend=["gpunufft"])
def test_pipe(backend, traj, shape, osf):
    """Test the pipe method."""
    distance = radial_distance(traj, shape)
    result = pipe(traj, shape, osf=osf, num_iterations=10)
    result = result / np.mean(result)
    distance = distance / np.mean(distance)
    if osf != 2:
        # If OSF < 2, we dont perfectly estimate
        assert_correlate(result, distance, slope=1, slope_err=None, r_value_err=0.2)
    else:
        assert_correlate(result, distance, slope=1, slope_err=0.1, r_value_err=0.1)
