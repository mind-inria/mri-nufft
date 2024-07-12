"""Specific test for testing densities specific to backend."""

import numpy as np
import pytest
from pytest_cases import parametrize, parametrize_with_cases

from case_trajectories import CasesTrajectories
from helpers import assert_correlate
from mrinufft.density import pipe
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
@parametrize(backend=["gpunufft", "tensorflow"])
def test_pipe(backend, traj, shape, osf):
    """Test the pipe method."""
    if backend == "tensorflow" and osf != 2:
        pytest.skip("Tensorflow does not support OSF != 2")
    result = pipe(traj, shape, backend, osf=osf, num_iterations=10)

    distance = radial_distance(traj, shape)
    result = result / np.mean(result)
    distance = distance / np.mean(distance)
    if osf != 2:
        # If OSF < 2, we dont perfectly estimate
        assert_correlate(result, distance, slope=1, slope_err=None, r_value_err=0.2)
    else:
        assert_correlate(result, distance, slope=1, slope_err=0.1, r_value_err=0.1)
