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


@parametrize("osf", [1, 1.5, 2])
@parametrize_with_cases(
    "traj, shape",
    cases=[
        CasesTrajectories.case_nyquist_radial2D,
        CasesTrajectories.case_nyquist_radial3D,
    ],
)
@parametrize(backend=["gpunufft", "tensorflow", "cufinufft", "finufft"])
def test_pipe(backend, traj, shape, osf):
    """Test the pipe method."""
    distance = radial_distance(traj, shape)
    if osf != 2 and backend == "tensorflow":
        pytest.skip("OSF < 2 not supported for tensorflow.")
    if osf == 1 and "finufft" in backend:
        pytest.skip("cufinufft and finufft dont support OSF=1")
    result = pipe(traj, shape, backend=backend, osf=osf, max_iter=10)
    if backend == "cufinufft":
        result = result.get()
    result = result / np.mean(result)
    distance = distance / np.mean(distance)
    r_err = 0.2
    slope_err = None
    if osf == 2:
        r_err = 0.1
        slope_err = 0.1
    if "finufft" in backend:
        r_err *= 3
        slope_err = slope_err * 4 if slope_err is not None else None
    elif backend == "tensorflow":
        r_err = 0.5
    assert_correlate(result, distance, slope=1, slope_err=slope_err, r_value_err=r_err)
