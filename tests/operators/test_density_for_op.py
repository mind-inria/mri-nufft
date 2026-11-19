"""Specific test for testing densities specific to backend."""

import numpy as np
import pytest
from pytest import mark

from case_trajectories import CasesTrajectories
from helpers import assert_correlate, case_id, cases_of, param_backends
from mrinufft.density import pipe
from mrinufft._utils import proper_trajectory


def radial_distance(traj, shape):
    """Compute the radial distance of a trajectory."""
    proper_traj = proper_trajectory(traj, normalize="unit")
    weights = np.linalg.norm(proper_traj, axis=-1)
    return weights


@mark.parametrize("osf", [1, 1.5, 2])
@mark.parametrize(
    "case",
    cases_of(CasesTrajectories, "nyquist_radial2D", "nyquist_radial3D"),
    ids=case_id,
)
@mark.parametrize(
    "backend", param_backends("gpunufft", "tensorflow", "cufinufft", "finufft")
)
def test_pipe(backend, case, osf):
    """Test the pipe method."""
    traj, shape = case()
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
    assert_correlate(
        result,
        distance.astype(result.dtype),
        slope=1,
        slope_err=slope_err,
        r_value_err=r_err,
    )
