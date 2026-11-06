import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from case_trajectories import CasesTrajectories
from mrinufft.trajectories.projection import project_trajectory
from mrinufft.trajectories.utils import (
    Acquisition,
    compute_gradients_and_slew_rates,
)


@pytest.fixture(autouse=True)
def require_pyproximal():
    pytest.importorskip("pyproximal")


@parametrize_with_cases("trajectory, _", cases=CasesTrajectories.case_projection2D)
def test_project_trajectory_is_idempotent(trajectory, _):
    acq = Acquisition.default

    projected = project_trajectory(trajectory, acq, max_iter=3000, TE_pos=0, verbose=0)
    projected_twice = project_trajectory(
        projected, acq, max_iter=3000, TE_pos=0, verbose=0
    )

    np.testing.assert_allclose(projected_twice, projected, atol=1e-4, rtol=1e-3)


@parametrize_with_cases("trajectory, _", cases=CasesTrajectories.case_projection2D)
def test_project_trajectory_satisfies_constraints(trajectory, _):
    acq = Acquisition.default
    projected = project_trajectory(
        trajectory, acq, max_iter=3000, TE_pos=0, verbose=0
    )

    gradients, slew_rates = compute_gradients_and_slew_rates(projected, acq)
    np.testing.assert_array_equal(projected[:, 0], 0)
    assert np.max(np.linalg.norm(gradients, axis=-1)) <= acq.gmax
    assert np.max(np.linalg.norm(slew_rates, axis=-1)) <= acq.smax
