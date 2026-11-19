import numpy as np
import pytest
from pytest import fixture, mark

from case_trajectories import CasesTrajectories
from helpers import case_id, cases_of
from mrinufft.trajectories.projection import project_trajectory
from mrinufft.trajectories.utils import (
    Acquisition,
    compute_gradients_and_slew_rates,
)


@fixture(autouse=True)
def require_pyproximal():
    pytest.importorskip("pyproximal")


@mark.parametrize("case", cases_of(CasesTrajectories, "projection2D"), ids=case_id)
def test_project_trajectory_is_idempotent(case):
    trajectory, _ = case()
    acq = Acquisition.default

    projected = project_trajectory(trajectory, acq, max_iter=3000, TE_pos=0, verbose=0)
    projected_twice = project_trajectory(
        projected, acq, max_iter=3000, TE_pos=0, verbose=0
    )

    np.testing.assert_allclose(projected_twice, projected, atol=1e-4, rtol=1e-3)


@mark.parametrize("case", cases_of(CasesTrajectories, "projection2D"), ids=case_id)
def test_project_trajectory_satisfies_constraints(case):
    trajectory, _ = case()
    acq = Acquisition.default
    projected = project_trajectory(trajectory, acq, max_iter=3000, TE_pos=0, verbose=0)

    gradients, slew_rates = compute_gradients_and_slew_rates(projected, acq)
    np.testing.assert_array_equal(projected[:, 0], 0)
    assert np.max(np.linalg.norm(gradients, axis=-1)) <= acq.gmax
    assert np.max(np.linalg.norm(slew_rates, axis=-1)) <= acq.smax
