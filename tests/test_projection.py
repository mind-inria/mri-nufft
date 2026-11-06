import numpy as np
import pytest

from mrinufft.trajectories.projection import project_trajectory
from mrinufft.trajectories.utils import (
    Acquisition,
    compute_gradients_and_slew_rates,
)


@pytest.fixture(autouse=True)
def require_pyproximal():
    pytest.importorskip("pyproximal")


def _trajectory():
    trajectory = np.zeros((1, 12, 2))
    trajectory[0, :, 0] = np.linspace(-0.1, 0.1, 12)
    trajectory[0, :, 1] = 0.02 * np.sin(np.linspace(0, 2 * np.pi, 12))
    return trajectory


def test_project_trajectory_is_idempotent():
    acq = Acquisition.default
    trajectory = _trajectory()

    projected = project_trajectory(trajectory, acq, max_iter=3000, TE_pos=0, verbose=0)
    projected_twice = project_trajectory(
        projected, acq, max_iter=3000, TE_pos=0, verbose=0
    )

    np.testing.assert_allclose(projected_twice, projected, atol=1e-4, rtol=1e-3)


def test_project_trajectory_satisfies_constraints():
    acq = Acquisition.default
    projected = project_trajectory(
        _trajectory(), acq, max_iter=3000, TE_pos=0, verbose=0
    )

    gradients, slew_rates = compute_gradients_and_slew_rates(projected, acq)
    np.testing.assert_array_equal(projected[:, 0], 0)
    assert np.max(np.linalg.norm(gradients, axis=-1)) <= acq.gmax
    assert np.max(np.linalg.norm(slew_rates, axis=-1)) <= acq.smax
