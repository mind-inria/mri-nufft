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
    projected = project_trajectory(trajectory, acq, max_iter=3000, TE_pos=0, verbose=0)

    gradients, slew_rates = compute_gradients_and_slew_rates(projected, acq)
    np.testing.assert_array_equal(projected[:, 0], 0)
    assert np.max(np.linalg.norm(gradients, axis=-1)) <= acq.gmax
    assert np.max(np.linalg.norm(slew_rates, axis=-1)) <= acq.smax


@pytest.mark.parametrize("use_boundary_gradients", [False, True])
@pytest.mark.parametrize("anisotropic", [False, True])
def test_project_trajectory_boundary_slew(use_boundary_gradients, anisotropic):
    acq = Acquisition(
        fov=(0.256, 0.192) if anisotropic else (0.256, 0.256),
        img_size=(256, 192) if not anisotropic else (256, 256),
        norm_factor=0.4,
    )
    trajectory = np.zeros((2, 20, 2))
    trajectory[0, :, 0] = np.linspace(0, 0.15, 20)
    trajectory[1, :, 1] = np.linspace(0, -0.15, 20)
    start = np.array([[0.004, -0.002], [-0.003, 0.002]])
    end = np.array([[-0.002, 0.003], [0.002, -0.004]])
    kwargs = {"start_grad": start, "end_grad": end} if use_boundary_gradients else {}
    if not use_boundary_gradients:
        start = end = np.zeros((2, 2))

    projected = project_trajectory(
        trajectory, acq, TE_pos=0, max_iter=6000, verbose=0, **kwargs
    )
    gradients, slew_rates = compute_gradients_and_slew_rates(projected, acq)
    boundary_slew = np.stack(
        (gradients[:, 0] - start, end - gradients[:, -1]), axis=1
    ) / acq.raster_time

    np.testing.assert_array_equal(projected[:, 0], 0)
    assert np.max(np.linalg.norm(gradients, axis=-1)) <= acq.gmax
    assert np.max(np.linalg.norm(slew_rates, axis=-1)) <= acq.smax
    assert np.max(np.linalg.norm(boundary_slew, axis=-1)) <= acq.smax
