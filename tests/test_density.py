"""Test the density compensation estimations."""
import numpy as np
from pytest_cases import parametrize_with_cases, parametrize, fixture
import pytest

from mrinufft.operators.interfaces.cufinufft import CUFINUFFT_AVAILABLE, pipe
from case_trajectories import CasesTrajectories


@parametrize_with_cases(
    "kspace_traj, shape",
    cases=[CasesTrajectories.case_radial3D],
)
@pytest.mark.skipif(not CUFINUFFT_AVAILABLE, reason="cufinufft not yet implemented")
def test_density_pipe(kspace_traj, shape):
    """Test the density compensation estimations."""
    density = pipe(kspace_traj, shape, num_iter=20).get()
    density_ref = pipe(kspace_traj, shape, num_iter=25).get()

    assert np.mean(abs(density - density_ref) ** 2) < 1e-3
