"""Test the density compensation estimations."""
import numpy as np
import numpy.testing as npt
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
    density = pipe(kspace_traj, shape, num_iter=20, tol=2e-7).get()
    density_ref = pipe(kspace_traj, shape, num_iter=25, tol=2e-7).get()

    # TODO: get tighter bounds.
    npt.assert_allclose(density, density_ref, atol=1, rtol=1)
