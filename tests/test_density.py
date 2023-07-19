"""Test the density compensation estimations."""
import numpy.testing as npt
from pytest_cases import parametrize_with_cases, parametrize

from mrinufft.trajectories.density import pipe
from case_trajectories import CasesTrajectories


@parametrize_with_cases(
    "kspace_traj, shape",
    cases=[CasesTrajectories.case_radial3D],
)
@parametrize("backend", ["cufinufft", "tensorflow"])
def test_density_pipe(kspace_traj, shape, backend):
    """Test the density compensation estimations."""
    density = pipe(kspace_traj, shape, backend=backend, num_iter=20, tol=2e-7).get()
    density_ref = pipe(kspace_traj, shape, backend=backend, num_iter=25, tol=2e-7).get()

    # TODO: get tighter bounds.
    npt.assert_allclose(density, density_ref, atol=1, rtol=1)
