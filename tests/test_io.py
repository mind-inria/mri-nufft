"""Test the trajectories io module."""

import numpy as np
from mrinufft.io import read_trajectory, write_trajectory
from mrinufft.io.utils import add_phase_to_kspace_with_shifts
from mrinufft.trajectories.trajectory2D import initialize_2D_radial
from mrinufft.trajectories.utils import Gammas, Acquisition, Hardware
from mrinufft.trajectories.tools import get_gradient_amplitudes_to_travel_for_set_time
from mrinufft.trajectories.trajectory3D import initialize_3D_cones
from pytest_cases import parametrize, parametrize_with_cases, fixture
from case_trajectories import CasesTrajectories
import pytest


class CasesIO:
    """Cases 2 for IO tests, each has different parameters."""

    def case_trajectory_2D(self):
        """Test the 2D trajectory."""
        trajectory = initialize_2D_radial(
            Nc=32, Ns=256, tilt="uniform", in_out=False
        ).astype(np.float32)
        return "2D", trajectory, (0.23, 0.23), (256, 256), 0.5, 2, Gammas.H, 1.1

    def case_trajectory_3D(self):
        """Test the 3D Trajectory."""
        trajectory = initialize_3D_cones(
            Nc=32, Ns=512, tilt="uniform", in_out=True
        ).astype(np.float32)
        return (
            "3D",
            trajectory,
            (0.23, 0.23, 0.1248),
            (64, 64, 32),
            0,
            5,
            Gammas.Na,
            1.2,
        )


@fixture(scope="module")
@parametrize("gmax", [0.1, Acquisition.default.gmax])
@parametrize("smax", [700, Acquisition.default.smax])
def acquisition(gmax, smax):
    """Create a default acquisition object."""
    return Acquisition(
        hardware=Hardware(
            gmax=gmax,
            smax=smax,
            grad_raster_time=Acquisition.default.raster_time,
        ),
        fov=(0.23, 0.23, 0.1248),
        img_size=(256, 256, 32),
        gamma=Gammas.Hydrogen,
    )


@parametrize_with_cases(
    "kspace_loc, shape",
    cases=[
        CasesTrajectories.case_radial2D,
        CasesTrajectories.case_radial3D,
        CasesTrajectories.case_in_out_radial2D,
    ],
)
def test_trajectory_state_changer_start(kspace_loc, shape, acquisition):
    """Test the trajectory state changer."""
    acq = acquisition
    dimension = len(shape)
    resolution = dimension * (0.23 / 256,)
    trajectory = kspace_loc / resolution
    gradients = np.diff(trajectory, axis=1) / acq.gamma / acq.raster_time
    GS = get_gradient_amplitudes_to_travel_for_set_time(
        kspace_end_loc=trajectory[:, 0],
        end_gradients=gradients[:, 0],
        acq=acq,
    )
    # Hardware constraints check
    assert np.all(np.abs(GS) <= acq.gmax)
    assert np.all(np.abs(np.diff(GS, axis=1) / acq.raster_time) <= acq.smax)
    assert np.all(np.abs(GS[:, -1] - gradients[:, 0]) / acq.raster_time < acq.smax)
    if np.all(trajectory[:, 0] == 0):
        # trajectory starts at the origin, check that the first gradient is zero
        assert np.all(GS.shape[1] < 10)
    assert GS.shape[1] < 200  # Checks to ensure we don't have too many samples
    # Check that ending location matches.
    np.testing.assert_allclose(
        np.sum(GS, axis=1) * acq.gamma * acq.raster_time,
        trajectory[:, 0],
        atol=1e-2 / min(resolution) / 2,
    )
    # Check that gradients match.
    np.testing.assert_allclose(GS[:, 0], 0, atol=1e-5)


@parametrize_with_cases(
    "kspace_loc, shape",
    cases=[
        CasesTrajectories.case_radial2D,
        CasesTrajectories.case_radial3D,
        CasesTrajectories.case_in_out_radial2D,
    ],
)
def test_trajectory_state_changer_end(
    kspace_loc,
    shape,
    acquisition,
):
    """Test the trajectory state changer for the end of the trajectory."""
    acq = acquisition
    dimension = len(shape)
    resolution = dimension * (0.23 / 256,)
    trajectory = kspace_loc / resolution
    gradients = np.diff(trajectory, axis=1) / acq.gamma / acq.raster_time
    GE = get_gradient_amplitudes_to_travel_for_set_time(
        kspace_start_loc=trajectory[:, -1],
        kspace_end_loc=np.zeros_like(trajectory[:, -1]),
        start_gradients=gradients[:, -1],
        acq=acq,
    )
    # Hardware constraints check
    assert np.all(np.abs(GE) <= acq.gmax)
    assert np.all(np.abs(np.diff(GE, axis=1) / acq.raster_time) <= acq.smax)
    assert np.all(np.abs(GE[:, -1]) / acq.raster_time < acq.smax)
    assert GE.shape[1] < 200  # Checks to ensure we don't have too many samples
    # Check that ending location matches.
    np.testing.assert_allclose(
        0,
        trajectory[:, -1] + np.sum(GE, axis=1) * acq.gamma * acq.raster_time,
        atol=1e-2 / min(resolution) / 2,
    )
    # Check that gradients match.
    np.testing.assert_allclose(GE[:, 0], gradients[:, -1], atol=1e-5)


@parametrize_with_cases(
    "name, trajectory, FOV, img_size, TE_pos, min_osf, gamma, recon_tag",
    cases=CasesIO,
)
@parametrize("version", [4.2, 5.0, 5.1])
@parametrize("postgrad", [None, "slowdown_to_center", "slowdown_to_edge"])
@parametrize("pregrad", [None, "prephase"])
def test_write_n_read(
    name,
    trajectory,
    FOV,
    img_size,
    TE_pos,
    min_osf,
    gamma,
    recon_tag,
    tmp_path,
    version,
    postgrad,
    pregrad,
):
    """Write the trajectory to a file and read it back."""
    if version < 5.1 and (postgrad is not None or pregrad is not None):
        pytest.skip("postgrad 'slowdown_to_edge' is not supported in version < 5.0")
    """Test function which writes the trajectory and reads it back."""
    if np.all(trajectory[:, 0] == 0) and pregrad is not None:
        pytest.skip("We dont need prephasors for UTE trajectories")

    write_trajectory(
        trajectory=trajectory,
        FOV=FOV,
        img_size=img_size,
        check_constraints=True,
        grad_filename=str(tmp_path / name),
        TE_pos=TE_pos,
        version=version,
        min_osf=min_osf,
        recon_tag=recon_tag,
        gamma=gamma,
        pregrad=pregrad,
        postgrad=postgrad,
    )
    read_traj, params = read_trajectory(
        str((tmp_path / name).with_suffix(".bin")), gamma=gamma, read_shots=True
    )
    np.testing.assert_allclose(params["version"], version)
    assert params["num_shots"] == trajectory.shape[0]
    assert params["num_samples_per_shot"] == trajectory.shape[1] - 1
    np.testing.assert_almost_equal(params["TE_pos"], TE_pos)
    np.testing.assert_allclose(params["gamma"], gamma)
    np.testing.assert_allclose(params["recon_tag"], recon_tag)
    assert params["min_osf"] == min_osf
    np.testing.assert_almost_equal(params["FOV"], FOV, decimal=6)
    np.testing.assert_equal(params["img_size"], img_size)
    np.testing.assert_almost_equal(read_traj, trajectory, decimal=4)


@parametrize_with_cases(
    "kspace_loc, shape",
    cases=[CasesTrajectories.case_random2D, CasesTrajectories.case_random3D],
)
def test_add_shift(kspace_loc, shape):
    """Test the add_phase_to_kspace_with_shifts function."""
    n_samples = np.prod(kspace_loc.shape[:-1])
    kspace_data = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    shifts = np.random.rand(kspace_loc.shape[-1])

    shifted_data = add_phase_to_kspace_with_shifts(kspace_data, kspace_loc, shifts)

    assert np.allclose(np.abs(shifted_data), np.abs(kspace_data))

    phase = np.exp(-2 * np.pi * 1j * np.sum(kspace_loc * shifts, axis=-1))
    np.testing.assert_almost_equal(shifted_data / phase, kspace_data, decimal=5)
