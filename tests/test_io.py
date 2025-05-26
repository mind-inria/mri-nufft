"""Test the trajectories io module."""

import numpy as np
from mrinufft.io import read_trajectory, write_trajectory
from mrinufft.io.utils import add_phase_to_kspace_with_shifts
from mrinufft.trajectories.trajectory2D import initialize_2D_radial
from mrinufft.trajectories.utils import (
    Gammas,
    DEFAULT_GMAX,
    DEFAULT_SMAX,
    DEFAULT_RASTER_TIME,
)
from mrinufft.trajectories.tools import get_gradients_for_set_time
from mrinufft.trajectories.trajectory3D import initialize_3D_cones
from pytest_cases import parametrize, parametrize_with_cases
from case_trajectories import CasesTrajectories
import pytest


class CasesIO:
    """Cases 2 for IO tests, each has different parameters."""

    def case_trajectory_2D(self):
        """Test the 2D trajectory."""
        trajectory = initialize_2D_radial(
            Nc=32, Ns=256, tilt="uniform", in_out=False
        ).astype(np.float32)
        return "2D", trajectory, (0.23, 0.23), (256, 256), False, 2, 42.576e3, 1.1

    def case_trajectory_3D(self):
        """Test the 3D Trajectory."""
        trajectory = initialize_3D_cones(
            Nc=32, Ns=256, tilt="uniform", in_out=True
        ).astype(np.float32)
        return (
            "3D",
            trajectory,
            (0.23, 0.23, 0.1248),
            (256, 256, 128),
            True,
            5,
            10e3,
            1.2,
        )


@parametrize("gamma", [Gammas.Hydrogen])
@parametrize("raster_time", [DEFAULT_RASTER_TIME])
@parametrize_with_cases(
    "kspace_loc, shape",
    cases=[
        CasesTrajectories.case_radial2D,
        CasesTrajectories.case_radial3D,
        CasesTrajectories.case_in_out_radial2D,
    ],
)
@parametrize("gmax", [0.1, DEFAULT_GMAX])
@parametrize("smax", [0.7, DEFAULT_SMAX])
def test_trajectory_state_changer(kspace_loc, shape, gamma, raster_time, gmax, smax):
    """Test the trajectory state changer."""
    dimension = len(shape)
    resolution = dimension * (0.23 / 256,)
    trajectory = kspace_loc / resolution
    gradients = np.diff(trajectory, axis=1) / gamma / raster_time
    GS = get_gradients_for_set_time(
        ke=trajectory[:, 0],
        ge=gradients[:, 0],
        gamma=gamma,
        raster_time=raster_time,
        gmax=gmax,
        smax=smax,
    )
    # Hardware constraints check
    assert np.all(np.abs(GS) <= gmax)
    assert np.all(np.abs(np.diff(GS, axis=1) / raster_time) <= smax)
    assert np.all(np.abs(GS[:, -1] - gradients[:, 0]) / raster_time < smax)
    # Check that ending location matches.
    np.testing.assert_allclose(
        np.sum(GS, axis=1) * gamma * raster_time,
        trajectory[:, 0],
        atol=1e-2 / min(resolution) / 2,
    )
    # Check that gradients match.
    np.testing.assert_allclose(GS[:, 0], 0, atol=1e-5)

    GE = get_gradients_for_set_time(
        ks=trajectory[:, -1],
        ke=np.zeros_like(trajectory[:, -1]),
        gs=gradients[:, -1],
        gamma=gamma,
        raster_time=raster_time,
        gmax=gmax,
        smax=smax,
    )
    # Hardware constraints check
    assert np.all(np.abs(GE) <= gmax)
    assert np.all(np.abs(np.diff(GE, axis=1) / raster_time) <= smax)
    assert np.all(np.abs(GE[:, -1]) / raster_time < smax)
    # Check that ending location matches.
    np.testing.assert_allclose(
        0,
        trajectory[:, -1] + np.sum(GE, axis=1) * gamma * raster_time,
        atol=1e-2 / min(resolution) / 2,
    )
    # Check that gradients match.
    np.testing.assert_allclose(GE[:, 0], gradients[:, -1], atol=1e-5)


@parametrize_with_cases(
    "name, trajectory, FOV, img_size, in_out, min_osf, gamma, recon_tag",
    cases=CasesIO,
)
@parametrize("version", [4.2, 5.0, 5.1])
@parametrize("postgrad", [None, "slowdown_to_center", "slowdown_to_edge"])
def test_write_n_read(
    name,
    trajectory,
    FOV,
    img_size,
    in_out,
    min_osf,
    gamma,
    recon_tag,
    tmp_path,
    version,
    postgrad,
):
    if version < 5.0 and (postgrad is not None):
        pytest.skip("postgrad 'slowdown_to_edge' is not supported in version < 5.0")
    """Test function which writes the trajectory and reads it back."""
    write_trajectory(
        trajectory=trajectory,
        FOV=FOV,
        img_size=img_size,
        check_constraints=True,
        grad_filename=str(tmp_path / name),
        TE=0.5 if in_out else 0,
        version=version,
        min_osf=min_osf,
        recon_tag=recon_tag,
        gamma=gamma,
        pregrad="prephase" if version < 5.0 else None,
        postgrad=postgrad,
    )
    read_traj, params = read_trajectory(
        str((tmp_path / name).with_suffix(".bin")), gamma=gamma, read_shots=True
    )
    assert params["version"] == version
    assert params["num_shots"] == trajectory.shape[0]
    assert params["num_samples_per_shot"] == trajectory.shape[1] - 1
    assert params["TE"] == (0.5 if in_out else 0)
    assert params["gamma"] == gamma
    assert params["recon_tag"] == recon_tag
    assert params["min_osf"] == min_osf
    np.testing.assert_almost_equal(params["FOV"], FOV, decimal=6)
    np.testing.assert_equal(params["img_size"], img_size)
    np.testing.assert_almost_equal(read_traj, trajectory, decimal=5)


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
