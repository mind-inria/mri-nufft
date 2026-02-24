"""Test the trajectories io module."""

import numpy as np
import pytest
from case_trajectories import CasesTrajectories
from pytest_cases import fixture, parametrize, parametrize_with_cases

from mrinufft.io import read_trajectory, write_trajectory
from mrinufft.io.pulseq import convert_trajectory_to_gradients
from mrinufft.io.utils import add_phase_to_kspace_with_shifts
from mrinufft.trajectories.gradients import get_prephasors_and_spoilers
from mrinufft.trajectories.inits import initialize_2D_radial, initialize_3D_cones
from mrinufft.trajectories.utils import Acquisition, Gammas, Hardware


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
    "name, trajectory, FOV, img_size, TE_pos, min_osf, gamma, recon_tag",
    cases=CasesIO,
)
@parametrize("version", [4.2, 5.0, 5.1])
@parametrize("postgrad", [None, "slowdown_to_center", "slowdown_to_edge"])
@parametrize("pregrad", [None, "prephase"])
@parametrize("grad_method", ["lp", "lp-minslew", "osqp"])
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
    grad_method,
):
    """Write the trajectory to a file and read it back."""
    if version < 5.1 and (postgrad is not None or pregrad is not None):
        pytest.skip("postgrad 'slowdown_to_edge' is not supported in version < 5.0")
    if (postgrad is None or pregrad is None) and grad_method != "lp":
        pytest.skip("pregrad and postgrad must be  defined to test grad_method")
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
        grad_method=grad_method,
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
    np.testing.assert_almost_equal(read_traj, trajectory, decimal=3.5)


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


@parametrize_with_cases(
    "kspace_loc, shape",
    cases=[
        CasesTrajectories.case_radial2D,
        CasesTrajectories.case_radial3D,
        CasesTrajectories.case_in_out_radial2D,
    ],
)
@parametrize("method", ["lp", "lp-minslew", "osqp"])
def test_prephasors(kspace_loc, shape, acquisition, method):
    """Test that the prephasors satisfies the gradients constraints."""
    acq = acquisition
    grad, init_points = convert_trajectory_to_gradients(kspace_loc, acq)
    prephasors = get_prephasors_and_spoilers(
        kspace_loc, acq=acq, method=method, spoil_loc=None, prephase_loc=(0, 0, 0)
    )

    assert np.all(np.abs(prephasors[:, 0, :]) <= acq.gmax)
    assert np.all(
        np.abs(np.diff(prephasors, axis=1) / acq.raster_time) <= acq.smax * 1.001
    )

    np.testing.assert_allclose(
        0,
        np.sum(prephasors, axis=1) * acq.raster_time - init_points / acq.gamma,
        atol=5e-3 / min(acq.res),
    )


@parametrize_with_cases(
    "kspace_loc, shape",
    cases=[
        CasesTrajectories.case_radial2D,
        CasesTrajectories.case_radial3D,
        CasesTrajectories.case_in_out_radial2D,
    ],
)
@parametrize("method", ["lp", "lp-minslew", "osqp"])
def test_spoilers(kspace_loc, shape, acquisition, method):
    """Test that the prephasors satisfies the gradients constraints."""
    acq = acquisition
    grad, _, end_points = convert_trajectory_to_gradients(
        kspace_loc, acq, get_final_positions=True
    )
    spoilers = get_prephasors_and_spoilers(
        kspace_loc, acq=acq, method=method, prephase_loc=None, spoil_loc=(1, 0, 0)
    )

    assert np.all(np.abs(spoilers[:, 0, :]) <= acq.gmax)
    assert np.all(
        np.abs(np.diff(spoilers, axis=1) / acq.raster_time) <= acq.smax * 1.001
    )

    np.testing.assert_allclose(
        0,
        np.sum(spoilers, axis=1) * acq.raster_time - end_points / acq.gamma,
        atol=5e-3 / min(acq.res),
    )
