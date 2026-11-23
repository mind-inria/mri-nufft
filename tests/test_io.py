"""Test the io module."""

import numpy as np
import pytest
from pytest import fixture, mark
from helpers import CasesTrajectories, case_id, cases_of

from scipy.io import savemat

from mrinufft.io import read_girf, read_gradients, read_trajectory, write_trajectory
from mrinufft.io import add_phase_to_kspace_with_shifts
from mrinufft.io.siemens import twix2nifti_affine
from mrinufft.trajectories import (
    get_prephasors_and_spoilers,
    convert_trajectory_to_gradients,
    initialize_2D_radial,
    initialize_3D_cones,
    Acquisition,
    Gammas,
    Hardware,
)


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


@fixture(
    scope="module",
    params=[(Acquisition.default.gmax, Acquisition.default.smax), (0.1, 700)],
    ids=["default_hardware", "fast_hardware"],
)
def acquisition(request):
    """Create an acquisition object."""
    gmax, smax = request.param
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


@mark.parametrize("case", cases_of(CasesIO), ids=case_id)
@mark.parametrize("version", [4.2, 5.0, 5.1])
@mark.parametrize("postgrad", [None, "slowdown_to_center", "slowdown_to_edge"])
@mark.parametrize("pregrad", [None, "prephase"])
@mark.parametrize("grad_method", ["lp", "lp-minslew", "osqp"])
def test_write_n_read(
    case,
    tmp_path,
    version,
    postgrad,
    pregrad,
    grad_method,
):
    """Write the trajectory to a file and read it back."""
    name, trajectory, FOV, img_size, TE_pos, min_osf, gamma, recon_tag = case()
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


@mark.parametrize("pad_to_multiple_of_4, expected_samples", [(True, 8), (False, 5)])
def test_write_trajectory_pad_samples(
    caplog, tmp_path, pad_to_multiple_of_4, expected_samples
):
    """Pad gradient samples to a multiple of four when requested."""
    trajectory = np.zeros((1, 6, 2), dtype=np.float32)
    trajectory[0, :, 0] = np.linspace(-0.2, 0.2, 6)
    write_trajectory(
        trajectory=trajectory,
        FOV=(0.23, 0.23),
        img_size=(256, 256),
        grad_filename=str(tmp_path / "trajectory"),
        check_constraints=False,
        pad_to_multiple_of_4=pad_to_multiple_of_4,
    )

    read_traj, params = read_trajectory(
        str(tmp_path / "trajectory.bin"), read_shots=True
    )
    assert params["num_samples_per_shot"] == expected_samples
    if pad_to_multiple_of_4:
        expected_padding = np.broadcast_to(read_traj[0, -1], (4, 2))
        np.testing.assert_allclose(read_traj[0, -4:], expected_padding)
    else:
        assert "not a multiple of 4" in caplog.text


@mark.parametrize("case", cases_of(CasesIO), ids=case_id)
@mark.parametrize("version", [4.2, 5.0, 5.1])
def test_read_trajectory_acquisition_and_rewrite(case, tmp_path, version):
    """Read acquisition metadata and pass the complete parameter dict to the writer."""
    _, trajectory, fov, img_size, _, _, _, _ = case()
    gamma = Gammas.Na / 1e3
    write_trajectory(
        trajectory,
        FOV=fov,
        img_size=img_size,
        grad_filename=str(tmp_path / "original"),
        gamma=gamma,
        raster_time=0.02,
        norm_factor=0.4,
        min_osf=4,
        version=version,
        check_constraints=False,
    )
    trajectory, params = read_trajectory(
        str(tmp_path / "original.bin"),
        raster_time=0.02,
        dwell_time=0.02,
        normalize_factor=0.4,
        read_shots=True,
    )
    acq = params["acq"]
    assert isinstance(acq, Acquisition)
    np.testing.assert_allclose(acq.fov, fov)
    np.testing.assert_equal(acq.img_size, img_size)
    assert acq.gamma == pytest.approx(Gammas.Na)
    assert acq.adc_dwell_time == pytest.approx(20e-6)
    assert acq.raster_time == pytest.approx(20e-6)
    assert acq.norm_factor == 0.4
    assert acq.hardware.gmax == Hardware().gmax
    assert acq.hardware.smax == Hardware().smax

    _, oversampled_params = read_trajectory(
        str(tmp_path / "original.bin"), raster_time=0.02, dwell_time="min_osf"
    )
    assert oversampled_params["acq"].adc_dwell_time == pytest.approx(5e-6)

    write_trajectory(trajectory, grad_filename=str(tmp_path / "rewritten"), **params)
    rewritten, rewritten_params = read_trajectory(
        str(tmp_path / "rewritten.bin"),
        raster_time=0.02,
        dwell_time=0.02,
        normalize_factor=0.4,
        read_shots=True,
    )
    np.testing.assert_allclose(rewritten, trajectory, atol=1e-5)
    assert rewritten_params["acq"].gamma == pytest.approx(acq.gamma)


@mark.parametrize("version", [4.0, 4.2])
def test_read_trajectory_acquisition_unnormalized(tmp_path, version):
    """Unnormalized legacy files remain readable without geometry metadata."""
    trajectory = initialize_2D_radial(Nc=2, Ns=8).astype(np.float32)
    write_trajectory(
        trajectory,
        FOV=(0.23, 0.24),
        img_size=(64, 80),
        grad_filename=str(tmp_path / "trajectory"),
        version=version,
        check_constraints=False,
    )
    _, params = read_trajectory(str(tmp_path / "trajectory.bin"), normalize_factor=None)
    if version < 4.1:
        assert params["acq"] is None
    else:
        assert params["acq"].norm_factor == Acquisition.default.norm_factor


@mark.parametrize(
    "case", cases_of(CasesTrajectories, "random2D", "random3D"), ids=case_id
)
def test_add_shift(case):
    """Test the add_phase_to_kspace_with_shifts function."""
    kspace_loc, shape = case()
    n_samples = np.prod(kspace_loc.shape[:-1])
    kspace_data = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    shifts = np.random.rand(kspace_loc.shape[-1])

    shifted_data = add_phase_to_kspace_with_shifts(kspace_data, kspace_loc, shifts)

    assert np.allclose(np.abs(shifted_data), np.abs(kspace_data))

    phase = np.exp(-2 * np.pi * 1j * np.sum(kspace_loc * shifts, axis=-1))
    np.testing.assert_almost_equal(shifted_data / phase, kspace_data, decimal=5)


@mark.parametrize(
    "case",
    cases_of(CasesTrajectories, "radial2D", "radial3D", "in_out_radial2D"),
    ids=case_id,
)
@mark.parametrize("method", ["lp", "lp-minslew", "osqp"])
def test_prephasors(case, acquisition, method):
    """Test that the prephasors satisfies the gradients constraints."""
    kspace_loc, shape = case()
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


@mark.parametrize(
    "case",
    cases_of(CasesTrajectories, "radial2D", "radial3D", "in_out_radial2D"),
    ids=case_id,
)
@mark.parametrize("method", ["lp", "lp-minslew", "osqp"])
def test_spoilers(case, acquisition, method):
    """Test that the prephasors satisfies the gradients constraints."""
    kspace_loc, shape = case()
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


def test_read_girf(tmp_path):
    """The GIRF is read one-sided, Hermitian, in Hz, and cut above fmax."""
    df = 0.05  # kHz
    f = np.arange(-400, 400) * df
    tau, T = 3e-6, 20e-6
    hermitian = np.exp(-2j * np.pi * f * 1e3 * tau) / (1 + 2j * np.pi * f * 1e3 * T)
    girf = np.broadcast_to(hermitian, (3, 16, len(f))) + 0.1j * np.cos(f)
    order = np.random.default_rng(0).permutation(len(f))
    savemat(tmp_path / "girf.mat", {"f": f[None, order], "girf": girf[..., order]})

    freqs, read = read_girf(tmp_path / "girf.mat")
    np.testing.assert_allclose(freqs, np.arange(400) * df * 1e3, atol=1e-6)
    assert read.shape == (3, 16, 400)
    np.testing.assert_allclose(read, np.broadcast_to(hermitian[400:], read.shape))
    assert np.isreal(read[..., 0]).all()

    freqs, read = read_girf(tmp_path / "girf.mat", fmax=10e3)
    assert np.all(read[..., freqs > 10e3] == 0)
    assert np.all(read[..., freqs <= 10e3] != 0)


def test_read_gradients(tmp_path):
    """Gradients are read in full, and integrate to the ADC trajectory."""
    trajectory = initialize_3D_cones(Nc=16, Ns=256, in_out=True).astype(np.float32)
    write_trajectory(
        trajectory=trajectory,
        FOV=(0.23, 0.23, 0.1248),
        img_size=(64, 64, 32),
        grad_filename=str(tmp_path / "traj"),
        version=5.1,
        pregrad="prephase",
        postgrad="slowdown_to_center",
        grad_method="lp",
    )
    filename = tmp_path / "traj.bin"
    gradients, initial_positions, params = read_gradients(filename)
    start, end = params["start_skip_samples"], params["end_skip_samples"]
    assert start > 0 and end > 0
    assert gradients.shape == (16, start + 255 + end, 3)
    np.testing.assert_allclose(initial_positions, 0)

    raster_time = 10e-6
    kspace = initial_positions[:, None] + np.cumsum(
        gradients * params["gamma"] * 1e3 * raster_time, axis=1
    )
    adc_kspace, _ = read_trajectory(
        str(filename), read_shots=True, normalize_factor=None
    )
    np.testing.assert_allclose(
        kspace[:, start - 1 : start + 255], adc_kspace, atol=1e-3 * np.abs(kspace).max()
    )


class _FakeLines:
    def __init__(self, slice_data):
        self._sd = slice_data
        self.image = self

    def __getitem__(self, item):
        return self

    def headers(self):
        return [{"SliceData": self._sd}]


class _FakeTwix:
    def __init__(self, meas_yaps, slice_data):
        self.hdr = type("Hdr", (), {"MeasYaps": meas_yaps})()
        self.lines = _FakeLines(slice_data)


def _yaps(n_img, n_part, thick_vox, sos):
    yaps = {
        ("sSliceArray", "asSlice", 0, "dThickness"): n_img * thick_vox,
        ("sSliceArray", "asSlice", 0, "dReadoutFOV"): 192.0,
        ("sSliceArray", "asSlice", 0, "dPhaseFOV"): 192.0,
        ("sKSpace", "lBaseResolution"): 160,
        ("sKSpace", "lPhaseEncodingLines"): 160,
        ("sKSpace", "ucDimension"): 4,
        ("sKSpace", "lPartitions"): n_part,
    }
    if sos is not None:
        yaps[("sKSpace", "dSliceOversamplingForDialog")] = sos
    return yaps


@pytest.mark.parametrize("sos", [None, 0.0, 1 / 14])
def test_affine_centred_on_slab_with_slice_oversampling(sos):
    """The volume holds all lPartitions, oversampling included.

    Its centre must be the slab centre.
    """
    n_img, thick_vox = 112, 1.2
    n_part = round(n_img * (1 + (sos or 0)))
    pos = {"Sag": 0.0, "Cor": 0.0, "Tra": 10.0}
    sd = {"Quaternion": [0.0, 0.0, 0.0, 1.0], "SlicePos": pos}
    aff = twix2nifti_affine(_FakeTwix(_yaps(n_img, n_part, thick_vox, sos), sd))
    ref = twix2nifti_affine(_FakeTwix(_yaps(n_img, n_img, thick_vox, None), sd))
    np.testing.assert_allclose(abs(aff[2, 2]), thick_vox, rtol=1e-6)
    centre = aff @ np.array([80.0, 80.0, n_part / 2, 1.0])
    centre_ref = ref @ np.array([80.0, 80.0, n_img / 2, 1.0])
    np.testing.assert_allclose(centre[:3], centre_ref[:3], atol=1e-6)
