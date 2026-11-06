"""Tests for the GIRF model of the gradient chain."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mrinufft.extras import compute_girf_interpolators
from mrinufft.trajectories import (
    Acquisition,
    Hardware,
    apply_girf,
    convert_gradients_to_trajectory,
    convert_trajectory_to_gradients,
    normalize_trajectory,
    spherical_harmonics,
)
from mrinufft.trajectories.girf import _reexpansion

DT = 10e-6
DF = 100.0
FREQS = np.arange(0, 0.5 / DT + DF, DF)
ACQ = Acquisition(
    fov=(0.2, 0.2, 0.1),
    img_size=(128, 128, 48),
    hardware=Hardware(grad_raster_time=DT),
)


def constant_girf(b0=(0, 0, 0), gains=(1, 1, 1), cross=None, n_terms=4):
    """GIRF constant over frequency, ``cross`` maps (input axis, term) to values."""
    girf = np.zeros((3, n_terms, len(FREQS)), dtype=complex)
    girf[:, 0] = np.asarray(b0)[:, None]
    for i, gain in enumerate(gains):
        girf[i, i + 1] = gain
    for (i, j), value in (cross or {}).items():
        girf[i, j] = value
    return girf


def oblique_acq(seed, dim=3, offset=(0.02, -0.01, 0.04)):
    """Acquisition of an oblique, off-centre image."""
    rotation = Rotation.random(random_state=seed).as_matrix()
    res = np.array([2e-3, 2e-3, 3e-3])[:dim]
    size = np.array([64, 64, 32])[:dim]
    scale = np.ones(3)
    scale[:dim] = res
    centre = np.zeros(3)
    centre[:dim] = size // 2
    affine = np.eye(4)
    affine[:3, :3] = rotation * scale
    affine[:3, 3] = np.asarray(offset) - affine[:3, :3] @ centre
    acq = Acquisition(
        fov=tuple(res * size),
        img_size=tuple(int(n) for n in size),
        hardware=Hardware(grad_raster_time=DT),
        affine=affine,
    )
    return acq, rotation, np.asarray(offset)


@pytest.fixture
def gradients():
    """Smooth waveforms starting and ending at 0, in T/m."""
    rng = np.random.default_rng(0)
    t = np.arange(400) * DT
    window = np.sin(np.pi * t / t[-1])[None, :, None] ** 2
    freqs = rng.uniform(200, 3000, size=(5, 1, 3))
    phases = rng.uniform(0, 2 * np.pi, size=(5, 1, 3))
    return 20e-3 * window * np.sin(2 * np.pi * freqs * t[None, :, None] + phases)


def test_spherical_harmonics():
    """The terms are harmonic, ordered as the Skope ones, and z is 0 in 2D."""
    rng = np.random.default_rng(0)
    r = rng.uniform(-0.1, 0.1, size=(20, 3))
    x, y, z = r.T
    values = spherical_harmonics(r)
    np.testing.assert_allclose(values[:, 4], x * y)
    np.testing.assert_allclose(values[:, 6], 2 * z**2 - x**2 - y**2)
    np.testing.assert_allclose(values[:, 10], x * y * z)
    # Central differences are exact for cubic polynomials.
    h = 1e-2
    laplacian = sum(
        spherical_harmonics(r + h * e) - 2 * values + spherical_harmonics(r - h * e)
        for e in np.eye(3)
    )
    np.testing.assert_allclose(laplacian, 0, atol=1e-12)
    np.testing.assert_allclose(
        spherical_harmonics(r[:, :2], 9), spherical_harmonics(r * [1, 1, 0], 9)
    )
    with pytest.raises(ValueError, match="between 1 and 16"):
        spherical_harmonics(r, 17)


@pytest.mark.parametrize("n_terms", [4, 9, 16])
def test_reexpansion(n_terms):
    """The closed form matches a regression, reflections included."""
    rng = np.random.default_rng(n_terms)
    rotation = Rotation.random(random_state=n_terms).as_matrix() @ np.diag([1, 1, -1])
    origin = rng.uniform(-0.1, 0.1, 3)
    u = rng.uniform(-0.2, 0.2, size=(4 * n_terms, 3))
    regression, *_ = np.linalg.lstsq(
        spherical_harmonics(u, n_terms),
        spherical_harmonics(u @ rotation.T + origin, n_terms),
        rcond=None,
    )
    np.testing.assert_allclose(
        _reexpansion(rotation, origin, n_terms), regression, atol=1e-12
    )


def test_identity(gradients):
    """A unit GIRF plays the nominal gradients, whatever the geometry."""
    acq, _, _ = oblique_acq(0)
    played, fields = apply_girf(
        girf=constant_girf(n_terms=16), freqs=FREQS, gradients=gradients, acq=acq
    )
    np.testing.assert_allclose(played, gradients, atol=1e-12)
    np.testing.assert_allclose(fields, 0, atol=1e-12)


def test_constant_terms(gradients):
    """Frequency-independent terms, of any order, act as a matrix on gradients."""
    rng = np.random.default_rng(4)
    girf = constant_girf(
        b0=(1e-4, 0, -2e-4), gains=(1.01, 0.99, 1), cross={(0, 2): 0.01}, n_terms=9
    )
    girf[:, 4:] = rng.normal(size=(3, 5, 1)) * 1e-2
    played, fields = apply_girf(girf=girf, freqs=FREQS, gradients=gradients, acq=ACQ)
    deviation = girf[..., 0].real.copy()
    deviation[:, 1:4] -= np.eye(3)
    assert fields.shape == (*gradients.shape[:2], 9)
    np.testing.assert_allclose(played, gradients @ girf[:, 1:4, 0].real, atol=1e-12)
    np.testing.assert_allclose(fields, gradients @ deviation, atol=1e-12)


def test_delay(gradients):
    """A delay in the GIRF or as a parameter shifts the gradients alike."""
    shift = 3
    played, _ = apply_girf(
        girf=constant_girf(),
        freqs=FREQS,
        gradients=gradients,
        acq=ACQ,
        delay=shift * DT,
    )
    np.testing.assert_allclose(played[:, shift:], gradients[:, :-shift], atol=1e-12)
    np.testing.assert_allclose(played[:, :shift], 0, atol=1e-12)

    girf = constant_girf() * np.exp(-2j * np.pi * FREQS * shift * DT)
    from_girf, _ = apply_girf(girf=girf, freqs=FREQS, gradients=gradients, acq=ACQ)
    np.testing.assert_allclose(from_girf, played, atol=1e-4 * np.abs(gradients).max())


def test_first_order_low_pass():
    """A first-order low-pass is causal and delays k-space by its time constant."""
    tau = 50e-6
    # A smooth band limit, as for measured GIRFs, keeps the ringing short.
    taper = 0.5 * (1 + np.cos(np.pi * np.clip((FREQS - 20e3) / 20e3, 0, 1)))
    girf = constant_girf() * taper / (1 + 2j * np.pi * FREQS * tau)
    step = np.zeros((1, 2000, 3))
    step[:, 500:] = 10e-3
    played, _ = apply_girf(girf=girf, freqs=FREQS, gradients=step, acq=ACQ)

    # Only the Gibbs ringing of the band limit precedes the step.
    assert np.abs(played[:, :497]).max() < 1e-2 * step.max()
    np.testing.assert_allclose(played[:, 505], (1 - np.exp(-1)) * step.max(), rtol=0.1)
    lag = np.sum(step - played, axis=1) * DT
    np.testing.assert_allclose(lag, 10e-3 * tau, rtol=5e-3)


@pytest.mark.parametrize("t_start", [0.0, -1e-3, -4e-3])
def test_causality(t_start):
    """An excess of DC gain spreads over [t_start, t_start + 1/df) only."""
    girf = constant_girf()
    girf[..., 0] *= 1.01
    onset, height = 1000, 10e-3
    step = np.zeros((1, 3000, 3))
    step[:, onset:] = height
    played, _ = apply_girf(
        girf=girf, freqs=FREQS, gradients=step, acq=ACQ, t_start=t_start
    )

    # The end of the input is anticipated as well, by -t_start.
    n_h, n_pre = round(1 / (DF * DT)), round(-t_start / DT)
    n_valid = step.shape[1] - n_pre
    ramp = np.clip(np.arange(n_valid) - onset + n_pre + 1, 0, n_h) / n_h
    np.testing.assert_allclose(
        played[:, :n_valid],
        step[:, :n_valid] + 0.01 * height * ramp[None, :, None],
        atol=1e-15,
    )


def test_slice_offset(gradients):
    """Off the isocentre, a second-order term also deviates the first order."""
    girf = constant_girf(cross={(0, 7): 0.05}, n_terms=9)  # x -> xz
    z0 = 0.03
    acq, _, _ = oblique_acq(0, offset=(0, 0, z0))
    affine = np.diag([*acq.res, 1.0])
    affine[:3, 3] = (0, 0, z0) - np.asarray(acq.res) * (np.asarray(acq.img_size) // 2)
    acq = Acquisition(
        fov=acq.fov, img_size=acq.img_size, hardware=acq.hardware, affine=affine
    )
    g_x = gradients * [1, 0, 0]
    played, fields = apply_girf(girf=girf, freqs=FREQS, gradients=g_x, acq=acq)
    np.testing.assert_allclose(fields[..., 7], 0.05 * g_x[..., 0], atol=1e-15)
    np.testing.assert_allclose(fields[..., 1], 0.05 * z0 * g_x[..., 0], atol=1e-15)
    np.testing.assert_allclose(np.delete(fields, [1, 7], axis=-1), 0, atol=1e-15)
    np.testing.assert_allclose(
        played[..., 0], (1 + 0.05 * z0) * g_x[..., 0], atol=1e-15
    )


def test_frames(gradients):
    """Deviations expanded in the image frame match those of the GIRF frame."""
    rng = np.random.default_rng(1)
    girf = constant_girf(b0=rng.normal(size=3) * 1e-4, n_terms=16)
    girf += 0.05 * rng.normal(size=(3, 16, 1))
    acq, rotation, offset = oblique_acq(2)
    girf_affine = np.diag([-1.0, 1.0, -1.0, 1.0])
    girf_affine[:3, 3] = (0.001, 0.002, -0.003)
    _, fields = apply_girf(
        girf=girf, freqs=FREQS, gradients=gradients, acq=acq, girf_affine=girf_affine
    )

    to_girf = girf_affine[:3, :3] @ rotation
    nominal = gradients @ to_girf.T
    expected = nominal @ girf[..., 0].real
    expected[..., 1:4] -= nominal
    u = rng.uniform(-0.05, 0.05, size=(30, 3))
    r = u @ to_girf.T + girf_affine[:3, :3] @ offset + girf_affine[:3, 3]
    np.testing.assert_allclose(
        fields @ spherical_harmonics(u).T,
        expected @ spherical_harmonics(r).T,
        atol=1e-12,
    )


def test_trajectory(gradients):
    """Trajectories are wrapped, initial positions kept, fields are consistent."""
    rng = np.random.default_rng(3)
    initial_positions = rng.uniform(-100, 100, size=(len(gradients), 3))
    trajectory = convert_gradients_to_trajectory(gradients, initial_positions, ACQ)
    girf = constant_girf(
        b0=(1e-4, 0, 0), gains=(1.02, 0.98, 0.95), cross={(0, 4): 3e-2}, n_terms=9
    )
    # Scaled to wrap around the edges of k-space.
    scaled = trajectory * 1.9
    played, fields = apply_girf(girf=girf, freqs=FREQS, trajectory=scaled, acq=ACQ)
    assert played.shape == trajectory.shape
    assert fields.shape == (len(trajectory), trajectory.shape[1] - 1, 9)
    assert np.all(np.abs(played) <= ACQ.norm_factor)
    np.testing.assert_allclose(played[:, 0], scaled[:, 0], atol=1e-12)

    nominal, _ = convert_trajectory_to_gradients(scaled, ACQ)
    np.testing.assert_allclose(fields[..., 0], 1e-4 * nominal[..., 0], atol=1e-15)
    np.testing.assert_allclose(fields[..., 4], 3e-2 * nominal[..., 0], atol=1e-15)
    kspace = ACQ.gamma * DT * np.cumsum(fields[..., 1:4], axis=1)
    unwrapped = scaled[:, 1:] + normalize_trajectory(kspace, ACQ)
    np.testing.assert_allclose(
        np.exp(2j * np.pi * played[:, 1:]), np.exp(2j * np.pi * unwrapped), atol=1e-9
    )


def test_2D(gradients):
    """2D gradients are handled as 3D ones with no component along the normal."""
    acq, rotation, offset = oblique_acq(3, dim=2)
    girf = constant_girf(gains=(1.01, 0.99, 1), n_terms=4)
    played, fields = apply_girf(
        girf=girf, freqs=FREQS, gradients=gradients[..., :2], acq=acq
    )
    assert played.shape == gradients[..., :2].shape
    deviation = np.diag([0.01, -0.01, 0])
    gains = np.eye(3) + rotation.T @ deviation @ rotation
    np.testing.assert_allclose(played, gradients[..., :2] @ gains[:2, :2], atol=1e-12)
    np.testing.assert_allclose(
        fields[..., 0],
        gradients[..., :2] @ (rotation.T @ deviation)[:2] @ offset,
        atol=1e-15,
    )


def test_errors(gradients):
    """Inputs, GIRF shapes, geometry and time window are checked."""
    girf = constant_girf()
    with pytest.raises(ValueError, match="exactly one"):
        apply_girf(girf=girf, freqs=FREQS)
    with pytest.raises(ValueError, match="exactly one"):
        apply_girf(girf=girf, freqs=FREQS, gradients=gradients, trajectory=gradients)
    with pytest.raises(ValueError, match="start at 0"):
        apply_girf(girf=girf, freqs=FREQS + DF, gradients=gradients)
    with pytest.raises(ValueError, match="n_terms in"):
        apply_girf(girf=constant_girf(n_terms=5), freqs=FREQS, gradients=gradients)
    with pytest.raises(ValueError, match="numbers of frequencies"):
        apply_girf(girf=girf, freqs=FREQS[:-1], gradients=gradients)
    for t_start in (1e-5, -1 / DF):
        with pytest.raises(ValueError, match="t_start"):
            apply_girf(girf=girf, freqs=FREQS, gradients=gradients, t_start=t_start)
    acq = Acquisition(
        fov=ACQ.fov, img_size=ACQ.img_size, hardware=ACQ.hardware, affine=np.eye(4)
    )
    with pytest.raises(ValueError, match="rigid"):
        apply_girf(girf=girf, freqs=FREQS, gradients=gradients, acq=acq)


@pytest.fixture
def higher_order():
    """Second-order fields, and the exact exponential at the 150 samples."""
    rng = np.random.default_rng(5)
    acq = Acquisition(
        fov=(0.2, 0.2), img_size=(40, 40), hardware=Hardware(grad_raster_time=DT)
    )
    fields = np.zeros((3, 149, 9))
    fields[..., 4:] = rng.normal(size=(3, 149, 5)) / (2 * np.pi * acq.gamma * DT)
    phase = np.zeros((3, 150, 9))
    phase[:, 1:] = 2 * np.pi * acq.gamma * DT * np.cumsum(fields, axis=1)
    u = (np.indices((40, 40)).reshape(2, -1).T - 20) * 0.2 / 40
    exact = np.exp(-1j * phase.reshape(-1, 9) @ spherical_harmonics(u, 9).T)
    return fields, acq, exact


@pytest.mark.parametrize("order", [-1, 4])
def test_girf_interpolators(higher_order, order):
    """B @ C reaches the tolerance with an automatic rank."""
    fields, acq, exact = higher_order
    B, C = compute_girf_interpolators(fields, acq, order=order, tol=1e-4)
    assert B.shape[0] == len(exact)
    assert C.shape == (B.shape[1], *acq.img_size)
    error = np.linalg.norm(exact - B @ C.reshape(len(C), -1)) / np.linalg.norm(exact)
    assert error < 2e-4


def test_girf_interpolators_skip(higher_order):
    """Phases accrue from the start of the fields, B starts after ``skip``."""
    fields, acq, exact = higher_order
    skip = 30
    B, C = compute_girf_interpolators(fields, acq, tol=1e-4, skip=skip)
    expected = exact.reshape(len(fields), -1, exact.shape[1])[:, skip:]
    expected = expected.reshape(-1, exact.shape[1])
    error = np.linalg.norm(expected - B @ C.reshape(len(C), -1))
    assert error < 2e-4 * np.linalg.norm(expected)
    with pytest.raises(ValueError, match="skip"):
        compute_girf_interpolators(fields, acq, skip=fields.shape[1] + 1)
    with pytest.raises(ValueError, match="row per sample"):
        compute_girf_interpolators(fields, acq, skip=skip, interpolators=(B[:5], C))


def test_girf_interpolators_product(higher_order):
    """A prior factorization, e.g. of the static off-resonance, is multiplied."""
    fields, acq, exact = higher_order
    rng = np.random.default_rng(6)
    b0_map = rng.uniform(-50, 50, size=acq.img_size)
    time = np.arange(fields.shape[1] + 1) * DT
    prior = np.exp(2j * np.pi * np.outer(time, b0_map.ravel()))
    u, s, vh = np.linalg.svd(prior, full_matrices=False)
    B, C = compute_girf_interpolators(
        fields, acq, tol=1e-4, interpolators=(u * s, vh.reshape(-1, *acq.img_size))
    )
    expected = exact * np.tile(prior, (len(fields), 1))
    error = np.linalg.norm(expected - B @ C.reshape(len(C), -1))
    assert error < 2e-4 * np.linalg.norm(expected)
