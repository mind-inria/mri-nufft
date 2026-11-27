"""Tests for Pulseq trajectory rotations."""

import logging

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mrinufft.io import detect_trajectory_rotations, pulseq_gre
from mrinufft.trajectories import Acquisition
from mrinufft.trajectories.utils import normalize_trajectory


@pytest.mark.parametrize("ndim", [2, 3])
def test_detect_trajectory_rotations(ndim):
    rng = np.random.default_rng(42)
    base = rng.normal(size=(25, ndim))
    if ndim == 2:
        rotations = Rotation.from_rotvec(
            [[0, 0, 0], [0, 0, 0.6], [0, 0, -2]]
        ).as_matrix()
        padded = np.pad(base, ((0, 0), (0, 1)))
        shots = np.array([padded @ r.T for r in rotations])[:, :, :2]
    else:
        rotations = Rotation.from_rotvec([[0, 0, 0], [0.2, -0.5, 0.7]]).as_matrix()
        shots = np.array([base @ r.T for r in rotations])

    reference, detected = detect_trajectory_rotations(shots)
    assert detected is not None
    expected = np.pad(shots, ((0, 0), (0, 0), (0, 3 - ndim)))
    np.testing.assert_allclose(reference, expected[0])
    np.testing.assert_allclose(
        np.array([reference @ r.T for r in detected]), expected, atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.det(detected), 1, atol=1e-12)


def test_detect_rotations_degenerate_and_nonrotational():
    base = np.column_stack((np.linspace(-1, 1, 20), np.zeros((20, 2))))
    shots = np.array([base, -base, np.zeros_like(base)])
    _, rotations = detect_trajectory_rotations(shots[:2])
    np.testing.assert_allclose(base @ rotations[1].T, shots[1], atol=1e-12)
    assert detect_trajectory_rotations(shots)[1] is None
    shots[1] = -base
    shots[1, 4, 1] = 0.01
    assert detect_trajectory_rotations(shots[:2])[1] is None
    assert detect_trajectory_rotations(np.zeros((2, 10, 3)))[1] is not None
    with pytest.raises(ValueError):
        detect_trajectory_rotations(np.full((1, 3, 3), np.nan))


def test_pulseq_rotations_and_fallback(tmp_path, monkeypatch, caplog):
    pp = pytest.importorskip("pypulseq")
    if not hasattr(pp, "rotate_3d"):
        pytest.skip("PyPulseq does not provide rotate_3d")
    acq = Acquisition(
        fov=(0.2, 0.24, 0.3),
        img_size=(64, 64, 64),
    )
    t = np.linspace(0, 1, 64)
    base = np.column_stack((0.2 * t, 0.05 * t**2, 0.03 * t**3))
    matrices = Rotation.from_rotvec([[0, 0, 0], [0.3, -0.4, 0.5]]).as_matrix()
    physical = np.array([base @ r.T for r in matrices])
    trajectory = normalize_trajectory(physical, acq)
    calls = []
    rotate = pp.rotate_3d

    def tracked_rotate(*args, **kwargs):
        calls.append(kwargs["rotation"])
        return rotate(*args, **kwargs)

    monkeypatch.setattr(pp, "rotate_3d", tracked_rotate)
    with caplog.at_level(logging.INFO, logger="mrinufft.io.pulseq"):
        seq = pulseq_gre(trajectory, TR=100, TE=50, FA=10, acq=acq, grad_spoil_factor=0)
    assert len(calls) == 2
    assert "Using PyPulseq rotations" in caplog.text
    monkeypatch.delattr(pp, "rotate_3d")
    fallback = pulseq_gre(
        trajectory, TR=100, TE=50, FA=10, acq=acq, grad_spoil_factor=0
    )

    np.testing.assert_allclose(
        seq.calculate_kspace()[0], fallback.calculate_kspace()[0], atol=1e-9, rtol=1e-6
    )
    path = tmp_path / "rotated.seq"
    seq.write(path)
    restored = pp.Sequence()
    restored.read(path)
    np.testing.assert_allclose(
        restored.calculate_kspace()[0],
        seq.calculate_kspace()[0],
        atol=1e-7,
        rtol=1e-5,
    )


def test_nonrotational_pulseq_uses_individual_gradients(monkeypatch):
    pp = pytest.importorskip("pypulseq")
    if not hasattr(pp, "rotate_3d"):
        pytest.skip("PyPulseq does not provide rotate_3d")
    trajectory = np.zeros((2, 32, 3))
    trajectory[0, :, 0] = np.linspace(0, 0.001, 32)
    trajectory[1, :, 0] = np.linspace(0, 0.002, 32)

    def unexpected_rotation(*args, **kwargs):
        raise AssertionError("non-rotational shots must not be rotated")

    monkeypatch.setattr(pp, "rotate_3d", unexpected_rotation)
    seq = pulseq_gre(trajectory, TR=100, TE=50, FA=10, grad_spoil_factor=0)
    assert len(seq.block_events) == 8
