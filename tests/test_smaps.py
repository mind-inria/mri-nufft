"""Tests for sensitivity maps estimation (mrinufft.extras.smaps)."""

import numpy as np
import pytest

from mrinufft import get_operator
from mrinufft.extras import cartesian_espirit, cartesian_pisco, low_frequency
from mrinufft.extras.cartesian import fft

from case_trajectories import CasesTrajectories
from helpers import assert_correlate

N = 32
N_COILS = 4
# tolerance on the smaps-recovery correlation/slope checks below
SLOPE_ERR = 0.02
R_VALUE_ERR = 0.01


def _synthetic_smaps(shape, n_coils):
    """Build smooth, complex, SOS-normalized synthetic coil sensitivity maps."""
    grid = np.meshgrid(
        *[np.linspace(-1, 1, s, endpoint=False) for s in shape], indexing="ij"
    )
    maps = np.zeros((n_coils,) + shape, dtype=np.complex64)
    for c in range(n_coils):
        theta = 2 * np.pi * c / n_coils
        center = [1.5 * np.cos(theta), 1.5 * np.sin(theta)] + [0.0] * (len(shape) - 2)
        r = sum((g - ctr) for g, ctr in zip(grid, center)) * 1j + sum(
            (g - ctr) ** 2 for g, ctr in zip(grid, center)
        )
        maps[c] = 1 / (r + 1.5)
    sos = np.linalg.norm(maps, axis=0)
    return (maps / sos).astype(np.complex64)


def _phantom(shape):
    """Build a smooth disk-shaped phantom with some internal structure."""
    grid = np.meshgrid(
        *[np.linspace(-1, 1, s, endpoint=False) for s in shape], indexing="ij"
    )
    radius = np.sqrt(sum(g**2 for g in grid))
    img = (radius < 0.8).astype(np.float32)
    img *= 1 + 0.4 * grid[0] + 0.3 * grid[1]
    return img.astype(np.complex64)


def _phase_reference(smaps):
    """Rephase Smaps to coil 0's phase, matching the module's own convention."""
    return smaps * np.conj(smaps[0] / (np.abs(smaps[0]) + 1e-12))


@pytest.fixture(scope="module")
def phantom_2d():
    """Return a (image, ground-truth smaps, support mask) triplet."""
    shape = (N, N)
    smaps = _synthetic_smaps(shape, N_COILS)
    image = _phantom(shape)
    mask = np.abs(image) > 0
    return image, smaps, mask


@pytest.fixture(scope="module")
def cartesian_kspace_2d(phantom_2d):
    """Return the exact Cartesian multi-coil k-space of the phantom."""
    image, smaps, _ = phantom_2d
    coil_images = image[None] * smaps
    return fft(coil_images, dims=2)


@pytest.fixture(scope="module")
def radial_traj():
    """Return a 2D radial trajectory."""
    trajectory, _ = CasesTrajectories().case_radial2D(Nc=32, Ns=64, N=N)
    return trajectory.astype(np.float32).reshape(-1, 2)


@pytest.fixture(scope="module")
def operator(radial_traj):
    """Return a finufft operator for the radial trajectory."""
    return get_operator("finufft")(
        radial_traj, (N, N), n_coils=N_COILS, density=True, squeeze_dims=True
    )


@pytest.fixture(scope="module")
def ksp_data(operator, phantom_2d):
    """Return the simulated multi-coil non-Cartesian k-space of the phantom."""
    image, smaps, _ = phantom_2d
    coil_images = image[None] * smaps
    return operator.op(coil_images)


# -- Cartesian (calibration-based) estimators --


@pytest.mark.parametrize("decim", [1, 2])
def test_cartesian_pisco_recovers_smaps(cartesian_kspace_2d, phantom_2d, decim):
    """PISCO should recover the ground-truth maps up to a global phase."""
    _, smaps_gt, mask = phantom_2d
    est = cartesian_pisco(
        cartesian_kspace_2d, (N, N), calib_width=20, kernel_size=5, decim=decim
    )
    assert_correlate(
        _phase_reference(est)[:, mask],
        _phase_reference(smaps_gt)[:, mask],
        slope_err=SLOPE_ERR,
        r_value_err=R_VALUE_ERR,
    )


def test_cartesian_espirit_recovers_smaps(cartesian_kspace_2d, phantom_2d):
    """ESPIRiT should recover the ground-truth maps up to a global phase."""
    _, smaps_gt, mask = phantom_2d
    est = cartesian_espirit(cartesian_kspace_2d, (N, N), calib_width=20, kernel_size=5)
    assert_correlate(
        _phase_reference(est)[:, mask],
        _phase_reference(smaps_gt)[:, mask],
        slope_err=SLOPE_ERR,
        r_value_err=R_VALUE_ERR,
    )


def test_noncartesian_low_frequency_matches_ground_truth(
    operator, ksp_data, phantom_2d
):
    """low_frequency should approximately recover the ground-truth maps."""
    _, smaps_gt, mask = phantom_2d
    est = low_frequency(
        operator.samples,
        (N, N),
        kspace_data=ksp_data,
        backend="finufft",
        density=operator.density,
    )
    assert_correlate(
        _phase_reference(est)[:, mask],
        _phase_reference(smaps_gt)[:, mask],
        slope_err=SLOPE_ERR,
        r_value_err=R_VALUE_ERR,
    )
