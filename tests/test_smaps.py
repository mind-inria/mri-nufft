"""Tests for sensitivity maps estimation (mrinufft.extras.smaps)."""

import numpy as np
import pytest

from mrinufft import get_operator
from mrinufft.extras import cartesian_espirit, cartesian_pisco, low_frequency
from mrinufft.extras.cartesian import fft, ifft
from mrinufft._array_compat import CUPY_AVAILABLE
from mrinufft.extras.smaps import (
    _chc_exact,
    _chc_exact_gpu,
    _crop_or_pad,
    _kernel_mask,
)

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


def _calibration_matrix(calib, kernel_width, mask=None):
    """Explicit ESPIRiT calibration matrix: one row per valid kernel position."""
    n_coils, *calib_width = calib.shape
    windows = np.lib.stride_tricks.sliding_window_view(
        calib, kernel_width, axis=tuple(range(1, calib.ndim))
    )
    windows = np.moveaxis(windows, 0, len(calib_width)).reshape(
        -1, n_coils, int(np.prod(kernel_width))
    )
    taps = (
        np.ones(kernel_width, bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    ).ravel()
    return windows[:, :, taps].reshape(-1, n_coils * taps.sum())


def _reference_espirit(kspace, calib_width, kernel_width, thresh):
    """Textbook 2D ESPIRiT (Uecker et al. 2014), with an exact eigendecomposition.

    Explicit calibration matrix from the valid kernel positions, SVD, signal
    subspace above `thresh`, and the top eigenvector of the image-domain
    operator at every pixel (`numpy.linalg.eigh`, no power iteration).
    """
    n_coils, *shape = kspace.shape
    calib = _crop_or_pad(kspace, (n_coils, calib_width, calib_width))
    _, sv, vh = np.linalg.svd(_calibration_matrix(calib, kernel_width), False)
    kernels = vh[sv > thresh * sv.max()].reshape(
        -1, n_coils, kernel_width, kernel_width
    )
    img_k = ifft(_crop_or_pad(kernels, kernels.shape[:2] + tuple(shape)), dims=2)
    eigval, eigvec = np.linalg.eigh(np.einsum("rcxy,rdxy->xycd", img_k, np.conj(img_k)))
    eig = eigval[..., -1] * np.prod(shape) / kernel_width**2
    return np.moveaxis(eigvec[..., -1], -1, 0), eig


def _abs_corr(a, b):
    """Correlation magnitude of two complex arrays (insensitive to a global phase)."""
    a, b = a.ravel(), b.ravel()
    return np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))


@pytest.mark.parametrize("kernel_shape", ["rect", "ellipsoid"])
def test_chc_exact_matches_calibration_matrix(kernel_shape):
    """C^H C must come from the valid rows of the calibration matrix only."""
    rng = np.random.default_rng(0)
    calib = rng.standard_normal((3, 10, 9)) + 1j * rng.standard_normal((3, 10, 9))
    mask = _kernel_mask((4, 4), kernel_shape, np)
    chc, patch_size = _chc_exact(calib.astype(np.complex64), (4, 4), mask)
    C = _calibration_matrix(calib, 4, mask)
    assert patch_size == (16 if mask is None else mask.sum())
    np.testing.assert_allclose(chc, C.conj().T @ C, rtol=1e-4, atol=1e-3)


_array_modules = pytest.mark.parametrize(
    "xp_name",
    [
        "numpy",
        pytest.param(
            "cupy",
            marks=pytest.mark.skipif(not CUPY_AVAILABLE, reason="cupy not available"),
        ),
    ],
)


def _array_module(name):
    if name == "numpy":
        return np
    import cupy

    return cupy


@_array_modules
@pytest.mark.parametrize("kernel_shape", ["rect", "ellipsoid"])
def test_chc_exact_gpu_matches_chc_exact(xp_name, kernel_shape):
    """The coil-blocked C^H C must match the single-product one."""
    xp = _array_module(xp_name)
    rng = np.random.default_rng(0)
    calib = rng.standard_normal((5, 12, 11)) + 1j * rng.standard_normal((5, 12, 11))
    calib = calib.astype(np.complex64)
    mask = _kernel_mask((4, 5), kernel_shape, np)
    expected, patch_size = _chc_exact(calib, (4, 5), mask)
    # Budget of two coils per group: groups of 2, 2 and 1 coils.
    n_rows = (12 - 4 + 1) * (11 - 5 + 1)
    max_bytes = 2 * patch_size * n_rows * calib.itemsize
    chc, size = _chc_exact_gpu(xp.asarray(calib), (4, 5), mask, max_bytes=max_bytes)
    chc = chc.get() if hasattr(chc, "get") else chc
    assert size == patch_size
    np.testing.assert_allclose(chc, expected, rtol=1e-5, atol=1e-3)


@_array_modules
@pytest.mark.parametrize("decim", [1, 2])
@pytest.mark.parametrize(
    "method", [cartesian_espirit, cartesian_pisco], ids=["espirit", "pisco"]
)
def test_cartesian_smaps_keep_single_precision(
    cartesian_kspace_2d, xp_name, decim, method
):
    """complex64 k-space must give complex64 maps, with or without decimation."""
    xp = _array_module(xp_name)
    kspace = xp.asarray(cartesian_kspace_2d.astype(np.complex64))
    est = method(kspace, (N, N), calib_width=16, kernel_size=5, decim=decim)
    assert est.dtype == np.complex64


@pytest.mark.parametrize("snr", [5, 10, 20])
def test_cartesian_espirit_matches_reference_noisy(snr):
    """ESPIRiT should match a textbook implementation on noisy calibration data."""
    shape, n_coils, calib_width, kernel_width = (32, 32), 8, 20, 5
    smaps_gt = _synthetic_smaps(shape, n_coils)
    image = _phantom(shape)
    kspace = fft(image[None] * smaps_gt, dims=2)
    rng = np.random.default_rng(0)
    sigma = np.abs(kspace).max() / snr / 10
    kspace = kspace + sigma * (
        rng.standard_normal(kspace.shape) + 1j * rng.standard_normal(kspace.shape)
    )
    kspace = kspace.astype(np.complex64)

    ref, eig = _reference_espirit(kspace, calib_width, kernel_width, thresh=0.02)
    est = cartesian_espirit(
        kspace,
        shape,
        calib_width=calib_width,
        kernel_size=kernel_width,
        kernel_shape="rect",
        thresh=0.02,
        crop=1.0,  # keep every pixel, the support is chosen below
    )
    mask = (eig / eig.max() > 0.9) & (np.abs(image) > 0)
    min_corr = 0.99 if snr == 5 else 0.999
    assert _abs_corr(
        _phase_reference(est)[:, mask], _phase_reference(ref)[:, mask]
    ) > min_corr
    assert (
        _abs_corr(_phase_reference(est)[:, mask], _phase_reference(smaps_gt)[:, mask])
        > 0.99
    )


def test_cartesian_pisco_matches_reference_noisy():
    """PISCO should use the exact calibration Gram matrix on noisy data."""
    shape, n_coils, calib_width, kernel_width, snr = (32, 32), 8, 20, 5, 5
    smaps_gt = _synthetic_smaps(shape, n_coils)
    image = _phantom(shape)
    kspace = fft(image[None] * smaps_gt, dims=2)
    rng = np.random.default_rng(0)
    sigma = np.abs(kspace).max() / snr / 10
    kspace = kspace + sigma * (
        rng.standard_normal(kspace.shape) + 1j * rng.standard_normal(kspace.shape)
    )
    kspace = kspace.astype(np.complex64)

    ref, eig = _reference_espirit(kspace, calib_width, kernel_width, thresh=0.02)
    est = cartesian_pisco(
        kspace,
        shape,
        calib_width=calib_width,
        kernel_size=kernel_width,
        kernel_shape="rect",
        thresh=0.02,
        crop=1.0,
    )
    mask = (eig / eig.max() > 0.9) & (np.abs(image) > 0)
    assert (
        _abs_corr(_phase_reference(est)[:, mask], _phase_reference(ref)[:, mask])
        > 0.98
    )


def test_cartesian_espirit_decimated_recovers_smaps(cartesian_kspace_2d, phantom_2d):
    """ESPIRiT on a decimated grid, interpolated back, should recover the maps."""
    _, smaps_gt, mask = phantom_2d
    est = cartesian_espirit(
        cartesian_kspace_2d, (N, N), calib_width=16, kernel_size=5, decim=2
    )
    assert (
        _abs_corr(_phase_reference(est)[:, mask], _phase_reference(smaps_gt)[:, mask])
        > 0.99
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
