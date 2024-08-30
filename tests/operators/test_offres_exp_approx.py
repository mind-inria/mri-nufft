"""Test off-resonance spatial coefficient and temporal interpolator estimation."""

import math

import numpy as np
import numpy.testing as npt

import pytest

import mrinufft
from mrinufft.operators.base import CUPY_AVAILABLE, AUTOGRAD_AVAILABLE
from mrinufft._utils import get_array_module

from helpers import to_interface, from_interface


def calculate_true_offresonance_term(fieldmap, t):
    """Calculate non-approximate off-resonance modulation term."""
    xp = get_array_module(fieldmap)
    t = xp.asarray(t)
    if xp.__name__ == "torch":
        t = t.to(device=fieldmap.device)
    arg = t * fieldmap[..., None]
    arg = arg[None, ...].swapaxes(0, -1)[..., 0]
    return xp.exp(-arg)


def calculate_approx_offresonance_term(B, C):
    """Calculate approximate off-resonance modulation term."""
    field_term = 0.0
    for n in range(B.shape[0]):
        tmp = B[n] * C[n][..., None]
        tmp = tmp[None, ...].swapaxes(0, -1)[..., 0]
        field_term += tmp
    return field_term


def _make_b0map(ndim, npix=64):
    """Make B0 map (units: Hz)."""
    shape = ndim * [npix]
    if ndim == 2:
        mask, fieldmap = _make_disk(shape)
    elif ndim == 3:
        mask, fieldmap = _make_sphere(shape)
    fieldmap *= mask
    fieldmap = 600 * fieldmap / fieldmap.max() - 300  # Hz
    fieldmap *= mask
    return fieldmap.astype(np.float32), mask


def _make_t2smap(ndim, npix=64):
    """Make T2* map (units: ms)."""
    shape = ndim * [npix]
    if ndim == 2:
        mask, _ = _make_disk(shape)
    elif ndim == 3:
        mask, _ = _make_sphere(shape)
    fieldmap = 15.0 * mask  # ms
    return fieldmap.astype(np.float32), mask


def _make_disk(shape, frac_radius=0.3):
    """Make circular binary mask."""
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    yy, xx = yy - ny // 2, xx - nx // 2
    yy, xx = yy / ny, xx / nx
    rr = (xx**2 + yy**2) ** 0.5
    return rr < frac_radius, rr


def _make_sphere(shape, frac_radius=0.3):
    """Make spherical binary mask."""
    nz, ny, nx = shape
    zz, yy, xx = np.mgrid[:nz, :ny, :nx]
    zz, yy, xx = zz - nz // 2, yy - ny // 2, xx - nx // 2
    zz, yy, xx = zz / nz, yy / ny, xx / nx
    rr = (xx**2 + yy**2 + zz**2) ** 0.5
    return rr < frac_radius, rr


# Parameter combinations (ndims, backend, device)
params = [(2, "cpu", None), (3, "cpu", None)]

if CUPY_AVAILABLE:
    params.extend([(2, "gpu", None), (3, "gpu", None)])

if AUTOGRAD_AVAILABLE:
    params.extend([(2, "torch", "cpu"), (3, "torch", "cpu")])

if AUTOGRAD_AVAILABLE and CUPY_AVAILABLE:
    params.extend([(2, "torch", "gpu"), (3, "torch", "gpu")])


@pytest.fixture(scope="module", params=params)
def map_fixture(request):
    """Fixture to generate B0 and T2* maps based on dimension and backend."""
    ndim, module, device = request.param
    interface = module if "torch" not in module else f"{module}-{device}"

    # Generate maps
    b0map, mask = _make_b0map(ndim)
    t2smap, _ = _make_t2smap(ndim)

    # Convert maps to the appropriate interface
    b0map = to_interface(b0map, interface)
    mask = to_interface(mask, interface)
    t2smap = to_interface(t2smap, interface)

    return ndim, b0map, t2smap, mask, interface


def assert_allclose(actual, expected, atol, rtol, interface):
    """Backend agnostic assertion using from_interface helper."""
    actual_np = from_interface(actual, interface)
    expected_np = from_interface(expected, interface)
    npt.assert_allclose(actual_np, expected_np, atol=atol, rtol=rtol)


def test_b0map_coeff(map_fixture):
    """Test exponential approximation for B0 field only."""
    ndim, b0map, _, mask, interface = map_fixture

    # Generate readout times
    tread = to_interface(np.linspace(0.0, 5e-3, 501, dtype=np.float32), interface)

    # Generate coefficients
    B, C = mrinufft.get_interpolators_from_fieldmap(
        b0map, tread, mask=mask, n_time_segments=100
    )

    # Assert properties
    assert B.shape == (100, 501)
    assert C.shape == (100, *b0map.shape)

    # Correct approximation
    expected = calculate_true_offresonance_term(0 + 2 * math.pi * 1j * b0map, tread)
    actual = calculate_approx_offresonance_term(B, C)
    assert_allclose(actual, expected, atol=1e-3, rtol=1e-3, interface=interface)


def test_zmap_coeff(map_fixture):
    """Test exponential approximation for complex Zmap = R2* + 1j * B0 field."""
    ndim, b0map, t2smap, mask, interface = map_fixture

    # Generate readout times
    tread = to_interface(np.linspace(0.0, 5e-3, 501, dtype=np.float32), interface)

    # Convert T2* map to R2* map
    t2smap = t2smap * 1e-3  # ms -> s
    r2smap = 1.0 / (t2smap + 1e-9)  # Hz
    r2smap = mask * r2smap

    # Calculate complex fieldmap (Zmap)
    zmap = r2smap + 1j * b0map

    # Generate coefficients
    B, C = mrinufft.get_interpolators_from_fieldmap(
        zmap, tread, mask=mask, n_time_segments=100
    )

    # Assert properties
    assert B.shape == (100, 501)
    assert C.shape == (100, *zmap.shape)

    # Correct approximation
    expected = calculate_true_offresonance_term(2 * math.pi * zmap, tread)
    actual = calculate_approx_offresonance_term(B, C)
    assert_allclose(actual, expected, atol=1e-3, rtol=1e-3, interface=interface)
