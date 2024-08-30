"""Test off-resonance spatial coefficient and temporal interpolator estimation."""

import math

import numpy as np
import numpy.testing as npt

import pytest

import mrinufft
from mrinufft.operators.base import CUPY_AVAILABLE, AUTOGRAD_AVAILABLE
from mrinufft._utils import get_array_module

if CUPY_AVAILABLE:
    import cupy as cp

if AUTOGRAD_AVAILABLE:
    import torch

# parameter combinations (ndims, backend, device)
params = [(2, "cpu", None), (3, "cpu", None)]

if CUPY_AVAILABLE:
    params.extend([(2, "gpu", None), (3, "gpu", None)])

if AUTOGRAD_AVAILABLE:
    params.extend([(2, "torch", "cpu"), (3, "torch", "cpu")])

if AUTOGRAD_AVAILABLE and CUPY_AVAILABLE:
    params.extend([(2, "torch", "cuda"), (3, "torch", "cuda")])


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


def make_b0map(ndim, npix=64):
    """Make B0 map (units: Hz)."""
    # generate support
    shape = ndim * [npix]
    if ndim == 2:
        mask, fieldmap = _make_disk(shape)
    elif ndim == 3:
        mask, fieldmap = _make_sphere(shape)

    # mask map
    fieldmap *= mask

    # rescale
    fieldmap = 600 * fieldmap / fieldmap.max() - 300  # Hz

    # mask map
    fieldmap *= mask

    return fieldmap.astype(np.float32), mask


def make_t2smap(ndim, npix=64):
    """Make T2* map (units: ms)."""
    shape = ndim * [npix]
    if ndim == 2:
        mask, _ = _make_disk(shape)
    elif ndim == 3:
        mask, _ = _make_sphere(shape)

    # rescale
    fieldmap = 15.0 * mask  # ms

    return fieldmap.astype(np.float32), mask


def _make_disk(shape, frac_radius=0.3):
    """Make circular binary mask."""
    # calculate grid
    ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    yy, xx = yy - ny // 2, xx - nx // 2
    yy, xx = yy / ny, xx / nx

    # radius
    rr = (xx**2 + yy**2) ** 0.5
    return rr < frac_radius, rr


def _make_sphere(shape, frac_radius=0.3):
    """Make spherical binary mask."""
    # calculate grid
    nz, ny, nx = shape
    zz, yy, xx = np.mgrid[:nz, :ny, :nx]
    zz, yy, xx = zz - nz // 2, yy - ny // 2, xx - nx // 2
    zz, yy, xx = zz / nz, yy / ny, xx / nx

    # radius
    rr = (xx**2 + yy**2 + zz**2) ** 0.5
    return rr < frac_radius, rr


def get_backend(backend):
    if backend == "cpu":
        return np
    elif backend == "gpu":
        return cp
    elif backend == "torch":
        return torch


def assert_allclose(actual, expected, atol, rtol):
    """Backend agnostic assertion."""
    xp = get_array_module(actual)
    if xp.__name__ == "numpy":
        npt.assert_allclose(actual, expected, atol, rtol)
    if xp.__name__ == "cupy":
        npt.assert_allclose(actual.get(), expected.get(), atol, rtol)
    if xp.__name__ == "torch":
        npt.assert_allclose(
            actual.numpy(force=True), expected.numpy(force=True), atol, rtol
        )


@pytest.mark.parametrize("ndim, module, device", params)
def test_b0map_coeff(ndim, module, device):
    """Test exponential approximation for B0 field only."""
    # get module
    xp = get_backend(module)

    # generate readout times
    tread = xp.linspace(0.0, 5e-3, 501, dtype=xp.float32)  # (50,), 5 ms

    # generate map
    b0map, mask = make_b0map(ndim)
    b0map, mask = xp.asarray(b0map), xp.asarray(mask)

    # cast to device if torch tensor
    if xp.__name__ == "torch":
        b0map = b0map.to(device)

    # generate coefficients
    B, C = mrinufft.get_interpolators_from_fieldmap(
        b0map, tread, mask=mask, n_time_segments=100
    )

    # correct backend
    xpb, xpc = get_array_module(B), get_array_module(C)
    assert xpb.__name__ == xp.__name__
    assert xpc.__name__ == xp.__name__

    if xp.__name__ == "torch":
        assert B.device == b0map.device
        assert C.device == b0map.device

    # correct dtype
    assert B.dtype == xp.complex64
    assert C.dtype == xp.complex64

    # correct shape
    npt.assert_allclose(B.shape, (100, 501))
    npt.assert_allclose(C.shape, (100, *b0map.shape))

    # correct approximation
    expected = calculate_true_offresonance_term(0 + 2 * math.pi * 1j * b0map, tread)
    actual = calculate_approx_offresonance_term(B, C)
    assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("ndim, module, device", params)
def test_zmap_coeff(ndim, module, device):
    """Test exponential approximation for complex Zmap = R2* + 1j * B0 field."""
    # get module
    xp = get_backend(module)

    # generate readout times
    tread = xp.linspace(0.0, 5e-3, 501, dtype=xp.float32)  # (50,), 5 ms

    # generate B0 map
    b0map, mask = make_b0map(ndim)
    b0map, mask = xp.asarray(b0map), xp.asarray(mask)

    # generate T2* map
    t2smap, mask = make_t2smap(ndim)
    t2smap, mask = xp.asarray(t2smap), xp.asarray(mask)
    t2smap *= 1e-3  # ms -> s
    t2smap[t2smap == 0.0] = 1.0
    r2smap = 1.0 / t2smap  # Hz
    r2smap = mask * r2smap

    # calculate complex fieldmap
    zmap = r2smap + 1j * b0map

    # cast to device if torch tensor
    if xp.__name__ == "torch":
        zmap = zmap.to(device)

    # generate coefficients
    B, C = mrinufft.get_interpolators_from_fieldmap(
        zmap, tread, mask=mask, n_time_segments=100
    )

    # correct backend
    xpb, xpc = get_array_module(B), get_array_module(C)
    assert xpb.__name__ == xp.__name__
    assert xpc.__name__ == xp.__name__

    if xp.__name__ == "torch":
        assert B.device == zmap.device
        assert C.device == zmap.device

    # correct dtype
    assert B.dtype == xp.complex64
    assert C.dtype == xp.complex64

    # correct shape
    npt.assert_allclose(B.shape, (100, 501))
    npt.assert_allclose(C.shape, (100, *zmap.shape))

    # correct approximation
    expected = calculate_true_offresonance_term(2 * math.pi * zmap, tread)
    actual = calculate_approx_offresonance_term(B, C)
    assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
