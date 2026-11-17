"""Test off-resonance spatial coefficient and temporal interpolator estimation."""

import logging

import numpy as np
import numpy.testing as npt
import pytest
from helpers import to_interface
from helpers.factories import _param_array_interface_np_cp, from_interface
from pytest_cases import parametrize, parametrize_with_cases

from mrinufft import get_operator
from mrinufft.extras import (
    get_complex_fieldmap_rad,
    get_orc_factorization,
    make_b0map,
    make_t2smap,
)
from mrinufft.operators.off_resonance import MRIFourierCorrected


class CasesB0maps:
    """B0 field maps cases we want to test.

    Each case return a field map and the binary spatial support of the object.
    """

    def case_real2D(self, N=64, b0_range=(-300, 300)):
        """Create a real (B0 only) 2D field map."""
        b0_map, mask = make_b0map(2 * [N])
        return b0_map, None, mask

    def case_real3D(self, N=32, b0range=(-300, 300)):
        """Create a real (B0 only) 3D field map."""
        b0_map, mask = make_b0map(3 * [N], b0range)
        return b0_map, None, mask

    def case_complex2D(self, N=64, b0range=(-300, 300), t2svalue=15.0):
        """Create a complex (R2* + 1j * B0) 2D field map."""
        # Generate real and imaginary parts
        t2s_map, _ = make_t2smap(2 * [N], t2svalue)
        b0_map, mask = make_b0map(2 * [N], b0range)

        # Convert T2* map to R2* map
        t2s_map = t2s_map * 1e-3  # ms -> s
        r2s_map = 1.0 / (t2s_map + 1e-9)  # Hz
        r2s_map = mask * r2s_map

        return b0_map, r2s_map, mask

    def case_complex3D(self, N=32, b0range=(-300, 300), t2svalue=15.0):
        """Create a complex (R2* + 1j * B0) 3D field map."""
        # Generate real and imaginary parts
        t2s_map, _ = make_t2smap(3 * [N], t2svalue)
        b0_map, mask = make_b0map(3 * [N], b0range)

        # Convert T2* map to R2* map
        t2s_map = t2s_map * 1e-3  # ms -> s
        r2s_map = 1.0 / (t2s_map + 1e-9)  # Hz
        r2s_map = mask * r2s_map
        return b0_map, r2s_map, mask


@_param_array_interface_np_cp
@parametrize_with_cases(
    "b0_map, r2s_map, mask", cases=[CasesB0maps.case_real2D, CasesB0maps.case_complex2D]
)
@parametrize("method", ["svd-full", "mti", "mfi"])
@parametrize("L, lazy", [(40, True), (-1, True), (40, False)])
def test_b0map_coeff(b0_map, r2s_map, mask, method, L, lazy, array_interface):
    """Test exponential approximation for B0 field only."""
    # Generate readout times
    Nt = 400
    tread = np.linspace(0.0, 5e-3, Nt, dtype=np.float32)

    cpx_fieldmap = get_complex_fieldmap_rad(b0_map, r2s_map).astype(np.complex64)

    E_full = np.exp(np.outer(tread, cpx_fieldmap[mask]))

    kwargs = {}
    if method == "svd-full":  # Truncated SVD is flacky (esp. for cupy)
        kwargs["partial_svd"] = False
        method = "svd"
    B, C, _ = get_orc_factorization(method)(
        to_interface(cpx_fieldmap, array_interface),
        to_interface(tread, array_interface),
        to_interface(mask, array_interface),
        L=L,
        lazy=lazy,
        n_bins=4096,
        **kwargs,
    )

    if L == -1:
        L = B.shape[1]
        print(L)
    # Assert properties
    assert B.shape == (Nt, L)
    assert C.shape == (L, *b0_map.shape)

    if lazy:
        C = np.stack([C[l] for l in range(len(C))])
    # Check that the approximation match the full matrix.
    B = from_interface(B, array_interface)
    C = from_interface(C, array_interface)
    E2 = B @ C[:, mask]
    # TODO get closer bound somehow ?
    npt.assert_allclose(E2, E_full, atol=5e-3, rtol=5e-3)


def test_b0_map_upsampling_warns_and_matches_shape(caplog):
    """Test that MRIFourierCorrected upscales the b0_map and warns if shape mismatch exists."""
    shape_target = (16, 16, 16)
    b0_shape = (8, 8, 8)

    b0_map = np.ones(b0_shape, dtype=np.float32)
    kspace = np.zeros((10, 3), dtype=np.float32)
    smaps = np.ones((1, *shape_target), dtype=np.complex64)
    readout_time = np.ones(10, dtype=np.float32)

    nufft = get_operator("finufft")(
        samples=kspace,
        shape=shape_target,
        n_coils=1,
        smaps=smaps,
        density=False,
    )

    with caplog.at_level(logging.WARNING):
        op = MRIFourierCorrected(
            nufft,
            b0_map=b0_map,
            readout_time=readout_time,
        )

    assert (
        "field_map and mask will be interpolated to match image shape." in caplog.text
    )
    # check that no exception is raised and internal shape matches
    assert op.B.shape[0] == len(readout_time)
    assert op.shape == shape_target


def _brain_like_complex_field(shape=(24, 24, 24)):
    """B0 over ~1400 Hz and R2* in 10-200 /s, in rad/s (Delta = -R2* + 2j pi df)."""
    g = np.indices(shape).astype(np.float32)
    b0 = 700 * np.sin(g[0] / 4) * np.cos(g[1] / 6)
    r2s = 10 + 190 * (g[2] / (shape[2] - 1))
    return get_complex_fieldmap_rad(b0.astype(np.float32), r2s.astype(np.float32))


def test_resolve_n_bins():
    """-1 sets the bin width from the readout; an integer is the B0 bin count."""
    from mrinufft.extras.field_map import _BIN_TOL, _resolve_n_bins

    t_max = 14e-3
    # B0 only: |z - z_k| <= width / 2
    assert _resolve_n_bins(-1, (0.0, 8796.0), t_max) == (
        1,
        int(np.ceil(8796.0 * t_max / (2 * _BIN_TOL))),
    )
    assert _resolve_n_bins(-1, (0.0, 0.0), t_max) == (1, 1)
    assert _resolve_n_bins(1024, (190.0, 8796.0), t_max) == (23, 1024)
    assert _resolve_n_bins(1024, (190.0, 0.0), t_max) == (1024, 1)
    assert _resolve_n_bins((4, 8), (0.0, 1.0), t_max) == (1, 8)


def test_histogram_keeps_occupied_bins():
    """Only occupied bins are kept, and each voxel is within the error bound."""
    from mrinufft.extras.field_map import _BIN_TOL, _create_histogram

    field = _brain_like_complex_field()
    mask = np.zeros(field.shape, dtype=bool)
    mask[4:-4, 4:-4, 4:-4] = True
    t = np.linspace(-14e-3, 14e-3, 300, dtype=np.float32)
    counts, centers, inverse, max_err = _create_histogram(field, mask, t)

    assert centers.size <= field.size
    assert counts.sum() == mask.sum()
    assert max_err <= _BIN_TOL + 1e-6
    dz = np.abs(field[mask] - centers[inverse.reshape(field.shape)[mask]])
    assert np.all(dz * np.abs(t).max() <= max_err * (1 + 1e-3))


def test_auto_bins_memory_cap(monkeypatch, caplog):
    """Past _MAX_E_SIZE, n_bins=-1 uses coarser bins and warns."""
    import mrinufft.extras.field_map as fm

    max_bins = 500
    t = np.linspace(-14e-3, 14e-3, 300, dtype=np.float32)
    monkeypatch.setattr(fm, "_MAX_E_SIZE", t.size * max_bins)
    with caplog.at_level(logging.WARNING):
        _, centers, _, max_err = fm._create_histogram(
            _brain_like_complex_field(), None, t
        )
    assert centers.size <= max_bins
    assert max_err > fm._BIN_TOL
    assert "binning the field map" in caplog.text


@pytest.mark.parametrize("t0", [-14e-3, 0.0])
@pytest.mark.parametrize("with_r2s", [True, False])
def test_auto_bins_pointwise_accuracy(t0, with_r2s):
    """With n_bins=-1, B C matches exp(Delta t) at every voxel to about _BIN_TOL.

    The binning error grows with max|t|, not with the readout extent: starting the
    readout at 0 doubles it for a fixed number of bins.
    """
    from mrinufft.extras.field_map import _BIN_TOL

    field = _brain_like_complex_field()
    if not with_r2s:
        field = 1j * field.imag
    t = np.linspace(t0, t0 + 28e-3, 300, dtype=np.float32)
    B, C, _ = get_orc_factorization("svd")(field, t, mask=None)
    approx = np.asarray(B) @ np.asarray(C).reshape(C.shape[0], -1)
    exact = np.exp(np.outer(t, field.ravel()))
    err = (np.abs(approx - exact) / np.abs(exact)).max(axis=0)
    assert err.max() < 1.1 * _BIN_TOL


@pytest.mark.parametrize("method", ["svd", "mti", "mfi"])
def test_constant_field_map(method):
    """A single occupied bin (fewer than L) does not break the factorizations."""
    b0 = np.full((8, 8), 50.0, dtype=np.float32)
    t = np.linspace(0, 5e-3, 64, dtype=np.float32)
    B, C, _ = get_orc_factorization(method)(b0, t, mask=None)
    C = np.stack([np.asarray(C[ll]) for ll in range(len(C))]).reshape(len(C), -1)
    exact = np.exp(np.outer(t, 2j * np.pi * b0.ravel()))
    npt.assert_allclose(np.asarray(B) @ C, exact, atol=1e-4)
