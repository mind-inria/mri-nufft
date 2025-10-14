"""Test off-resonance spatial coefficient and temporal interpolator estimation."""

from mrinufft.extras import get_orc_factorization, get_complex_fieldmap_rad
import numpy as np
import numpy.testing as npt
import pytest
from pytest_cases import parametrize_with_cases, parametrize


from mrinufft.operators.off_resonance import MRIFourierCorrected
from mrinufft import get_operator
from mrinufft.extras import make_b0map, make_t2smap

from helpers import to_interface
from helpers.factories import _param_array_interface_np_cp, from_interface


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


def test_b0_map_upsampling_warns_and_matches_shape():
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

    with pytest.warns(UserWarning):
        op = MRIFourierCorrected(
            nufft,
            b0_map=b0_map,
            readout_time=readout_time,
        )

        # check that no exception is raised and internal shape matches
        assert op.B.shape[0] == len(readout_time)
        assert op.shape == shape_target
