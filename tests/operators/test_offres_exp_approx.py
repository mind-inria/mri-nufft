"""Test off-resonance spatial coefficient and temporal interpolator estimation."""

import math

import numpy as np

import pytest
from pytest_cases import parametrize_with_cases


import mrinufft
from mrinufft._utils import get_array_module
from mrinufft.operators.base import CUPY_AVAILABLE
from mrinufft.operators.off_resonance import MRIFourierCorrected


from helpers import to_interface, assert_allclose
from helpers.factories import _param_array_interface
from case_fieldmaps import CasesB0maps, CasesZmaps


def calculate_true_offresonance_term(fieldmap, t, array_interface):
    """Calculate non-approximate off-resonance modulation term."""
    fieldmap = to_interface(fieldmap, array_interface)
    t = to_interface(t, array_interface)

    xp = get_array_module(fieldmap)
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


@_param_array_interface
@parametrize_with_cases("b0map, mask", cases=CasesB0maps)
def test_b0map_coeff(b0map, mask, array_interface):
    """Test exponential approximation for B0 field only."""
    if array_interface == "torch-gpu" and not CUPY_AVAILABLE:
        pytest.skip("GPU computations requires cupy")

    # Generate readout times
    tread = np.linspace(0.0, 5e-3, 501, dtype=np.float32)

    # Generate coefficients
    B, tl = mrinufft.get_interpolators_from_fieldmap(
        to_interface(b0map, array_interface), tread, mask=mask, n_time_segments=100
    )

    # Calculate spatial coefficients
    C = MRIFourierCorrected.get_spatial_coefficients(
        to_interface(2 * math.pi * 1j * b0map, array_interface), tl
    )

    # Assert properties
    assert B.shape == (100, 501)
    assert C.shape == (100, *b0map.shape)

    # Correct approximation
    expected = calculate_true_offresonance_term(
        0 + 2 * math.pi * 1j * b0map, tread, array_interface
    )
    actual = calculate_approx_offresonance_term(B, C)
    assert_allclose(actual, expected, atol=1e-3, rtol=1e-3, interface=array_interface)


@_param_array_interface
@parametrize_with_cases("zmap, mask", cases=CasesZmaps)
def test_zmap_coeff(zmap, mask, array_interface):
    """Test exponential approximation for complex Z = R2* + 1j *B0 field."""
    if array_interface == "torch-gpu" and CUPY_AVAILABLE is False:
        pytest.skip("GPU computations requires cupy")

    # Generate readout times
    tread = np.linspace(0.0, 5e-3, 501, dtype=np.float32)

    # Generate coefficients
    B, tl = mrinufft.get_interpolators_from_fieldmap(
        to_interface(zmap.imag, array_interface),
        tread,
        mask=mask,
        r2star_map=to_interface(zmap.real, array_interface),
        n_time_segments=100,
    )

    # Calculate spatial coefficients
    C = MRIFourierCorrected.get_spatial_coefficients(
        to_interface(2 * math.pi * zmap, array_interface), tl
    )

    # Assert properties
    assert B.shape == (100, 501)
    assert C.shape == (100, *zmap.shape)

    # Correct approximation
    expected = calculate_true_offresonance_term(
        2 * math.pi * zmap, tread, array_interface
    )
    actual = calculate_approx_offresonance_term(B, C)
    assert_allclose(actual, expected, atol=1e-3, rtol=1e-3, interface=array_interface)
