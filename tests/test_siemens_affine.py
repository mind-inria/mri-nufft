"""Affine of Siemens Twix data with slice oversampling."""

import numpy as np
import pytest

from mrinufft.io.siemens import twix2nifti_affine


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
