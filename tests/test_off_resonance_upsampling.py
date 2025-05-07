import numpy as np
import pytest

from mrinufft.operators.off_resonance import MRIFourierCorrected
from mrinufft import get_operator


def test_b0_map_upsampling_warns_and_matches_shape():
    """Test that MRIFourierCorrected upscales the b0_map and warns if shape mismatch exists."""

    shape_target = (16, 16, 16)
    b0_shape = (8, 8, 8)

    b0_map = np.ones(b0_shape, dtype=np.float32)
    kspace = np.zeros((10, 3), dtype=np.float32)
    smaps = np.ones((1, *shape_target), dtype=np.complex64)
    readout_time = np.ones(10, dtype=np.float32)

    nufft = get_operator("gpunufft")(
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
        assert op.B.shape[1] == len(readout_time)
        assert op.shape == shape_target
