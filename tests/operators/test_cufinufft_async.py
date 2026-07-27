"""Test the async CUDA-stream pipelining of the cufinufft host-input code paths."""

import numpy as np
import numpy.testing as npt
import pytest
from pytest_cases import parametrize

from mrinufft import get_operator
from mrinufft.operators.base import check_backend
from mrinufft.trajectories import initialize_2D_radial

from helpers.factories import image_from_op, kspace_from_op

if not check_backend("cufinufft"):
    pytest.skip("cufinufft backend not available", allow_module_level=True)


def _make_operators(n_batchs, n_coils, n_trans, sense):
    """Build a pair of identical operators, only differing by async_transfer."""
    samples = initialize_2D_radial(32 * 4, 16).astype(np.float32)
    shape = (32, 32)
    if sense:
        smaps = 1j * np.random.rand(n_coils, *shape)
        smaps += np.random.rand(n_coils, *shape)
        smaps = smaps.astype(np.complex64)
        smaps /= np.linalg.norm(smaps, axis=0)
    else:
        smaps = None

    ops = [
        get_operator("cufinufft")(
            samples,
            shape,
            n_coils=n_coils,
            smaps=smaps,
            n_batchs=n_batchs,
            n_trans=n_trans,
            squeeze_dims=False,
            async_transfer=async_transfer,
        )
        for async_transfer in (False, True)
    ]
    return ops


@parametrize(
    "n_batchs, n_coils, n_trans, sense",
    [
        (1, 1, 1, False),  # n_iter == 1, exercises the fallback path.
        (3, 4, 2, False),  # n_iter > 1, calibless, exercises the pipeline.
        (3, 4, 2, True),  # n_iter > 1, sense, exercises the pipeline.
        (3, 4, 1, True),  # n_iter > 1, sense, no batching over n_trans.
    ],
)
def test_async_transfer_matches_sync(n_batchs, n_coils, n_trans, sense):
    """Compare op/adj_op results with async_transfer on and off."""
    op_sync, op_async = _make_operators(n_batchs, n_coils, n_trans, sense)

    image_data = image_from_op(op_sync)
    kspace_data = kspace_from_op(op_sync)

    ksp_sync = op_sync.op(image_data)
    ksp_async = op_async.op(image_data)
    npt.assert_allclose(ksp_sync, ksp_async, rtol=1e-6, atol=1e-6)

    img_sync = op_sync.adj_op(kspace_data)
    img_async = op_async.adj_op(kspace_data)
    npt.assert_allclose(img_sync, img_async, rtol=1e-6, atol=1e-6)


def test_async_transfer_readonly():
    """Async path must not mutate the caller's input arrays."""
    op_sync, op_async = _make_operators(3, 4, 2, sense=True)

    image_data = image_from_op(op_async)
    kspace_data = kspace_from_op(op_async)
    image_ref = image_data.copy()
    kspace_ref = kspace_data.copy()
    image_data.setflags(write=False)
    kspace_data.setflags(write=False)

    op_async.op(image_data)
    op_async.adj_op(kspace_data)

    npt.assert_equal(image_data, image_ref)
    npt.assert_equal(kspace_data, kspace_ref)
