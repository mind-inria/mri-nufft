"""Tests for the optional GROG backend."""

import numpy as np
import pytest

from mrinufft import get_operator
from mrinufft.operators import check_backend, list_backends
from mrinufft.operators.interfaces.grog import GROG_AVAILABLE


def test_grog_backend_registration():
    """GROG is registered even when its optional dependency is unavailable."""
    assert "grog" in list_backends()
    assert check_backend("grog") is GROG_AVAILABLE


def test_grog_backend_unavailable_error():
    """Selecting GROG reports how to install its optional dependency."""
    if GROG_AVAILABLE:
        pytest.skip("GROG is installed.")
    with pytest.raises(ValueError, match=r"mri-nufft\[grog\]"):
        get_operator("grog")


@pytest.mark.skipif(not GROG_AVAILABLE, reason="PyGROG is not installed.")
def test_grog_forward_and_adjoint():
    """The GROG transforms are adjoints under the MRI interface."""
    rng = np.random.default_rng(0)
    shape = (8, 8)
    samples = rng.uniform(-0.4, 0.4, size=(32, 2)).astype(np.float32)
    operator = get_operator("grog")(
        samples,
        shape,
    )
    image = (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ).astype(np.complex64)
    kspace = (
        rng.standard_normal(operator.n_samples)
        + 1j * rng.standard_normal(operator.n_samples)
    ).astype(np.complex64)

    forward = operator.op(image)
    adjoint = operator.adj_op(kspace)
    np.testing.assert_allclose(
        np.vdot(forward, kspace),
        np.vdot(image, adjoint),
        rtol=1e-4,
        atol=1e-4,
    )
