"""Tests for the reference backend."""

from pytest import fixture
from case_trajectories import CasesTrajectories

from mrinufft import get_operator
from mrinufft.operators.interfaces.nudft_numpy import MRInumpy
from helpers import (
    assert_almost_allclose,
    case_id,
    cases_of,
    kspace_from_op,
    image_from_op,
)


@fixture(
    scope="module",
    params=cases_of(CasesTrajectories, "random2D", "grid2D", "grid3D"),
    ids=case_id,
)
def ref_operator(request, ref_backend):
    """Generate a NFFT operator, matching the property of the first operator."""
    kspace, shape = request.param()
    return get_operator(ref_backend)(kspace, shape)


@fixture(scope="module")
def ndft_operator(ref_operator):
    """Get a NDFT operator matching the reference operator."""
    return MRInumpy(ref_operator.samples, ref_operator.shape)


@fixture(scope="module")
def image_data(ref_operator):
    """Generate a random image. Remains constant for the module."""
    return image_from_op(ref_operator)


@fixture(scope="module")
def kspace_data(ref_operator):
    """Generate a random kspace. Remains constant for the module."""
    return kspace_from_op(ref_operator)


def test_ref_nufft_forward(ref_operator, ndft_operator, image_data):
    """Test that the reference nufft matches the NDFT."""
    nufft_ksp = ref_operator.op(image_data)
    ndft_ksp = ndft_operator.op(image_data)

    assert_almost_allclose(
        nufft_ksp,
        ndft_ksp,
        atol=1e-4,
        rtol=1e-4,
        mismatch=5,
    )


def test_ref_nufft_adjoint(ref_operator, ndft_operator, kspace_data):
    """Test that the reference nufft matches the NDFT adjoint."""
    nufft_img = ref_operator.adj_op(kspace_data)
    ndft_img = ndft_operator.adj_op(kspace_data)

    assert_almost_allclose(
        nufft_img,
        ndft_img,
        atol=1e-4,
        rtol=1e-4,
        mismatch=5,
    )
