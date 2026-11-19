"""Test for the cg function."""

from pytest import fixture, mark
from mrinufft import get_operator
from case_trajectories import CasesTrajectories

from helpers import image_from_op, param_backends

from mrinufft.extras import loss_l2_reg


@fixture(
    scope="module",
    params=param_backends(
        "bart",
        "finufft",
        "cufinufft",
        "ducc0",
        "gpunufft",
        "sigpy",
        "torchkbnufft-cpu",
        "torchkbnufft-gpu",
        "tensorflow",
    ),
)
def operator(request):
    """Generate an operator."""
    kspace_locs, shape = CasesTrajectories().case_random2D()
    return get_operator(request.param)(kspace_locs, shape, n_coils=1, smaps=None)


@fixture(scope="module")
def image_data(operator):
    """Generate a random image. Remains constant for the module."""
    return image_from_op(operator)


@mark.parametrize("optim", ["lsqr", "lsmr", "cg"])
def test_pinv(operator, image_data, optim):
    """Compare the interface to the raw NUDFT implementation."""
    kspace_nufft = operator.op(image_data).squeeze()

    _, residuals = operator.pinv_solver(
        kspace_nufft, optim=optim, max_iter=10, callback=loss_l2_reg
    )
    assert residuals[-1] <= residuals[0]
