"""Test for update in trajectory, density, and sensitivity maps.

Only finufft, cufinufft and gpunufft support update.
"""

import numpy as np
import numpy.testing as npt
import pytest
from pytest import fixture, mark
from helpers import (
    CasesBatch,
    CasesTrajectories,
    case_id,
    cases_of,
    from_interface,
    parametrize_array_interface,
    random_smaps,
    to_interface,
)
from mrinufft import get_operator
from mrinufft._utils import get_array_module


def _variant(backend, **kwargs):
    name = "-".join([backend, *kwargs])
    return pytest.param((backend, kwargs), marks=mark.backend(backend), id=name)


@fixture(
    params=[
        _variant("finufft"),
        _variant("cufinufft"),
        _variant("cufinufft", smaps_cached=True),
        _variant("gpunufft"),
        _variant("gpunufft", density=True),
    ]
)
def variant(request):
    """Backend and its specific options."""
    return request.param


@fixture(params=cases_of(CasesTrajectories, "random2D", "random3D"), ids=case_id)
def trajectory(request):
    """Sampling pattern and image shape."""
    return request.param()


@fixture(
    params=cases_of(CasesBatch, "single", "sense", "batch_sense", "batch_coils_ntrans"),
    ids=case_id,
)
def config(request):
    """Batch, coils and SENSE configuration (n_trans is not used here)."""
    return request.param()


@fixture
def operator(variant, trajectory, config):
    """Generate an operator, rebuilt for each test as the tests modify it."""
    backend, options = variant
    kspace_locs, shape = trajectory
    n_batchs, n_coils, _, sense = config
    return get_operator(backend)(
        samples=kspace_locs.astype(np.float32),
        shape=shape,
        n_coils=n_coils,
        n_batchs=n_batchs,
        squeeze_dims=False,
        smaps=random_smaps(n_coils, shape) if sense else None,
        **{"density": False, **options},
    )


def update_operator(operator):
    """Generate a new operator with updated trajectory."""
    op_args = {
        k: getattr(operator, k)
        for k in ["samples", "shape", "n_coils", "n_batchs", "density", "smaps"]
    }
    op_args["squeeze_dims"] = False
    if operator.backend == "cufinufft":
        op_args["smaps_cached"] = operator.smaps_cached
        if operator.smaps is not None and not isinstance(operator.smaps, np.ndarray):
            op_args["smaps"] = operator.smaps.get()
    return get_operator(operator.backend)(**op_args)


@fixture
def image_data(operator):
    """Generate a random image."""
    if operator.uses_sense:
        shape = (operator.n_batchs, *operator.shape)
    else:
        shape = (operator.n_batchs, operator.n_coils, *operator.shape)
    img = (1j * np.random.rand(*shape)).astype(operator.cpx_dtype)
    img += np.random.rand(*shape).astype(operator.cpx_dtype)
    return img


@fixture
def kspace_data(operator):
    """Generate a random image."""
    shape = (operator.n_batchs, operator.n_coils, operator.n_samples)
    kspace = (1j * np.random.rand(*shape)).astype(operator.cpx_dtype)
    kspace += np.random.rand(*shape).astype(operator.cpx_dtype)
    return kspace


@fixture
def new_smaps(operator):
    """Generate a random new smaps."""
    return random_smaps(operator.n_coils, operator.shape)


@parametrize_array_interface("numpy", "cupy")
def test_op_samples(
    operator,
    array_interface,
    image_data,
):
    """Test the batch type 2 (forward)."""
    image_data = to_interface(image_data, array_interface)
    jitter = np.random.rand(*operator.samples.shape).astype(np.float32)
    # Add very little noise to the trajectory, variance of 1e-3
    xp = get_array_module(operator.samples)
    operator.samples += xp.array(jitter) / 100
    new_operator = update_operator(operator)
    kspace_changed = from_interface(operator.op(image_data), array_interface)
    kspace_true = from_interface(new_operator.op(image_data), array_interface)
    npt.assert_array_almost_equal(kspace_changed, kspace_true)


@parametrize_array_interface("numpy", "cupy")
def test_adj_op_samples(
    operator,
    array_interface,
    kspace_data,
):
    """Test the batch type 1 (adjoint)."""
    kspace_data = to_interface(kspace_data, array_interface)
    jitter = np.random.rand(*operator.samples.shape).astype(np.float32)
    # Add very little noise to the trajectory, variance of 1e-3
    xp = get_array_module(operator.samples)
    operator.samples += xp.array(jitter) / 100
    new_operator = update_operator(operator)
    image_changed = from_interface(operator.adj_op(kspace_data), array_interface)
    image_true = from_interface(new_operator.adj_op(kspace_data), array_interface)
    # Reduced accuracy for the GPU cases...
    npt.assert_allclose(
        image_changed,
        image_true,
        atol=5e-2 if operator.backend == "finufft" else 1e-3,
        rtol=1e-3,
    )


@parametrize_array_interface("numpy", "cupy")
def test_adj_op_density(
    operator,
    array_interface,
    kspace_data,
):
    """Test the batch type 1 (adjoint)."""
    kspace_data = to_interface(kspace_data, array_interface)
    jitter = np.random.rand(operator.samples.shape[0]).astype(np.float32)
    # Add very little noise to the trajectory, variance of 1e-3
    if operator.uses_density:
        # Test density can be updated
        xp = get_array_module(operator.density)

        operator.density += xp.array(jitter) / 100
    else:
        # Test that operator can handle density added later
        operator.density = 1e-2 + jitter / 100
    new_operator = update_operator(operator)
    image_changed = from_interface(operator.adj_op(kspace_data), array_interface)
    image_true = from_interface(new_operator.adj_op(kspace_data), array_interface)
    # Reduced accuracy for the GPU cases...
    npt.assert_allclose(image_changed, image_true, atol=1e-3, rtol=1e-3)
    if operator.uses_density:
        # Check if the operator can handle removing density compensation
        operator.density = None
        new_operator = update_operator(operator)
        image_changed = from_interface(operator.adj_op(kspace_data), array_interface)
        image_true = from_interface(new_operator.adj_op(kspace_data), array_interface)
        npt.assert_allclose(
            image_changed,
            image_true,
            atol=5e-2 if operator.backend == "finufft" else 1e-3,
            rtol=1e-3,
        )


@parametrize_array_interface("numpy", "cupy")
def test_op_smaps_update(
    operator,
    array_interface,
    image_data,
    new_smaps,
):
    """Test the batch type 2 (forward) with smaps."""
    image_data = to_interface(image_data, array_interface)
    if operator.smaps is None:
        pytest.skip("Skipping as we dont have smaps")
    operator.smaps = new_smaps
    new_operator = update_operator(operator)
    kspace_changed = from_interface(operator.op(image_data), array_interface)
    kspace_true = from_interface(new_operator.op(image_data), array_interface)
    npt.assert_array_almost_equal(kspace_changed, kspace_true)


@parametrize_array_interface("numpy", "cupy")
def test_adj_op_smaps_update(
    operator,
    array_interface,
    kspace_data,
    new_smaps,
):
    """Test the batch type 1 (adjoint)."""
    kspace_data = to_interface(kspace_data, array_interface)
    if operator.smaps is None:
        pytest.skip("Skipping as we dont have smaps")
    operator.smaps = new_smaps
    new_operator = update_operator(operator)
    image_changed = from_interface(operator.adj_op(kspace_data), array_interface)
    image_true = from_interface(new_operator.adj_op(kspace_data), array_interface)
    npt.assert_allclose(
        image_changed,
        image_true,
        atol=3e-2 if operator.backend == "finufft" else 1e-4,
        rtol=1e-4,
    )
