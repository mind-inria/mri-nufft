"""Test for update in trajectory, density, and sensitivity maps.

Only finufft, cufinufft and gpunufft support update.
"""

import numpy as np
import numpy.testing as npt
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture
from helpers import (
    param_array_interface,
    to_interface,
    from_interface,
)
from mrinufft import get_operator
from mrinufft._utils import get_array_module
from case_trajectories import CasesTrajectories


@fixture(scope="module")
@parametrize(
    "n_batch, n_coils, n_trans, sense",
    [
        (1, 1, 1, False),
        (3, 1, 1, False),
        (1, 4, 1, True),
        (1, 4, 1, False),
        (1, 4, 2, True),
        (1, 4, 2, False),
        (3, 2, 1, True),
        (3, 2, 1, False),
        (3, 4, 2, True),
        (3, 4, 2, False),
    ],
)
@parametrize_with_cases(
    "kspace_locs, shape",
    cases=CasesTrajectories,
    glob="*random*",
)
@parametrize(backend=["finufft", "cufinufft", "gpunufft"])
@parametrize(density=[False, True])
@parametrize(smaps_cached=[False, True])
def operator(
    request,
    kspace_locs,
    shape,
    n_coils=1,
    sense=None,
    n_batch=1,
    n_trans=1,
    density=False,
    backend="finufft",
    smaps_cached=False,
):
    """Generate a batch operator."""
    if n_trans != 1 and backend == "gpunufft":
        pytest.skip("Duplicate case.")
    if density and backend in ["cufinufft", "finufft"]:
        pytest.skip("Density estimation not supported for cufinufft and finufft.")
    if sense:
        smaps = 1j * np.random.rand(n_coils, *shape)
        smaps += np.random.rand(n_coils, *shape)
        smaps = smaps.astype(np.complex64)
        smaps /= np.linalg.norm(smaps, axis=0)
    else:
        smaps = None
    kspace_locs = kspace_locs.astype(np.float32)
    op_args = {
        "samples": kspace_locs,
        "shape": shape,
        "n_coils": n_coils,
        "n_batchs": n_batch,
        "squeeze_dims": False,
        "density": density,
        "smaps": smaps,
    }
    if backend in ["cufinufft"]:
        op_args["smaps_cached"] = smaps_cached
    else:
        if smaps_cached:
            pytest.skip(f"Skip test cause we dont have smaps_cached in {backend}")
    return get_operator(backend)(**op_args)


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


@fixture(scope="module")
def image_data(operator):
    """Generate a random image."""
    if operator.uses_sense:
        shape = (operator.n_batchs, *operator.shape)
    else:
        shape = (operator.n_batchs, operator.n_coils, *operator.shape)
    img = (1j * np.random.rand(*shape)).astype(operator.cpx_dtype)
    img += np.random.rand(*shape).astype(operator.cpx_dtype)
    return img


@fixture(scope="module")
def kspace_data(operator):
    """Generate a random image."""
    shape = (operator.n_batchs, operator.n_coils, operator.n_samples)
    kspace = (1j * np.random.rand(*shape)).astype(operator.cpx_dtype)
    kspace += np.random.rand(*shape).astype(operator.cpx_dtype)
    return kspace


@fixture(scope="module")
def new_smaps(operator):
    """Generate a random new smaps."""
    smaps = 1j * np.random.rand(operator.n_coils, *operator.shape)
    smaps += np.random.rand(operator.n_coils, *operator.shape)
    smaps = smaps.astype(np.complex64)
    smaps /= np.linalg.norm(smaps, axis=0)
    return smaps


@param_array_interface
def test_op_samples(
    operator,
    array_interface,
    image_data,
):
    """Test the batch type 2 (forward)."""
    image_data = to_interface(image_data, array_interface)
    jitter = np.random.rand(*operator.samples.shape).astype(np.float32)
    # Add very little noise to the trajectory, variance of 1e-3
    operator.samples += jitter / 100
    new_operator = update_operator(operator)
    kspace_changed = from_interface(operator.op(image_data), array_interface)
    kspace_true = from_interface(new_operator.op(image_data), array_interface)
    npt.assert_array_almost_equal(kspace_changed, kspace_true)


@param_array_interface
def test_adj_op_samples(
    operator,
    array_interface,
    kspace_data,
):
    """Test the batch type 1 (adjoint)."""
    kspace_data = to_interface(kspace_data, array_interface)
    jitter = np.random.rand(*operator.samples.shape).astype(np.float32)
    # Add very little noise to the trajectory, variance of 1e-3
    operator.samples += jitter / 100
    new_operator = update_operator(operator)
    image_changed = from_interface(operator.adj_op(kspace_data), array_interface)
    image_true = from_interface(new_operator.adj_op(kspace_data), array_interface)
    # Reduced accuracy for the GPU cases...
    npt.assert_allclose(image_changed, image_true, atol=1e-3, rtol=1e-3)


@param_array_interface
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
        npt.assert_allclose(image_changed, image_true, atol=1e-3, rtol=1e-3)


@param_array_interface
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


@param_array_interface
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
    npt.assert_allclose(image_changed, image_true, atol=1e-4, rtol=1e-4)
