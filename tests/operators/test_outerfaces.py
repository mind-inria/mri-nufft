"""Test the outerfaces (MRPro, DeepInv) interfaces."""

import numpy as np
from numpy.typing import NDArray
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture
from case_trajectories import CasesTrajectories
from mrinufft.operators import get_operator
from mrinufft._array_compat import MRPRO_AVAILABLE, AUTOGRAD_AVAILABLE

TORCH_AVAILABLE = True
try:
    import torch
except ImportError:
    TORCH_AVAILABLE = False


@fixture(scope="module")
@parametrize(backend=["finufft"])
@parametrize_with_cases(
    "kspace_loc, shape",
    cases=[
        CasesTrajectories.case_grid2D,
        CasesTrajectories.case_nyquist_radial2D,
    ],
)
def operator(kspace_loc: NDArray, shape: tuple[int, ...], backend: str):
    """Create NUFFT operator for outerface tests."""
    kspace_loc = kspace_loc.astype(np.float32)
    return get_operator(backend_name=backend)(
        samples=kspace_loc,
        shape=shape,
        squeeze_dims=False,
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
@pytest.mark.skipif(not MRPRO_AVAILABLE, reason="MRPro is not installed")
def test_mrpro_interface_forward(operator):
    """Test MRPro interface forward pass."""
    from mrinufft.operators.outerfaces import MRProNufftInterface

    mrpro_op = MRProNufftInterface(operator)

    # Create image data
    img_shape = operator.img_full_shape
    image = np.random.randn(*img_shape) + 1j * np.random.randn(*img_shape)
    image_torch = torch.from_numpy(image).to(torch.complex64)

    # Forward pass
    kspace_tuple = mrpro_op.forward(image_torch)

    # Check output is a tuple
    assert isinstance(kspace_tuple, tuple)
    assert len(kspace_tuple) == 1

    # Check shape
    expected_ksp_shape = operator.ksp_full_shape
    assert kspace_tuple[0].shape == expected_ksp_shape


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
@pytest.mark.skipif(not MRPRO_AVAILABLE, reason="MRPro is not installed")
def test_mrpro_interface_adjoint(operator):
    """Test MRPro interface adjoint pass."""
    from mrinufft.operators.outerfaces import MRProNufftInterface

    mrpro_op = MRProNufftInterface(operator)

    # Create k-space data
    ksp_shape = operator.ksp_full_shape
    kspace = np.random.randn(*ksp_shape) + 1j * np.random.randn(*ksp_shape)
    kspace_torch = torch.from_numpy(kspace).to(torch.complex64)

    # Adjoint pass
    image = mrpro_op.adjoint((kspace_torch,))

    # Check shape
    expected_img_shape = operator.img_full_shape
    assert image.shape == expected_img_shape


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
@pytest.mark.skipif(not MRPRO_AVAILABLE, reason="MRPro is not installed")
def test_mrpro_interface_consistency(operator):
    """Test MRPro interface forward/adjoint consistency."""
    from mrinufft.operators.outerfaces import MRProNufftInterface

    mrpro_op = MRProNufftInterface(operator)

    # Create image data
    img_shape = operator.img_full_shape
    image = np.random.randn(*img_shape) + 1j * np.random.randn(*img_shape)
    image_torch = torch.from_numpy(image).to(torch.complex64)

    # Forward pass
    kspace_tuple = mrpro_op.forward(image_torch)

    # Adjoint pass
    image_reconstructed = mrpro_op.adjoint(kspace_tuple)

    # Check shapes match
    assert image_reconstructed.shape == image_torch.shape


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
@pytest.mark.skipif(not MRPRO_AVAILABLE, reason="MRPro is not installed")
def test_mrpro_interface_ishape_oshape(operator):
    """Test MRPro interface ishape and oshape properties."""
    from mrinufft.operators.outerfaces import MRProNufftInterface

    mrpro_op = MRProNufftInterface(operator)

    assert mrpro_op.ishape == operator.img_full_shape
    assert mrpro_op.oshape == operator.ksp_full_shape


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_mrpro_interface_structure():
    """Test MRPro interface has expected methods."""
    from mrinufft.operators.outerfaces import MRProNufftInterface

    assert hasattr(MRProNufftInterface, "forward")
    assert hasattr(MRProNufftInterface, "adjoint")
    assert hasattr(MRProNufftInterface, "make")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_mrpro_make_classmethod(operator):
    """Test MRPro interface make classmethod."""
    from mrinufft.operators.outerfaces import MRProNufftInterface

    if not MRPRO_AVAILABLE:
        pytest.skip("MRPro is not installed")

    mrpro_op = MRProNufftInterface.make(operator)
    assert isinstance(mrpro_op, MRProNufftInterface)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is not installed")
def test_mrpro_no_autograd_error(operator):
    """Test error when backend does not support autograd."""
    from mrinufft.operators.outerfaces import MRProNufftInterface

    if not MRPRO_AVAILABLE:
        pytest.skip("MRPro is not installed")

    # Create an operator that doesn't support autograd
    # The finufft backend should support autograd, so we need to mock it
    class FakeOperator:
        autograd_available = False
        img_full_shape = (1, 1, 32, 32)
        ksp_full_shape = (1, 1, 512)

    fake_op = FakeOperator()

    with pytest.raises(ValueError, match="does not support auto-differentiation"):
        MRProNufftInterface(fake_op)
