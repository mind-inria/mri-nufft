"""Test for the NDFT implementations."""

import numpy as np
import scipy as sp
from pytest_cases import parametrize_with_cases

from mrinufft.operators.interfaces.nudft_numpy import (
    get_fourier_matrix,
    implicit_type1_ndft,
    implicit_type2_ndft,
    RawNDFT,
)

from case_trajectories import CasesTrajectories, case_grid1D
from helpers import assert_almost_allclose
from mrinufft import get_operator


@parametrize_with_cases(
    "kspace, shape",
    cases=[
        CasesTrajectories.case_random2D,
        CasesTrajectories.case_grid2D,
        CasesTrajectories.case_grid3D,
    ],
)
def test_ndft_implicit2(kspace, shape):
    """Test matching of explicit and implicit matrix for ndft."""
    matrix = get_fourier_matrix(kspace, shape)
    random_image = np.random.randn(*shape) + 1j * np.random.randn(*shape)

    linop_coef = implicit_type2_ndft(kspace, random_image.flatten(), shape)
    matrix_coef = matrix @ random_image.flatten()

    assert_almost_allclose(linop_coef, matrix_coef, atol=1e-4, rtol=1e-4, mismatch=5)


@parametrize_with_cases(
    "kspace, shape",
    cases=[
        CasesTrajectories.case_random2D,
        CasesTrajectories.case_grid2D,
        CasesTrajectories.case_grid3D,
    ],
)
def test_ndft_implicit1(kspace, shape):
    """Test matching of explicit and implicit matrix for ndft."""
    matrix = get_fourier_matrix(kspace, shape)
    random_kspace = 1j * np.random.randn(len(kspace))
    random_kspace += np.random.randn(len(kspace))

    linop_coef = implicit_type1_ndft(kspace, random_kspace.flatten(), shape)
    matrix_coef = matrix.conj().T @ random_kspace.flatten()

    assert_almost_allclose(linop_coef, matrix_coef, atol=1e-4, rtol=1e-4, mismatch=5)


@parametrize_with_cases(
    "kspace_grid, shape",
    cases=[
        case_grid1D,
        CasesTrajectories.case_grid2D,
        CasesTrajectories.case_grid3D,
    ],
)
def test_ndft_fft(kspace_grid, shape):
    """Test the raw ndft implementation."""
    ndft_op = RawNDFT(kspace_grid, shape)
    # Create a random image
    img = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex64)
    kspace = np.empty(ndft_op.n_samples, dtype=img.dtype)
    ndft_op.op(kspace, img)
    # Reshape the kspace to be on a grid (because the ndft is not doing it)
    kspace = kspace.reshape(img.shape)
    if len(shape) >= 2:
        kspace = kspace.swapaxes(0, 1)
    kspace_fft = sp.fft.fftn(sp.fft.fftshift(img))

    assert_almost_allclose(kspace, kspace_fft, atol=1e-4, rtol=1e-4, mismatch=5)
