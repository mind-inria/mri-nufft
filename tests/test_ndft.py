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


@parametrize_with_cases(
    "kspace_grid, shape",
    cases=[
        case_grid1D,
        CasesTrajectories.case_grid2D,
    ],  # 3D is ignored (too much possibility for the reordering)
)
def test_ndft_grid_matrix(kspace_grid, shape):
    """Test that  the ndft matrix is a good matrix for doing fft."""
    ndft_matrix = get_fourier_matrix(kspace_grid, shape)
    # Create a random image
    fft_matrix = [None] * len(shape)
    for i in range(len(shape)):
        fft_matrix[i] = sp.fft.fft(np.eye(shape[i]))
    fft_mat = fft_matrix[0]
    if len(shape) == 2:
        fft_mat = fft_matrix[0].flatten()[:, None] @ fft_matrix[1].flatten()[None, :]
        fft_mat = (
            fft_mat.reshape(shape * 2)
            .transpose(2, 0, 1, 3)
            .reshape((np.prod(shape),) * 2)
        )
    assert np.allclose(ndft_matrix, fft_mat)


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

    assert np.allclose(linop_coef, matrix_coef)


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

    assert np.allclose(linop_coef, matrix_coef)


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
    kspace_fft = sp.fft.fftn(img)

    assert_almost_allclose(kspace, kspace_fft, atol=1e-5, rtol=1e-5, mismatch=5)
