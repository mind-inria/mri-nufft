"""Test for the NDFT implementations."""
import numpy as np
import numpy.testing as npt
import scipy as sp
from pytest_cases import parametrize_with_cases, parametrize

from mrinufft.operators.interfaces.nudft_numpy import (
    get_fourier_matrix,
    implicit_type1_ndft,
    implicit_type2_ndft,
    RawNDFT,
)
from mrinufft.operators.interfaces.pykeops import KeopsRawNDFT

from case_trajectories import CasesTrajectories, case_grid1D


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
    npt.assert_allclose(ndft_matrix, fft_mat)


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

    npt.assert_allclose(linop_coef, matrix_coef)


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

    npt.assert_allclose(linop_coef, matrix_coef)


@parametrize("klass", [RawNDFT, KeopsRawNDFT])
@parametrize_with_cases(
    "kspace_grid, shape",
    cases=[
        case_grid1D,
        CasesTrajectories.case_grid2D,
        CasesTrajectories.case_grid3D,
    ],
)
def test_ndft_fft(klass, kspace_grid, shape):
    """Test the raw ndft implementation."""
    ndft_op = klass(kspace_grid, shape)
    # Create a random image
    img = (np.random.randn(*shape) + 1j * np.random.randn(*shape)).astype(np.complex64)
    kspace = np.empty(ndft_op.n_samples, dtype=img.dtype)
    ndft_op.op(kspace, img)
    # Reshape the kspace to be on a grid (because the ndft is not doing it)
    kspace = kspace.reshape(img.shape)
    if len(shape) >= 2:
        kspace = kspace.swapaxes(0, 1)
    kspace_fft = sp.fft.fftn(img)

    npt.assert_allclose(kspace, kspace_fft, atol=1e-3, rtol=1e-3)


@parametrize("klass", [RawNDFT, KeopsRawNDFT])
@parametrize_with_cases(
    "kspace_grid, shape",
    cases=[
        case_grid1D,
        CasesTrajectories.case_grid2D,
        CasesTrajectories.case_grid3D,
    ],
)
def test_ndft_ifft(klass, kspace_grid, shape):
    """Test the raw ndft implementation."""
    ndft_op = klass(kspace_grid, shape)
    # Create a random image
    kspace = (
        np.random.randn(*ndft_op.shape) + 1j * np.random.randn(*ndft_op.shape)
    ).astype(np.complex64)
    img = np.empty(ndft_op.shape, dtype=kspace.dtype)
    ndft_op.adj_op(kspace.flatten(), img)
    # Reshape the kspace to be on a grid (because the ndft is not doing it)
    if len(shape) >= 2:
        img = img.swapaxes(0, 1)
    img_ifft = sp.fft.ifftn(kspace)  # FIXME: find the correct norm factor.
    img_ifft *= np.prod(shape)
    npt.assert_allclose(img, img_ifft, atol=1e-3, rtol=1e-3)
