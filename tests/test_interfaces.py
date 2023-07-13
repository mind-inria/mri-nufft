"""Test the interfaces module."""
import numpy as np
import scipy as sp
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture

from mrinufft.operators.interfaces.nudft_numpy import RawNDFT, FourierOperatorCPU
from mrinufft import get_operator
from mrinufft.trajectories import initialize_2D_radial, initialize_3D_from_2D_expansion


class CasesTrajectories:
    """Trajectories cases we want to test.

    Each case return a sampling pattern in k-space and the shape of the image.
    """

    def case_random2D(self, M=1000, N=64, pdf="uniform", seed=0):
        """Create a random 2D trajectory."""
        np.random.seed(seed)
        samples = np.random.rand(M, 2) - 0.5
        return samples, (N, N)

    def case_random3D(self, M=20000, N=64, pdf="uniform", seed=0):
        """Create a random 3D trajectory."""
        np.random.seed(seed)
        samples = np.random.rand(M, 3) - 0.5
        return samples, (N, N, N)

    def case_radial2D(self, Nc=10, Ns=100, N=64):
        """Create a 2D radial trajectory."""
        trajectory = initialize_2D_radial(Nc, Ns)
        return trajectory, (N, N)

    def case_radial3D(self, Nc=10, Ns=10, Nr=10, N=64, expansion="rotation"):
        """Create a 3D radial trajectory."""
        trajectory = initialize_3D_from_2D_expansion("radial", expansion, Nc, Ns, Nr)
        return trajectory, (N, N, N)

    def case_grid1D(self, N=256):
        """Create a 1D cartesian grid of frequencies locations."""
        freq_1d = sp.fft.fftfreq(N)
        return freq_1d.reshape(-1, 1), (N,)

    def case_grid2D(self, N=16):
        """Create a 2D cartesian grid of frequencies locations."""
        freq_1d = sp.fft.fftfreq(N)
        freq_2d = np.stack(np.meshgrid(freq_1d, freq_1d), axis=-1)
        return freq_2d.reshape(-1, 2), (N, N)

    def case_grid3D(self, N=16):
        """Create a 3D cartesian grid of frequencies locations."""
        freq_1d = sp.fft.fftfreq(N)
        freq_3d = np.stack(np.meshgrid(freq_1d, freq_1d, freq_1d), axis=-1)
        return freq_3d.reshape(-1, 3), (N, N, N)


@parametrize_with_cases(
    "kspace_grid, shape",
    cases=[
        CasesTrajectories.case_grid1D,
        CasesTrajectories.case_grid2D,
    ],  # 3D is ignored (too much possibility for the reordering)
)
def test_ndft_matrix(kspace_grid, shape):
    """Test that  the ndft matrix is a good matrix for doing fft."""
    ndft_op = RawNDFT(kspace_grid, shape, explicit_matrix=True)
    # Create a random image
    fft_matrix = [None] * len(shape)
    for i in range(len(shape)):
        fft_matrix[i] = sp.fft.fft(np.eye(shape[i]), norm="ortho")
    fft_mat = fft_matrix[0]
    if len(shape) == 2:
        fft_mat = fft_matrix[0].flatten()[:, None] @ fft_matrix[1].flatten()[None, :]
        fft_mat = (
            fft_mat.reshape(shape * 2)
            .transpose(2, 0, 1, 3)
            .reshape((np.prod(shape),) * 2)
        )
    assert np.allclose(ndft_op._fourier_matrix, fft_mat)


@parametrize_with_cases(
    "kspace_grid, shape",
    cases=[
        CasesTrajectories.case_grid1D,
        CasesTrajectories.case_grid2D,
        CasesTrajectories.case_grid3D,
    ],
)
def test_ndft_fft(kspace_grid, shape):
    """Test the raw ndft implementation."""
    ndft_op = RawNDFT(kspace_grid, shape, explicit_matrix=True)
    # Create a random image
    # TODO: use a fixture
    img = np.random.randn(*shape) + 1j * np.random.randn(*shape)

    kspace = ndft_op.op(img).reshape(img.shape)
    if len(shape) >= 2:
        kspace = kspace.swapaxes(0, 1)
    kspace_fft = sp.fft.fftn(img, norm="ortho")

    assert np.allclose(kspace, kspace_fft)


@fixture(scope="module")
@parametrize_with_cases("kspace_locs, shape", cases=CasesTrajectories)
def operator_image(
    request,
    backend="finufft",
    dtype="complex64",
    kspace_locs=None,
    shape=None,
    n_coils=1,
    smaps=None,
):
    """Generate a random image and smaps."""
    if smaps is None and n_coils > 1:
        img = np.random.rand(n_coils, *shape)
    elif smaps is not None and n_coils > 1:
        img = np.random.rand(*shape).astype(dtype)
        smaps = np.random.randn(n_coils, *shape).astype(dtype)
        smaps += 1j * np.random.randn((n_coils, *shape))
        smaps = smaps.astype(dtype)
    else:
        img = np.random.randn(*shape).astype(dtype)
        img += 1j * np.random.randn(*shape)

    nufft_op = get_operator(backend)(
        kspace_locs, img.shape, n_coils=n_coils, smaps=smaps
    )

    return nufft_op, img


def test_interfaces(operator_image):
    """Test the raw ndft implementation."""
    nufft_op, img = operator_image
    kspace_nufft = nufft_op.op(img)

    # ndft_op = get_operator("numpy")(
    #     nufft_op.samples, nufft_op.shape, n_coils=nufft_op.n_coils, smaps=nufft_op.smaps
    # )

    ndft_op = FourierOperatorCPU(
        nufft_op.samples,
        nufft_op.shape,
        n_coils=nufft_op.n_coils,
        smaps=nufft_op.smaps,
        raw_op=RawNDFT(nufft_op.samples, nufft_op.shape),
    )

    kspace = ndft_op.op(img).reshape(img.shape)

    assert np.allclose(kspace, kspace_nufft)
