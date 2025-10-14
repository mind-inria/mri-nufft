"""Test that the ORC NUFFT approximates the Conjugate Phase (CP) expression."""

from mrinufft.extras.field_map import get_complex_fieldmap_rad
from mrinufft.operators.interfaces.nudft_numpy import get_fourier_matrix
import numpy as np
import numpy.testing as npt

from pytest_cases import parametrize_with_cases, parametrize, fixture

from helpers import image_from_op, kspace_from_op, assert_correlate

from case_trajectories import CasesTrajectories

from mrinufft import get_operator
from mrinufft.extras.field_map import make_b0map, make_t2smap


def get_extended_fourier_matrix(ktraj, shape, cpx_fieldmap, readout_time):
    """Generate the extended fourier matrix with off-resonnance.

    For test purposes only.
    """
    base_fourier = get_fourier_matrix(ktraj, shape, normalize=True)
    off_grid = np.outer(readout_time, cpx_fieldmap.ravel())
    base_fourier *= np.exp(off_grid).astype(np.complex64)
    return base_fourier


@fixture(scope="module")
@parametrize_with_cases("kspace_locs, shape", cases=CasesTrajectories.case_random2D)
@parametrize("backend", ["finufft", "cufinufft", "gpunufft"])
def operator(kspace_locs, shape, backend):
    """Create an operator with off resonance mapping support."""
    return get_operator(backend)(
        kspace_locs, shape, n_coils=1, n_batchs=1, density=False, squeeze_dims=True
    )


@fixture(scope="module")
def orc_info(operator):
    """Augment the operator to use B0 setup."""
    b0_map, mask = make_b0map(operator.shape, b0range=(-300, 300))
    # t2s_map, _ = make_t2smap(operator.shape, t2svalue=15)
    # # Convert T2* map to R2* map
    # t2s_map = t2s_map * 1e-3  # ms -> s
    # r2s_map = 1.0 / (t2s_map + 1e-9)  # Hz
    # r2s_map = mask * r2s_map
    r2s_map = None

    cpx_fieldmap = get_complex_fieldmap_rad(b0_map, r2s_map)
    readout_time = np.linspace(0, 5e-2, len(operator.samples), dtype=np.float32)
    cp_matrix = get_extended_fourier_matrix(
        operator.samples, operator.shape, cpx_fieldmap, readout_time
    )

    orc_nufft = operator.with_off_resonance_correction(
        b0_map=b0_map,
        r2star_map=r2s_map,
        mask=mask,
        readout_time=readout_time,
    )

    return orc_nufft, cp_matrix


@fixture(scope="module")
def image_data(operator):
    """Generate a random image. Remains constant for the module."""
    return image_from_op(operator)


@fixture(scope="module")
def kspace_data(operator):
    """Generate a random kspace. Remains constant for the module."""
    return kspace_from_op(operator)


def test_orc_forward(
    orc_info,
    image_data,
):
    """Test that the forward approximation works."""
    orc_nufft, ext_mat = orc_info
    ksp = orc_nufft.op(image_data)
    ksp_ideal = ext_mat @ image_data.ravel()

    assert_correlate(ksp.squeeze(), ksp_ideal, slope_err=0.005)


def test_orc_adjoint(orc_info, kspace_data):
    """Test taht the adjoint approximation works."""
    orc_nufft, ext_mat = orc_info
    img = orc_nufft.adj_op(kspace_data)
    img_ideal = ext_mat.conj().T @ kspace_data.ravel()
    img_ideal = img_ideal.reshape(orc_nufft.shape)
    assert_correlate(img.squeeze(), img_ideal, slope_err=0.005)
