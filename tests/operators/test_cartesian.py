"""Dedicated tests for Cartesian operator."""

import numpy as np
from pytest import fixture
from mrinufft import get_operator

from helpers import (
    case_id,
    cases_of,
    image_from_op,
    parametrize_array_interface,
    to_interface,
    from_interface,
)


class CasesCartesian:
    def case_full2D(self, N=64):
        return np.ones((N, N), dtype=bool), (N, N)

    def case_full3D(self, N=64):
        return np.ones((N, N, N), dtype=bool), (N, N, N)

    def case_random_lines2D(self, N=64, n_lines=10, seed=0):
        np.random.seed(seed)
        mask = np.zeros((N, N), dtype=bool)
        line_indices = np.random.choice(N, size=n_lines, replace=False)
        mask[line_indices] = True
        return mask, (N, N)


def kspace_from_op(operator):
    """Generate kspace data for a Cartesian  operator."""
    # For Cartesian, the kspace data is like the image.
    return image_from_op(operator)[:, :, operator.mask]


@fixture(scope="module", params=cases_of(CasesCartesian), ids=case_id)
def operator(request):
    """Generate a Cartesian operator."""
    mask, shape = request.param()
    return get_operator("cartesian")(mask, shape)


@parametrize_array_interface()
def test_cartesian_autoadjoint(operator, array_interface):
    """Test the auto-adjoint property of the Cartesian operator."""
    reldiff = np.zeros(10)

    for i in range(10):
        img_data = to_interface(image_from_op(operator), array_interface)
        ksp_data = to_interface(kspace_from_op(operator), array_interface)
        kspace = operator.op(img_data)

        rightadjoint = np.vdot(
            from_interface(kspace, array_interface),
            from_interface(ksp_data, array_interface),
        )

        image = operator.adj_op(ksp_data)
        leftadjoint = np.vdot(
            from_interface(img_data, array_interface),
            from_interface(image, array_interface),
        )
        reldiff[i] = abs(rightadjoint - leftadjoint) / abs(leftadjoint)
    assert np.mean(reldiff) < 5e-5
