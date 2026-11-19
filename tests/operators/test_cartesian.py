"""Dedicated tests for Cartesian operator."""

import numpy as np
from pytest import fixture
from mrinufft import get_operator

from helpers import assert_adjoint, case_id, cases_of, parametrize_array_interface


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


@fixture(scope="module", params=cases_of(CasesCartesian), ids=case_id)
def operator(request):
    """Generate a Cartesian operator."""
    mask, shape = request.param()
    return get_operator("cartesian")(mask, shape)


@parametrize_array_interface()
def test_cartesian_autoadjoint(operator, array_interface):
    """Test the auto-adjoint property of the Cartesian operator."""
    assert_adjoint(operator, array_interface)
