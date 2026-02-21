"""Dedicated tests for Cartesian operator."""

import numpy as np
import pytest
from pytest_cases import parametrize_with_cases, parametrize, fixture
from mrinufft import get_operator

from helpers import (
    image_from_op,
    to_interface,
    from_interface,
)
from helpers.factories import _param_array_interface


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


# the dummy parametrization for the backend is needed to trigger the
# pytest_generate_tests hook in conftest.py, which will skip the test if the
# backend is filtered out by the CLI options.
@fixture(scope="module")
@parametrize_with_cases("mask, shape", cases=CasesCartesian)
@parametrize("backend", ["cartesian"])
def operator(backend, mask, shape):
    """Generate a Cartesian operator."""
    return get_operator("cartesian")(mask, shape)


@_param_array_interface
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
