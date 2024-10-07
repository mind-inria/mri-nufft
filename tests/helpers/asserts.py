"""Utils for tests."""

import numpy as np
import numpy.testing as npt
import scipy as sp

from .factories import from_interface


def assert_almost_allclose(a, b, rtol, atol, mismatch, equal_nan=False):
    """Assert allclose with a tolerance on the number of mismatched elements.

    Parameters
    ----------
    a : array_like
        First array to compare.
    b : array_like
        Second array to compare.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.
    mismatch : int or float
        Maximum number of mismatched elements.
    equal_nan : bool
        If True, NaNs will compare equal.

    Raises
    ------
    AssertionError
        If the arrays are not equal up to specified tolerance.
    """
    val = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

    if mismatch < 1:
        mismatch_perc = mismatch
        mismatch = int(mismatch * np.prod(a.shape))
    else:
        mismatch_perc = mismatch / np.prod(a.shape)

    if np.sum(~val) > mismatch:
        try:
            npt.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
        except AssertionError as e:
            message = getattr(e, "message", "")
            message += "\nMismatched elements: "
            message += f"{np.sum(~val)} > {mismatch}(={mismatch_perc*100:.2f}%)"
            raise e


def assert_correlate(a, b, slope=1.0, slope_err=1e-3, r_value_err=1e-3):
    """Assert the correlation between two arrays."""
    slope_reg, intercept, rvalue, stderr, intercept_stderr = sp.stats.linregress(
        a.flatten(), b.flatten()
    )
    abs_slope_reg = abs(slope_reg)
    if r_value_err is not None and abs(rvalue - 1) > r_value_err:
        raise AssertionError(
            f"RValue {rvalue} != 1 ± {r_value_err}\n "
            f"intercept={intercept}, stderr={stderr}, "
            f"intercept_stderr={intercept_stderr}"
        )
    if slope_err is not None and abs(abs_slope_reg - slope) > slope_err:
        raise AssertionError(
            f"Slope {abs_slope_reg} != {slope} ± {slope_err}\n r={rvalue},"
            f"intercept={intercept}, stderr={stderr}, "
            f"intercept_stderr={intercept_stderr}"
        )


def assert_allclose(actual, expected, atol, rtol, interface):
    """Backend agnostic assertion using from_interface helper."""
    actual_np = from_interface(actual, interface)
    expected_np = from_interface(expected, interface)
    npt.assert_allclose(actual_np, expected_np, atol=atol, rtol=rtol)
