"""Utils for tests."""


import numpy as np
import numpy.testing as npt


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
            e.message += "\nMismatched elements: "
            e.message += f"{np.sum(~val)} > {mismatch}(={mismatch_perc*100:.2f}%)"
            raise e
