"""Prime-related functions."""

import numpy as np


def compute_coprime_factors(
    Nc: int, length: int, start: int = 1, update: int = 1
) -> list[int]:
    """Compute a list of coprime factors of Nc.

    Parameters
    ----------
    Nc : int
        Number to factorize.
    length : int
        Number of coprime factors to compute.
    start : int, optional
        First number to check. The default is 1.
    update : int, optional
        Increment between two numbers to check. The default is 1.

    Returns
    -------
    list[int]
        List of coprime factors of Nc.
    """
    count = start
    coprimes: list[int] = []
    while len(coprimes) < length:
        # Check greatest common divider (gcd)
        if np.gcd(Nc, count) == 1:
            coprimes.append(count)
        count += update
    return coprimes
