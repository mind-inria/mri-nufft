"""Utility functions for GPU Interface."""


def check_error(ier, message):  # noqa: D103
    if ier != 0:
        raise RuntimeError(message)


def sizeof_fmt(num, suffix="B"):
    """
    Return a number as a XiB format.

    Parameters
    ----------
    num: int
        The number to format
    suffix: str, default "B"
        The unit suffix

    References
    ----------
    https://stackoverflow.com/a/1094933
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def _next235beven(n, b):
    """Find the next even integer not less than n.

    This function finds the next even integer not less than n, with prime factors no
    larger than 5, and is a multiple of b (where b is a number that only
    has prime factors 2, 3, and 5).
    It is used in particular with `pipe` density compensation estimation.
    """
    if n <= 2:
        return 2
    if n % 2 == 1:
        n += 1  # make it even
    nplus = n - 2  # to cancel out the +=2 at start of loop
    numdiv = 2  # a dummy that is >1
    while numdiv > 1 or nplus % b != 0:
        nplus += 2  # stays even
        numdiv = nplus
        while numdiv % 2 == 0:
            numdiv //= 2  # remove all factors of 2, 3, 5...
        while numdiv % 3 == 0:
            numdiv //= 3
        while numdiv % 5 == 0:
            numdiv //= 5
    return nplus
