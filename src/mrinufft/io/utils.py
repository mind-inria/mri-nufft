"""Module containing utility functions for IO in MRI NUFFT."""

import numpy as np


def add_phase_to_kspace_with_shifts(kspace_data, kspace_loc, normalized_shifts):
    """
    Add phase shifts to k-space data.

    Parameters
    ----------
    kspace_data : np.ndarray
        The k-space data.
    kspace_loc : np.ndarray
        The k-space locations.
    normalized_shifts : tuple
        The normalized shifts to apply to each dimension of k-space.

    Returns
    -------
    ndarray
        The k-space data with phase shifts applied.

    Raises
    ------
    ValueError
        If the dimension of normalized_shifts does not match the number of
        dimensions in kspace_loc.
    """
    if len(normalized_shifts) != kspace_loc.shape[1]:
        raise ValueError(
            "Dimension mismatch between shift and kspace locations! "
            "Ensure that shifts are right"
        )
    phi = np.sum(kspace_loc * normalized_shifts, axis=-1)
    phase = np.exp(-2 * np.pi * 1j * phi)
    return kspace_data * phase
