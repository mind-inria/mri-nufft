"""Module containing utility functions for IO in MRI NUFFT."""

import numpy as np
from numpy.typing import NDArray


def add_phase_to_kspace_with_shifts(
    kspace_data: NDArray, kspace_loc: NDArray, normalized_shifts: NDArray
):
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
        They are expressed as a number of pixels to shifts.

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
    if len(normalized_shifts) != kspace_loc.shape[-1]:
        raise ValueError(
            "Dimension mismatch between shift and kspace locations! "
            "Ensure that shifts are right"
        )
    # Fold the kspace data to match the number of dimension in kspace_loc
    # This is useful when the kspace_loc has a shot dimension and the kspace_data
    # does not. For instance kspace_data has shape (NCha, NSamples) and kspace_loc
    # has shape (NShot, NSamples//Nshot, 2). In this case, we need to fold the
    # kspace_data to shape (NCha, NShot, NSamples//Nshot) to match the kspace_loc shape.
    if kspace_data.shape[-1] != kspace_loc.shape[-2]:
        kspace_data_ = kspace_data.reshape(
            *kspace_data.shape[:-1], -1, *kspace_loc.shape[:-1]
        )
    else:
        kspace_data_ = kspace_data

    phi = np.sum(kspace_loc * normalized_shifts, axis=-1)
    phase = np.exp(-2 * np.pi * 1j * phi)
    new_kspace_data = kspace_data_ * phase

    # Return the new kspace data with the same shape as the input kspace data
    return new_kspace_data.reshape(kspace_data.shape)


def remove_extra_kspace_samples(kspace_data: NDArray, num_samples_per_shot: int):
    """Remove extra samples from k-space data.

    This function is useful when the k-space data has extra samples
    mainly as ADC samples at only at specific number of samples.
    This sometimes leads to a situation where we will have more ADC samples
    than what is expected.

    Parameters
    ----------
    kspace_data : np.ndarray
        The k-space data ordered as NCha X NShot X NSamples.
    num_samples_per_shot : int
        The number of samples per shot in trajectory

    Returns
    -------
    np.ndarray
        The k-space data with extra samples removed.
    """
    n_samples = kspace_data.shape[-1]
    n_extra_samples = n_samples - num_samples_per_shot
    if n_extra_samples > 0:
        kspace_data = kspace_data[..., :-n_extra_samples]
    return kspace_data


def discard_frequency_outliers(
    kspace_data: NDArray | None, kspace_loc: NDArray, kmax=0.5
):
    """
    Remove samples in kspace_data and kspace_loc if outside [-k_max; k_max[.

    Parameters
    ----------
    kspace_data: numpy.ndarray
        The samples corresponding to kspace_loc defined above.
    kspace_loc: numpy.ndarray
        The sample locations previously normalized around [-k_max; k_max[.
    k_max: float
        The maximum k-space value to keep.

    Returns
    -------
    reduced_kspace_loc: numpy.ndarray
        The sample locations reduced strictly to [-0.5; 0.5[ by discarding
        outliers.
    reduced_kspace_data: numpy.ndarray
        The samples corresponding to reduced_kspace_loc defined above.
    """
    kspace_mask = np.all((kspace_loc < kmax) & (kspace_loc >= -kmax), axis=-1)
    kspace_loc = kspace_loc[kspace_mask]
    if kspace_data is not None:
        kspace_data = kspace_data[..., kspace_mask]
        return np.ascontiguousarray(kspace_loc), np.ascontiguousarray(kspace_data)
    return np.ascontiguousarray(kspace_loc)
