"""Module containing utility functions for IO in MRI NUFFT."""

import numpy as np
from scipy.spatial.transform import Rotation


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


def siemens_quat_to_rot_mat(quat):
    """
    Calculate the rotation matrix from Siemens Twix quaternion.

    Parameters
    ----------
    quat : np.ndarray
        The quaternion from the Siemens Twix file.

    Returns
    -------
    np.ndarray
        The affine rotation matrix which is a 4x4 matrix.
        This can be passed as input to `affine` parameter in `nibabel`.
    """
    R = np.zeros((4, 4))
    R[:3, :3] = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    R[:, (0, 1)] = R[:, (1, 0)]
    if np.linalg.det(R[:3, :3]) < 0:
        R[2] = -R[2]
    R[-1, -1] = 1
    return R


def remove_extra_kspace_samples(kspace_data, num_samples_per_shot):
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
