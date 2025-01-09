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


def siemens_quat_to_rot_mat(quat):
    """
    Calculate the rotation matrix from Siemens Twix quaternion.
    """
    a = quat[1]
    b = quat[2]
    c = quat[3]
    d = quat[0]

    R = np.zeros((4, 4))
    
    R[0,1] = 1.0 - 2.0 * (b * b + c * c)
    R[0,0] = 2.0 * (a * b - c * d)
    R[0,2] = 2.0 * (a * c + b * d)

    R[1,1] = 2.0 * (a * b + c * d)
    R[1,0] = 1.0 - 2.0 * (a * a + c * c)
    R[1,2] = 2.0 * (b * c - a * d)

    R[2,1] = 2.0 * (a * c - b * d)
    R[2,0] = 2.0 * (b * c + a * d)
    R[2,2] = 1.0 - 2.0 * (a * a + b * b)

    if (np.linalg.det(R[:3, :3]) < 0):
        R[2] = -R[2]
        
    R[-1,-1] = 1

    return R