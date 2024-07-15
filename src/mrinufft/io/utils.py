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


def _siemens_quat_to_orient_mat(q):
    """
    Calculate the rotation matrix from Siemens Twix quaternion.

    Parameters
    ----------
    q : np.ndarray
        The Siemens TWIX scan quaternion extracted from header.

    Returns
    -------
    np.ndarray
        The orientation matrix.

    """
    ds = 2.0 / np.sum(q**2)
    dxs = q[1] * ds
    dys = q[2] * ds
    dzs = q[3] * ds
    dwx = q[0] * dxs
    dwy = q[0] * dys
    dwz = q[0] * dzs
    dxx = q[1] * dxs
    dxy = q[1] * dys
    dxz = q[1] * dzs
    dyy = q[2] * dys
    dyz = q[2] * dzs
    dzz = q[3] * dzs

    R = np.zeros((4, 4))
    R[0, 0] = 1.0 - (dyy + dzz)
    R[0, 1] = dxy + dwz
    R[0, 2] = dxz - dwy
    R[1, 0] = dxy - dwz
    R[1, 1] = 1.0 - (dxx + dzz)
    R[1, 2] = dyz + dwx
    R[2, 0] = dxz + dwy
    R[2, 1] = dyz - dwx
    R[2, 2] = 1.0 - (dxx + dyy)

    R[-1, -1] = 1

    return R


def get_siemens_twix_orientation_matrix(twix_obj):
    """
    Extract the orientation matrix from Siemens Twix (twixtools) scan object.

    Parameters
    ----------
    twix_obj : dict
        The twix object read by twixtools.

    Returns
    -------
    np.ndarray
        The orientation matrix.
    """
    mdb = twix_obj['image'].mdb_list
    mdh = mdb[0].mdh
    quat = mdh.SliceData.Quaternion
    return _siemens_quat_to_orient_mat(quat)
