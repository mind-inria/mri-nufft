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


def siemens_quat_to_rot_mat(quat, return_det=False):
    """
    Calculate the rotation matrix from Siemens Twix quaternion.

    Parameters
    ----------
    quat : np.ndarray
        The quaternion from the Siemens Twix file.
    return_det : bool
        Whether to return the determinent of the rotation before norm

    Returns
    -------
    np.ndarray
        The affine rotation matrix which is a 4x4 matrix.
        This can be passed as input to `affine` parameter in `nibabel`.
    """
    R = np.zeros((4, 4))
    R[:3, :3] = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    R[:, (0, 1)] = R[:, (1, 0)]
    det = np.linalg.det(R[:3, :3])
    if det < 0:
        R[2] = -R[2]
    R[-1, -1] = 1
    if return_det:
        return R, det
    return R


def nifti_affine(twixObj):
    """
    Calculate the affine transformation matrix from Siemens Twix object.

    Parameters
    ----------
    twixObj : twixObj
        The twix object returned by mapVBVD.

    Returns
    -------
    np.ndarray
        The affine transformation matrix which is a 4x4 matrix.
        This can be passed as input to `affine` parameter in `nibabel`.
    """
    # required keys
    keys = {
        "dthick": ("sSliceArray", "asSlice", "0", "dThickness"),
        "dread": ("sSliceArray", "asSlice", "0", "dReadoutFOV"),
        "dphase": ("sSliceArray", "asSlice", "0", "dPhaseFOV"),
        "lbase": ("sKSpace", "lBaseResolution"),
        "lphase": ("sKSpace", "lPhaseEncodingLines"),
        "ucdim": ("sKSpace", "ucDimension"),
    }
    sos = ("sKSpace", "dSliceOversamplingForDialog")
    rot, det = siemens_quat_to_rot_mat(twixObj.image.slicePos[0][-4:], True)
    my = twixObj.hdr.MeasYaps

    for k in keys.keys():
        if keys[k] not in my:
            return rot

    dthick = my[keys["dthick"]]
    fov = np.array(
        [
            my[keys["dread"]],
            my[keys["dphase"]],
            dthick * (1 + my[sos] if sos in my else 1),
        ]
    )

    lpart = ("sKSpace", "lPartitions")
    res = np.array(
        [
            my[keys["lbase"]],
            my[keys["lphase"]],
            my[lpart] if my[keys["ucdim"]] == 4 and lpart in my else 1,
        ]
    )

    scale = np.diag([*(fov / res), 1])

    offset = twixObj.image.slicePos[0][:3]

    fovz = fov[2] - (my[sos] * dthick if sos in my else 0)
    center = [-fov[0] / 2, -fov[1] / 2, -fovz / 2, 1]

    t = (rot @ center)[:3] - offset
    if det < 0:
        t[2] = (rot @ center)[2] * 2 - t[2]

    full_mat = rot @ scale
    full_mat[:3, 3] = t

    return full_mat


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
