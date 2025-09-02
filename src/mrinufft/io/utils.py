"""Module containing utility functions for IO in MRI NUFFT."""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from ..trajectories.utils import (
    convert_trajectory_to_gradients,
    Gammas,
    DEFAULT_SMAX,
    DEFAULT_GMAX,
    DEFAULT_RASTER_TIME,
    KMAX,
)
from ..trajectories.tools import get_gradient_amplitudes_to_travel_for_set_time


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


def siemens_quat_to_rot_mat(quat: NDArray, return_det=False):
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


def prepare_trajectory_for_seq(
    trajectory: NDArray,
    fov: tuple[float, float, float],
    img_size: tuple[int, int, int],
    norm_factor: float = KMAX,
    pregrad: str = "prephase",
    postgrad: str = "slowdown_to_edge",
    gamma: float = Gammas.HYDROGEN,
    raster_time: float = DEFAULT_RASTER_TIME,
    gmax: float = DEFAULT_GMAX,
    smax: float = DEFAULT_SMAX,
):
    """Prepare gradients from trajectory.

    This function converts a k-space trajectory into full gradients with pre-
    and post-gradients.


    Parameters
    ----------
    trajectory : np.ndarray
        The k-space trajectory as a numpy array of shape (n_shots, n_samples, 3),
        where the last dimension corresponds to the x, y, and z coordinates in k-space.
    norm_factor : float
        The normalization factor for the trajectory. (default is 0.5)
    fov : tuple[float, float, float]
        The field of view in the x, y, and z dimensions, in meters.
    img_size : tuple[int, int, int]
        The image size in the x, y, and z dimensions, in pixels.
    pregrad : str, optional
        The type of pre-gradient to apply. Only "prephase" is supported currently.
    postgrad : str, optional
        The type of post-gradient to apply. Defaults to "slowdown_to_edge".

    Returns
    -------
    np.ndarray
        The full gradients as a numpy array of shape (n_shots, n_samples, 3),
        where the last dimension corresponds to the x, y, and z gradient amplitudes.
    int
        The number of samples to skip at the start of the trajectory.
    int
        The number of samples to skip at the end of the trajectory.


    See Also
    --------
    mrinufft.io.pulseq.pulseq_gre_3D: to create a Pulseq 3D-GRE sequence
    with arbitrary gradient waveform designed

    """
    # from #276 : We need to prewind the gradients to the first point of the
    # trajectory, and rewind them to the edge of k-space.

    # We will move from
    # init_pos -[prewind]-> start_pos -> trajectory -> end_pos -[postgrad]-> final_pos

    grads, start_pos, end_pos = convert_trajectory_to_gradients(
        trajectory,
        norm_factor=norm_factor,
        resolution=fov,
        raster_time=raster_time,
        gamma=gamma,
        get_final_positions=True,
    )

    # prewind the gradients to their first point:
    if pregrad == "prephase":
        init_pos = np.zeros_like(start_pos)
    else:
        raise ValueError("Only 'prephase' is supported for pregrad.")
    start_grads = get_gradient_amplitudes_to_travel_for_set_time(
        kspace_end_loc=start_pos,
        kspace_start_loc=init_pos,
        end_gradients=grads[:, 0, :],
        gamma=gamma,
        raster_time=raster_time,
        gmax=gmax,
        smax=smax,
    )
    skip_start = start_grads.shape[1]

    final_pos = np.zeros_like(end_pos)
    if postgrad == "slowdown_to_edge":
        # Set the edge location to [Kmax, 0,0], to prepare for gradient spoiling.
        final_pos[..., 0] = img_size[0] * fov[0] / 2
    else:
        raise ValueError("Only 'slowdown_to_edge' is supported for postgrad.")

    end_grads = get_gradient_amplitudes_to_travel_for_set_time(
        kspace_start_loc=end_pos,
        kspace_end_loc=final_pos,
        start_gradients=grads[:, -1, :],
        gamma=gamma,
        raster_time=raster_time,
        gmax=gmax,
        smax=smax,
    )

    skip_end = end_grads.shape[1]
    full_gradients = np.hstack([start_grads, grads, end_grads])

    return full_gradients, skip_start, skip_end
