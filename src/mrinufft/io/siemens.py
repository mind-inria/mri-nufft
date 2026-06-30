"""Siemens specific rawdat reader, wrapper over pymapVBVD."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def _parse_twix_header(twixObj):
    """Parse the header of a Siemens Twix object."""
    hdr = {
        "n_coils": int(twixObj.image.NCha),
        "n_shots": int(twixObj.image.NLin) * int(twixObj.image.NPar),
        "n_contrasts": int(twixObj.image.NSet),
        "n_adc_samples": int(twixObj.image.NCol),
        "n_slices": int(twixObj.image.NSli),
        "n_average": int(twixObj.image.NAve),
        "n_reps": int(twixObj.image.NRep),
        "orientation": _siemens_quat_to_rot_mat(twixObj.image.slicePos[0][-4:]),
        "affine": twix2nifti_affine(twixObj),
        "shifts": twixObj.image.slicePos[0][:3][::-1],
        "acs": None,
    }

    for key in ["alTR", "alTE", "alTD", "alTI", "adFlipAngleDegree"]:
        # get a list of all sequences times in the sequence
        vals = twixObj.search_header_for_val("Phoenix", (f"{key}",))
        nice_key = key[2:]  # strip prefix "al /ad"
        if len(vals) == 1:
            hdr[nice_key] = vals[0]
        elif len(vals) > 0:
            # the first element found is the length of the list, we dicard it.
            if vals[0] == len(vals[1:]):
                vals = vals[1:]
            hdr[nice_key] = vals
        # don't populate if not found.

    if "refscan" in twixObj.keys():
        twixObj.refscan.squeeze = True
        acs = twixObj.refscan[""].astype(np.complex64)
        hdr["acs"] = acs.swapaxes(0, 1)

    return hdr


def read_siemens_rawdat(
    filename: str,
    removeOS: bool = False,
    doAverage: bool = True,
    squeeze: bool = True,
    reshape: bool = False,
    return_twix: bool = False,
    slice_num: int | None = None,
    contrast_num: int | None = None,
):  # pragma: no cover
    """Read raw data from a Siemens MRI file.

    Parameters
    ----------
    filename : str
        The path to the Siemens MRI file.
    removeOS : bool, optional
        Whether to remove the oversampling, by default False.
    doAverage : bool, option
        Whether to average the data acquired along NAve dimension.
    squeeze : bool, optional
        Whether to squeeze the dimensions of the data, by default True.
    reshape : bool, optional
        Whether to reshape the data into a
        Nc X Nsamples X Nslices X Ncontrasts format,
        by default False.
    data_type : str, optional
        The type of data to read, by default 'ARBGRAD_VE11C'.
    return_twix : bool, optional
        Whether to return the twix object, by default False.
    slice_num : int, optional
        The slice to read, by default None. This applies for 2D data.
    contrast_num: int, optional
        The contrast to read, by default None.

    Returns
    -------
    data: ndarray
        Imported data formatted as n_coils X n_samples X n_slices X n_contrasts
    hdr: dict
        Extra information about the data parsed from the twix file
        This header also contains the ACS data as "acs" if it was found in raw data.

    Raises
    ------
    ImportError
        If the mapVBVD module is not available.

    Notes
    -----
    This function requires the mapVBVD module to be installed.
    You can install it using the following command::

        pip install pymapVBVD
    """
    try:
        from mapvbvd import mapVBVD
    except ImportError as err:
        raise ImportError(
            "The mapVBVD module is not available. Please install "
            "it along with the [extra] dependencies "
            "or using `pip install pymapVBVD`."
        ) from err
    twixObj = mapVBVD(filename)
    if isinstance(twixObj, list):
        twixObj = twixObj[-1]
    twixObj.image.flagRemoveOS = removeOS
    twixObj.image.flagDoAverage = doAverage
    hdr = _parse_twix_header(twixObj)
    # Add sequence information
    if slice_num is not None and hdr["n_slices"] < slice_num:
        raise ValueError("The slice number is out of bounds.")
    if contrast_num is not None and hdr["n_contrasts"] < contrast_num:
        raise ValueError("The contrast number is out of bounds.")
    # Shape : NCol X NCha X NLin X NAve X NSli X NPar X ..., NSet
    if slice_num is not None and contrast_num is not None:
        raw_kspace = twixObj.image[
            (slice(None),) * 4 + (slice_num,) + (slice(None),) * 4 + (contrast_num,)
        ]
    elif slice_num is not None:
        raw_kspace = twixObj.image[(slice(None),) * 4 + (slice_num,)]
    elif contrast_num is not None:
        raw_kspace = twixObj.image[(slice(None),) * 9 + (contrast_num,)]
    else:
        raw_kspace = twixObj.image[""]
    if squeeze:
        raw_kspace = np.squeeze(raw_kspace)
    if reshape:
        # Format as coils x shots x samples x slices x contrasts x averages
        data = np.moveaxis(raw_kspace, 0, 2)
        data = data.reshape(
            hdr["n_coils"],
            hdr["n_shots"],
            hdr["n_adc_samples"],
            hdr["n_slices"] if slice_num is None else 1,
            hdr["n_reps"],
            hdr["n_contrasts"] if contrast_num is None else 1,
            hdr["n_average"] if hdr["n_average"] > 1 and not doAverage else 1,
        )
    else:
        # Cartesian data, format as coils x readout_samples x paritions_y x partitions_z
        data = np.moveaxis(raw_kspace, 1, 0)
    if return_twix:
        return data, hdr, twixObj
    return data, hdr


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


def twix2nifti_affine(twixObj):
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
