"""Siemens specific rawdat reader, wrapper over pymapVBVD."""

from __future__ import annotations

import numpy as np
from .utils import siemens_quat_to_rot_mat


def read_siemens_rawdat(
    filename: str,
    removeOS: bool = False,
    doAverage: bool = True,
    squeeze: bool = True,
    return_twix: bool = True,
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
    data_type : str, optional
        The type of data to read, by default 'ARBGRAD_VE11C'.
    return_twix : bool, optional
        Whether to return the twix object, by default True.
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
    You can install it using the following command:
        `pip install pymapVBVD`
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
    hdr = {
        "n_coils": int(twixObj.image.NCha),
        "n_shots": int(twixObj.image.NLin) * int(twixObj.image.NPar),
        "n_contrasts": int(twixObj.image.NSet),
        "n_adc_samples": int(twixObj.image.NCol),
        "n_slices": int(twixObj.image.NSli),
        "n_average": int(twixObj.image.NAve),
        "orientation": siemens_quat_to_rot_mat(twixObj.image.slicePos[0][-4:]),
        "acs": None,
    }
    if "refscan" in twixObj.keys():
        twixObj.refscan.squeeze = True
        acs = twixObj.refscan[""].astype(np.complex64)
        hdr["acs"] = acs.swapaxes(0, 1)
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
    data = np.moveaxis(raw_kspace, 0, 2)

    data = data.reshape(
        hdr["n_coils"],
        hdr["n_shots"],
        hdr["n_adc_samples"],
        hdr["n_slices"] if slice_num is None else 1,
        hdr["n_contrasts"] if contrast_num is None else 1,
        hdr["n_average"] if hdr["n_average"] > 1 and not doAverage else 1,
    )
    if return_twix:
        return data, hdr, twixObj
    return data, hdr
