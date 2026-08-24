"""Siemens specific rawdat reader, wrapper over turbotwix."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from typing import TypedDict, overload, TYPE_CHECKING, Literal
from typing_extensions import NotRequired  # backport for Python < 3.11

if TYPE_CHECKING:
    from turbotwix import Measurement as TwixObj


class TwixHeaderDict(TypedDict, total=False):
    """Header information extracted from a Siemens Twix object."""

    n_coils: int
    n_shots: int
    n_contrasts: int
    n_adc_samples: int
    n_slices: int
    n_average: int
    n_reps: int
    dwell_time: float  # in seconds
    orientation: NDArray
    affine: NDArray
    shifts: tuple[float, ...]
    acs: NDArray | None
    noise: NDArray | None
    type: NotRequired[str]
    oversampling_factor: NotRequired[int]
    trajectory_name: NotRequired[str]
    TE: NotRequired[list[float]]
    TR: NotRequired[list[float]]
    TD: NotRequired[list[float]]
    TI: NotRequired[list[float]]
    FlipAngleDegree: NotRequired[list[float]]


def _remove_oversampling(data: NDArray, axis: int) -> NDArray:
    """Remove the standard 2x readout oversampling along `axis`.

    Matches the convention used by mapVBVD/twixtools: an ifft to image space along the
    readout, keep the central half of the field of view, fft back to k-space.
    """
    ncol = data.shape[axis]
    keep = np.concatenate([np.arange(ncol // 4), np.arange(ncol * 3 // 4, ncol)])
    image = np.fft.ifft(data, axis=axis)
    image = np.take(image, keep, axis=axis)
    return np.fft.fft(image, axis=axis)


def _slice_position_shifts(twixObj: TwixObj) -> tuple[float, float, float]:
    """Get the slice/volume position offset from ``sSliceArray``.

    ``sSliceArray`` is the standard MrProt field for slice/volume position
    and is populated identically across sequences, unlike the raw
    ``slicePos`` mdh field which can be sequence-dependent.
    """
    return tuple(
        twixObj.hdr.Phoenix.get(("sSliceArray", "asSlice", 0, "sPosition", ax), 0.0)
        for ax in ("dSag", "dCor", "dTra")
    )


def _parse_twix_header(twixObj: TwixObj) -> TwixHeaderDict:
    """Parse the header of a Siemens Twix measurement."""
    image_lines = twixObj.lines.image
    slice_data = image_lines[0:1].headers()[0]["SliceData"]
    quat = np.asarray(slice_data["Quaternion"])
    ph = twixObj.hdr.Phoenix

    hdr: TwixHeaderDict = {
        "n_coils": image_lines.NCha,
        "n_shots": image_lines.NLin * image_lines.NPar,
        "n_contrasts": image_lines.NSet,
        "n_adc_samples": image_lines.NCol,
        "n_slices": image_lines.NSli,
        "n_average": image_lines.NAve,
        "n_reps": image_lines.NRep,
        "orientation": _siemens_quat_to_rot_mat(quat, False),
        "affine": twix2nifti_affine(twixObj),
        "shifts": _slice_position_shifts(twixObj),
        "acs": None,
        "noise": None,
        "dwell_time": float(ph.get(("sRXSPEC", "alDwellTime", 0), 0))
        * 1e-9,  # convert from ns to s
    }

    for key in ["alTR", "alTE", "alTD", "alTI", "adFlipAngleDegree"]:
        # get a list of all sequence times in the sequence
        vals = ph.get(key)
        if vals is None:
            continue  # don't populate if not found.
        nice_key = key[2:]  # strip prefix "al" / "ad"
        if not isinstance(vals, list):
            vals = [vals]
        hdr[nice_key] = vals[0] if len(vals) == 1 else vals  # type: ignore

    refscan = twixObj.lines.refscan
    if len(refscan) > 0:
        hdr["acs"] = refscan.read()
    noise = twixObj.lines.noise
    if len(noise) > 0:
        hdr["noise"] = noise.read()

    return hdr


@overload
def read_siemens_rawdat(
    filename: str,
    removeOS: bool = False,
    doAverage: bool = True,
    squeeze: bool = True,
    reshape: bool = False,
    return_twix: Literal[True] = True,
    slice_num: int | None = None,
    contrast_num: int | None = None,
) -> tuple[NDArray, TwixHeaderDict, TwixObj]: ...


@overload
def read_siemens_rawdat(
    filename: str,
    removeOS: bool = False,
    doAverage: bool = True,
    squeeze: bool = True,
    reshape: bool = False,
    return_twix: Literal[False] = False,
    slice_num: int | None = None,
    contrast_num: int | None = None,
) -> tuple[NDArray, TwixHeaderDict]: ...


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
        Whether to return the twix measurement, by default False.
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
        If the turbotwix module is not available.

    Notes
    -----
    This function requires the turbotwix module to be installed.
    """
    try:
        import turbotwix as tw
    except ImportError as err:
        raise ImportError(
            "The turbotwix module is not available. Please install "
            "it along with the [extra] dependencies."
        ) from err
    twixObj = tw.open_twix(filename).scan
    hdr = _parse_twix_header(twixObj)
    if slice_num is not None and hdr["n_slices"] < slice_num:
        raise ValueError("The slice number is out of bounds.")
    if contrast_num is not None and hdr["n_contrasts"] < contrast_num:
        raise ValueError("The contrast number is out of bounds.")

    lines = twixObj.lines.image
    if slice_num is not None:
        lines = lines[lines.counter("Sli") == slice_num]
    if contrast_num is not None:
        lines = lines[lines.counter("Set") == contrast_num]

    if reshape:
        # Fold onto (Cha, Lin, Par, Sli, Rep, Set, Ave, Col), then merge the
        # adjacent Lin/Par axes into a single shots axis.
        dims = ("Lin", "Par", "Sli", "Rep", "Set", "Ave")
        raw = twixObj.read(lines, dims=dims)
        if removeOS:
            raw = _remove_oversampling(raw, axis=-1)
        raw = np.moveaxis(raw, -1, 3)  # (Cha, Lin, Par, Col, Sli, Rep, Set, Ave)
        ncha, nlin, npar, ncol, nsli, nrep, nset, nave = raw.shape
        data = raw.reshape(ncha, nlin * npar, ncol, nsli, nrep, nset, nave)
        if doAverage:
            data = data.mean(axis=-1, keepdims=True)
        if squeeze:
            data = np.squeeze(data)
    else:
        dims = ("Lin", "Par", "Sli", "Ave", "Eco", "Phs", "Rep", "Set")
        raw = twixObj.read(lines, dims=dims)
        if removeOS:
            raw = _remove_oversampling(raw, axis=-1)
        if doAverage:
            raw = raw.mean(axis=dims.index("Ave") + 1, keepdims=True)
        if squeeze:
            raw = np.squeeze(raw)
        # Cartesian data, format as coils x readout_samples x paritions_y x partitions_z
        data = np.moveaxis(raw, -1, 1)
    if return_twix:
        return data, hdr, twixObj
    return data, hdr


@overload
def _siemens_quat_to_rot_mat(
    quat: NDArray, return_det: Literal[True]
) -> tuple[NDArray, float]: ...


@overload
def _siemens_quat_to_rot_mat(
    quat: NDArray, return_det: Literal[False] = False
) -> NDArray: ...


def _siemens_quat_to_rot_mat(
    quat: NDArray, return_det=False
) -> NDArray | tuple[NDArray, float]:
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

    See Also
    --------
    twix2nifti_affine
        Calculate the affine transformation matrix from Siemens Twix object.
        This is the preferred method to get the affine matrix from a Siemens Twix file.

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


def twix2nifti_affine(twixObj: TwixObj) -> NDArray:
    """
    Calculate the affine transformation matrix from Siemens Twix measurement.

    Parameters
    ----------
    twixObj : TwixObj
        The turbotwix measurement.

    Returns
    -------
    np.ndarray
        The affine transformation matrix which is a 4x4 matrix.
        This can be passed as input to `affine` parameter in `nibabel`.

    See Also
    --------
    siemens_quat_to_rot_mat
        Calculate the rotation matrix from Siemens Twix quaternion.
    read_siemens_rawdat
        Read raw data from a Siemens MRI file,
        use return_twix=True to get the twix measurement.
    read_arbgrad_rawdat
        Read raw data from a Siemens MRI file from neurospin,
        use return_twix=True to get the twix measurement.
    """
    # required keys
    keys = {
        "dthick": ("sSliceArray", "asSlice", 0, "dThickness"),
        "dread": ("sSliceArray", "asSlice", 0, "dReadoutFOV"),
        "dphase": ("sSliceArray", "asSlice", 0, "dPhaseFOV"),
        "lbase": ("sKSpace", "lBaseResolution"),
        "lphase": ("sKSpace", "lPhaseEncodingLines"),
        "ucdim": ("sKSpace", "ucDimension"),
    }
    sos = ("sKSpace", "dSliceOversamplingForDialog")
    image_lines = twixObj.lines.image
    slice_data = image_lines[0:1].headers()[0]["SliceData"]
    quat = np.asarray(slice_data["Quaternion"])
    rot, det = _siemens_quat_to_rot_mat(quat, True)
    my = twixObj.hdr.MeasYaps

    values = {k: my.get(path) for k, path in keys.items()}
    if any(v is None for v in values.values()):
        return rot

    dthick = values["dthick"]
    sos_val = my.get(sos)
    fov = np.array(
        [
            values["dread"],
            values["dphase"],
            dthick * (1 + sos_val if sos_val is not None else 1),
        ]
    )

    lpart = ("sKSpace", "lPartitions")
    lpart_val = my.get(lpart)
    res = np.array(
        [
            values["lbase"],
            values["lphase"],
            lpart_val if values["ucdim"] == 4 and lpart_val is not None else 1,
        ]
    )

    scale = np.diag([*(fov / res), 1])

    slice_pos = slice_data["SlicePos"]
    offset = np.array([slice_pos[ax] for ax in ("Sag", "Cor", "Tra")])

    fovz = fov[2] - (sos_val * dthick if sos_val is not None else 0)
    center = [-fov[0] / 2, -fov[1] / 2, -fovz / 2, 1]

    t = (rot @ center)[:3] - offset
    if det < 0:
        t[2] = (rot @ center)[2] * 2 - t[2]

    full_mat = rot @ scale
    full_mat[:3, 3] = t

    return full_mat
