"""Data generator module."""

import numpy as np

from collections.abc import Sequence


def get_brainweb_map(sub_id: int) -> np.ndarray:
    """
    Get M0, T1 and T2 parametric maps from a brainweb crisp segmentation.

    Output maps have the same shape as the tissue segmentation.

    Parameters
    ----------
    sub_id : int
        Subject ID.

    Raises
    ------
    ImportError
        If brainweb-dl is not installed.

    Returns
    -------
    M0 : np.ndarray
        Proton Density map. For sub_id > 0,
        it is a binary mask.
    T1 : np.ndarray
        T1 map map in [ms].
    T2 : np.ndarray
        T2 map map in [ms].

    """
    try:
        import brainweb_dl
    except ImportError as err:
        raise ImportError(
            "The brainweb-dl module is not available. Please install it using "
            "the following command: pip install brainweb-dl"
        ) from err

    # get segmentation
    segmentation = brainweb_dl.get_mri(sub_id, "crisp") / 455
    segmentation = segmentation.astype(int)

    # get properties
    model = brainweb_dl._brainweb.BrainWebTissueMap
    if sub_id == 0:
        properties = brainweb_dl._brainweb._load_tissue_map(model.v1)
    else:
        properties = brainweb_dl._brainweb._load_tissue_map(model.v2)

    # initialize maps
    if sub_id == 0:
        M0 = np.zeros_like(segmentation, dtype=np.float32)
    T1 = np.zeros_like(segmentation, dtype=np.float32)
    T2 = np.zeros_like(segmentation, dtype=np.float32)

    # fill maps
    for tissue in properties:
        idx = segmentation == int(tissue["Label"])
        if sub_id == 0:
            M0 += float(tissue["PD (ms)"]) * idx
        T1 += float(tissue["T1 (ms)"]) * idx
        T2 += float(tissue["T2 (ms)"]) * idx

    if sub_id != 0:
        M0 = (segmentation != 0).astype(np.float32)

    # pad to square
    pad_width = segmentation.shape[1] - segmentation.shape[2]
    pad = ((0, 0), (0, 0), (int(pad_width // 2), int(pad_width // 2)))
    M0 = np.pad(M0, pad)
    T1 = np.pad(T1, pad)
    T2 = np.pad(T2, pad)

    return M0, T1, T2


def fse_simulation(
    M0: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    TE: float | Sequence[float],
    TR: float | Sequence[float],
) -> np.ndarray:
    """Perform simple analytical Fast Spin Echo simulation.

    Assume that refocusing angles are 180° and
    k-space center is sampled for each echo in the Echo Train
    (e.g., as in spiral or radial imaging).

    Parameters
    ----------
    M0 : np.ndarray
        Input equilibrium magnetization.
    T1 : np.ndarray
        Input T1 in [ms].
    T2 : np.ndarray
        Input T2 in [ms].
    TE : float | Sequence[float]
        Sequence Echo Time in [ms].
    TR : float | Sequence[float]
        Sequence Repetition Time in [ms].

    Returns
    -------
    signal : np.ndarray
        Simulated signal of shape ``(nTE*nTR, *M0)``.

    """
    # preprocess sequence parameters
    TE, TR = np.broadcast_arrays(np.atleast_1d(TE), np.atleast_1d(TR)[:, None])
    TE, TR = TE.ravel().astype(np.float32), TR.ravel().astype(np.float32)

    # preprocess tissue parameters
    M0, T1, T2 = np.atleast_1d(M0), np.atleast_1d(T1), np.atleast_1d(T2)
    M0, T1, T2 = M0[..., None], T1[..., None], T2[..., None]
    T1 += 1e-9
    T2 += 1e-9

    # compute signal
    signal = M0 * (1 - np.exp(-(TR - TE) / T1)) * np.exp(-TE / T2)

    # post process
    signal = signal[None, ...].swapaxes(0, -1)[..., 0]
    signal = signal.squeeze()
    if signal.size == 1:
        signal = signal.item()

    return signal
