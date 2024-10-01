"""Data generator module."""

import numpy as np


def get_brainweb_map(sub_id: int) -> np.ndarray:
    """Get MRI parametric maps from a brainweb crisp segmentation.

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
