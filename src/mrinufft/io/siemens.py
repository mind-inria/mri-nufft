import numpy as np

try:
    from mapvbvd import mapVBVD
    MAPVBVD_FOUND = True
except ImportError:
    MAPVBVD_FOUND = False


def read_rawdat(filename: str, removeOS: bool = False, squeeze: bool = True,
                data_type: str = "SPARKLING_VE11C"):
    """Read raw data from a Siemens MRI file.
    
    Parameters
    ----------
    filename : str
        The path to the Siemens MRI file.
    removeOS : bool, optional
        Whether to remove the oversampling, by default False.
    squeeze : bool, optional
        Whether to squeeze the dimensions of the data, by default True.
    data_type : str, optional
        The type of data to read, by default 'SPARKLING_VE11C'.

    Returns
    -------
    data: ndarray
        Imported data formatted as n_coils X n_samples X n_slices X n_contrasts
    hdr: dict
        Extra information about the data parsed from the twix file

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
    if not MAPVBVD_FOUND:
        raise ImportError(
            "The mapVBVD module is not available. Please install it using "
            "the following command: pip install pymapVBVD"
        )
    twixObj = mapVBVD(filename)
    if isinstance(twixObj, list):
        twixObj = twixObj[-1]
    twixObj.image.flagRemoveOS = removeOS
    twixObj.image.squeeze = squeeze
    raw_kspace = twixObj.image['']
    data = np.moveaxis(raw_kspace, 0, 2)
    hdr = {
        "n_coils": int(twixObj.image.NCha),
        "n_shots": int(twixObj.image.NLin),
        "n_contrasts": int(twixObj.image.NSet),
        "n_adc_samples": int(twixObj.image.NCol),
        "n_slices": int(twixObj.image.NSli),
    }
    data = data.reshape(
        hdr["n_coils"], 
        hdr["n_shots"]*hdr["n_adc_samples"], 
        hdr["n_slices"], 
        hdr["n_contrasts"]
    )
    if "ARBGRAD_VE11C" in data_type:
        hdr["type"] = "ARBGRAD_GRE"
        hdr["shifts"] = ()
        for s in [7, 6, 8]:
            shift = twixObj.search_header_for_val(
                "Phoenix", ("sWiPMemBlock", "adFree", str(s))
            )
            hdr["shifts"] += (0,) if shift == [] else (shift[0],)
        hdr["oversampling_factor"] = twixObj.search_header_for_val(
            "Phoenix", ("sWiPMemBlock", "alFree", "4")
        )[0]
        hdr["trajectory_name"] = twixObj.search_header_for_val(
            "Phoenix", ("sWipMemBlock", "tFree")
        )[0][1:-1]
        if(hdr["num_contrasts"] > 1):
            hdr["turboFactor"] = twixObj.search_header_for_val(
                "Phoenix", ("sFastImaging", "lTurboFactor")
            )[0]
            hdr["type"] = "ARBGRAD_MP2RAGE"
    return data, hdr