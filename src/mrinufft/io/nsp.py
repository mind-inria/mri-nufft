# type: ignore
"""Read/Write trajectory for Neurospin sequences."""

from __future__ import annotations

import os
import warnings
from array import array
from datetime import datetime

import numpy as np

from mrinufft.trajectories.utils import (
    Acquisition,
    Hardware,
    Gammas,
    SI,
    check_hardware_constraints,
    convert_gradients_to_slew_rates,
    convert_trajectory_to_gradients,
    DEFAULT_GMAX,
    DEFAULT_SMAX,
    KMAX,
    DEFAULT_RASTER_TIME,
)
from mrinufft.trajectories.tools import get_gradient_amplitudes_to_travel_for_set_time

from .siemens import read_siemens_rawdat


def write_gradients(
    gradients: np.ndarray,
    initial_positions: np.ndarray,
    grad_filename: str,
    img_size: tuple[int, ...],
    FOV: tuple[float, ...],
    TE_pos: float = 0.5,
    min_osf: int = 5,
    gamma: float = Gammas.HYDROGEN,
    version: float = 4.2,
    recon_tag: float = 1.1,
    timestamp: float | None = None,
    keep_txt_file: bool = False,
    final_positions: np.ndarray | None = None,
    start_skip_samples: int = 0,
    end_skip_samples: int = 0,
):
    """Create gradient file from gradients and initial positions.

    Parameters
    ----------
    gradients : np.ndarray
        Gradients. Shape (num_shots, num_samples_per_shot, dimension). in T/m.
    initial_positions : np.ndarray
        Initial positions. Shape (num_shots, dimension).
    grad_filename : str
        Gradient filename.
    img_size : tuple[int, ...]
        Image size.
    FOV : tuple[float, ...]
        Field of view.
    TE_pos : float, optional
        The ratio of trajectory when TE occurs, with 0 as start of
        trajectory and 1 as end. By default 0.5, which is the
        center of the trajectory (in-out trajectory).
    min_osf : int, optional
        Minimum oversampling factor needed at ADC, by default 5
    gamma : float, optional
        Gyromagnetic ratio in kHz/T, by default 42.576e3
    version : float, optional
        Trajectory versioning, by default 4.2
    recon_tag : float, optional
        Reconstruction tag for online recon, by default 1.1
    timestamp : float, optional
        Timestamp of trajectory, by default None
    keep_txt_file : bool, optional
        Whether to keep the text file used temporarily which holds data pushed to
        binary file, by default False
    final_positions : np.ndarray, optional
        Final positions. Shape (num_shots, dimension), by default None
    start_skip_samples : int, optional
        Number of samples to skip in ADC at start of each shot, by default 0
        This works only for version >= 5.1.
    end_skip_samples : int, optional
        Number of samples to skip in ADC at end of each shot, by default 0
        This works only for version >= 5.1.

    """
    num_shots = gradients.shape[0]
    num_samples_per_shot = gradients.shape[1]
    dimension = initial_positions.shape[-1]
    if len(gradients.shape) == 3:
        gradients = gradients.reshape(-1, gradients.shape[-1])
    # Convert gradients to mT/m
    gradients = gradients * 1e3
    max_grad = np.max(np.abs(gradients))
    file = open(grad_filename + ".txt", "w")
    if version >= 4.1:
        file.write(str(version) + "\n")
    # Write the dimension, num_samples_per_shot and num_shots
    file.write(str(dimension) + "\n")
    if version >= 4.1:
        img_size = img_size
        FOV = FOV
        if isinstance(img_size, int):
            img_size = (img_size,) * dimension
        if isinstance(FOV, float):
            FOV = (FOV,) * dimension
        for fov in FOV:
            file.write(str(fov) + "\n")
        for sz in img_size:
            file.write(str(sz) + "\n")
        file.write(str(min_osf) + "\n")
        file.write(str(gamma * 1000) + "\n")
    file.write(str(num_shots) + "\n")
    file.write(str(num_samples_per_shot) + "\n")
    if version >= 4.1:
        if TE_pos == 0:
            if np.sum(initial_positions) != 0:
                warnings.warn(
                    "The initial positions are not all zero for center-out trajectory"
                )
        file.write(str(TE_pos) + "\n")
        # Write the maximum Gradient
        file.write(str(max_grad) + "\n")
        # Write recon Pipeline version tag
        file.write(str(recon_tag) + "\n")
        left_over = 10
        if version >= 4.2:
            # Inset datetime tag
            if timestamp is None:
                timestamp = float(datetime.now().timestamp())
            file.write(str(timestamp) + "\n")
            left_over -= 1
        if version >= 5.1:
            file.write(str(start_skip_samples) + "\n")
            file.write(str(end_skip_samples) + "\n")
            left_over -= 2
        file.write(str("0\n" * left_over))
    # Write all the k0 values
    file.write(
        "\n".join(
            " ".join([f"{iter2:5.4f}" for iter2 in iter1])
            for iter1 in initial_positions
        )
        + "\n"
    )
    if version >= 5:
        if final_positions is None:
            warnings.warn(
                "Final positions not provided for version >= 5,"
                "calculating final positions from gradients"
            )
            final_positions = initial_positions + np.sum(gradients, axis=1)
        file.write(
            "\n".join(
                " ".join([f"{iter2:5.4f}" for iter2 in iter1])
                for iter1 in final_positions
            )
            + "\n"
        )
    if version < 4.1:
        # Write the maximum Gradient
        file.write(str(max_grad) + "\n")
    # Normalize gradients
    gradients = gradients / max_grad
    file.write(
        "\n".join(" ".join([f"{iter2:5.6f}" for iter2 in iter1]) for iter1 in gradients)
        + "\n"
    )
    file.close()
    y = []
    with open(grad_filename + ".txt") as txtfile:
        for line in txtfile:
            x = line.split(" ")
            for val in x:
                y.append(float(val))
    float_array = array("f", y)
    with open(grad_filename + ".bin", "wb") as binfile:
        float_array.tofile(binfile)
    if not keep_txt_file:
        os.remove(grad_filename + ".txt")


def _pop_elements(array, num_elements=1, type=np.float32):
    """Pop elements from an array.

    Parameters
    ----------
    array : np.ndarray
        Array to pop elements from.
    num_elements : int, optional
        number of elements to pop, by default 1
    type : str, optional
        Type of the element being popped, by default 'float'.


    Returns
    -------
    element_popped:
        Element popped from array with type as specified.
    array: np.ndarray
        Array with elements popped.
    """
    if num_elements == 1:
        return array[0].astype(type, copy=False), array[1:]
    else:
        return array[0:num_elements].astype(type, copy=False), array[num_elements:]


def write_trajectory(
    trajectory: np.ndarray,
    FOV: tuple[float, ...],
    img_size: tuple[int, ...],
    grad_filename: str,
    norm_factor: float = KMAX,
    gamma: float = Gammas.HYDROGEN / 1e3,
    raster_time: float = DEFAULT_RASTER_TIME,
    check_constraints: bool = True,
    TE_pos: float = 0.5,
    gmax: float = DEFAULT_GMAX,
    smax: float = DEFAULT_SMAX,
    pregrad: str | None = None,
    postgrad: str | None = None,
    version: float = 5,
    **kwargs,
):
    """Calculate gradients from k-space points and write to file.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory in k-space points.
        Shape (num_shots, num_samples_per_shot, dimension).
    FOV : tuple
        Field of view
    img_size : tuple
        Image size
    grad_filename : str
        Gradient filename
    norm_factor : float, optional
        Trajectory normalization factor, by default 0.5
    gamma : float, optional
        Gyromagnetic ratio in kHz/T, by default 42.576e3
    raster_time : float, optional
        Gradient raster time in ms, by default 0.01
    check_constraints : bool, optional
        Check scanner constraints, by default True
    TE_pos : float, optional
        The ratio of trajectory when TE occurs, with 0 as start of
        trajectory and 1 as end. By default 0.5, which is the
        center of the trajectory (in-out trajectory).
    gmax : float, optional
        Maximum gradient magnitude in T/m, by default 0.04
    smax : float, optional
        Maximum slew rate in T/m/ms, by default 0.1
    pregrad : str, optional
        Pregrad method, by default `prephase`
        `prephase` will add a prephasing gradient to the start of the trajectory.
    postgrad : str, optional
        Postgrad method, by default 'slowdown_to_edge'
        `slowdown_to_edge` will add a gradient to slow down to the edge of the k-space
        along x-axis for all the shots i.e. go to (Kmax, 0, 0).
        This is useful for sequences needing a spoiler at the end of the trajectory.
        However, spoiler is still not added, it is expected that the sequence
        handles the spoilers, which can be variable.
        `slowdown_to_center` will add a gradient to slow down to the center
        of the k-space.
    version: float, optional
        Trajectory versioning, by default 5
    kwargs : dict, optional
        Additional arguments for writing the gradient file.
        These are arguments passed to write_gradients function above.
    """
    # Convert normalized trajectory to gradients
    acq = Acquisition(
        fov=FOV,
        img_size=img_size,
        gamma=gamma,
        norm_factor=norm_factor,
        hardware=Hardware(gmax=gmax, smax=smax, grad_raster_time=raster_time),
    )
    gradients, initial_positions, final_positions = convert_trajectory_to_gradients(
        trajectory,
        acq=acq,
        get_final_positions=True,
    )
    Ns_to_skip_at_start = 0
    Ns_to_skip_at_end = 0
    if pregrad == "prephase":
        if version < 5.1:
            raise ValueError(
                "pregrad is only supported for version >= 5.1, "
                "please set version to 5.1 or higher."
            )
        start_gradients = get_gradient_amplitudes_to_travel_for_set_time(
            kspace_end_loc=initial_positions,
            end_gradients=gradients[:, 0],
            acq=acq,
        )
        initial_positions = np.zeros_like(initial_positions)
        gradients = np.hstack([start_gradients, gradients])
        Ns_to_skip_at_start = start_gradients.shape[1]
    if postgrad:
        if version < 5.1:
            raise ValueError(
                "postgrad is only supported for version >= 5.1, "
                "please set version to 5.1 or higher."
            )
        edge_locations = np.zeros_like(final_positions)
        if postgrad == "slowdown_to_edge":
            # Always end at KMax, the spoilers can be handeled by the sequence.
            edge_locations[..., 0] = img_size[0] / FOV[0] / 2
        end_gradients = get_gradient_amplitudes_to_travel_for_set_time(
            kspace_end_loc=edge_locations,
            start_gradients=gradients[:, -1],
            kspace_start_loc=final_positions,
            acq=acq,
        )
        gradients = np.hstack([gradients, end_gradients])
        Ns_to_skip_at_end = end_gradients.shape[1]
    # Check constraints if requested
    if check_constraints:
        slewrates, _ = convert_gradients_to_slew_rates(gradients, acq)
        valid, maxG, maxS = check_hardware_constraints(
            gradients=gradients,
            slewrates=slewrates,
            acq=acq,
        )
        if not valid:
            warnings.warn(
                "Hard constraints violated! "
                f"Maximum gradient amplitude: {maxG:.3f} > {gmax:.3f}"
                f"Maximum slew rate: {maxS:.3f} > {smax:.3f}"
            )
        if pregrad != "prephase":
            border_slew_rate = gradients[:, 0] / raster_time
            if np.any(np.abs(border_slew_rate) > smax):
                warnings.warn(
                    "Slew rate at start of trajectory exceeds maximum slew rate!"
                    f"Maximum slew rate: {np.max(np.abs(border_slew_rate)):.3f}"
                    f" > {smax:.3f}. Please use prephase gradient to avoid this "
                    " issue."
                )

    # Write gradients in file
    write_gradients(
        gradients=gradients,
        initial_positions=initial_positions,
        final_positions=final_positions,
        grad_filename=grad_filename,
        img_size=img_size,
        FOV=FOV,
        TE_pos=TE_pos,
        gamma=gamma,
        version=version,
        start_skip_samples=Ns_to_skip_at_start,
        end_skip_samples=Ns_to_skip_at_end,
        **kwargs,
    )


def read_trajectory(
    grad_filename: str,
    dwell_time: float | str = DEFAULT_RASTER_TIME,
    num_adc_samples: int | None = None,
    gamma: Gammas | float = Gammas.HYDROGEN,
    raster_time: float = DEFAULT_RASTER_TIME,
    read_shots: bool = False,
    normalize_factor: float = KMAX,
    pre_skip: int = 0,
):
    """Get k-space locations from gradient file.

    Parameters
    ----------
    grad_filename : str
        Gradient filename.
    dwell_time : float | str, optional
        Dwell time of ADC in ms, by default 0.01
        It can also be string 'min_osf' to select dwell time
        based on minimum OSF needed to get Nyquist sampling
        (This is obtained from SPARKLING trajectory header).
    num_adc_samples : int, optional
        Number of ADC samples, by default None
    gamma : float, optional
        Gyromagnetic ratio in kHz/T, by default 42.576e3
    gradient_raster_time : float, optional
        Gradient raster time in ms, by default 0.01
    read_shots : bool, optional
        Whether in read shots configuration which accepts an extra
        point at end, by default False
    normalize_factor : float, optional
        Whether to normalize the k-space locations, by default 0.5
        When None, normalization is not done.
    pre_skip: int, optional
        Number of samples to skip from the start of each shot,
        by default 0. This is useful when we want to avoid artifacts
        from ADC switching in UTE sequences.

    Returns
    -------
    kspace_loc : np.ndarray
        K-space locations. Shape (num_shots, num_adc_samples, dimension).
    """
    with open(grad_filename, "rb") as binfile:
        data = np.fromfile(binfile, dtype=np.float32)
        if float(data[0]) > 4:
            version, data = _pop_elements(data)
            version = np.around(version, 2)
        else:
            version = 1
        dimension, data = _pop_elements(data, type="int")
        if version >= 4.1:
            fov, data = _pop_elements(data, dimension)
            img_size, data = _pop_elements(data, dimension, type="int")
            min_osf, data = _pop_elements(data, type="int")
            gamma, data = _pop_elements(data)
            gamma = gamma / 1000
            if dwell_time == "min_osf":
                dwell_time = raster_time / min_osf
        (num_shots, num_samples_per_shot), data = _pop_elements(data, 2, type="int")
        if version > 4:
            TE_pos, data = _pop_elements(data)
            grad_max, data = _pop_elements(data)
            recon_tag, data = _pop_elements(data)
            recon_tag = np.around(recon_tag, 2)
            left_over = 10
            if version > 4.1:
                timestamp, data = _pop_elements(data)
                timestamp = datetime.fromtimestamp(float(timestamp))
                left_over -= 1
            if version > 5:
                packed_skips, data = _pop_elements(data, num_elements=2, type="int")
                start_skip_samples, end_skip_samples = packed_skips
                left_over -= 2
            else:
                start_skip_samples = 0
                end_skip_samples = 0
            _, data = _pop_elements(data, left_over)
        initial_positions, data = _pop_elements(data, dimension * num_shots)
        initial_positions = np.reshape(initial_positions, (num_shots, dimension))
        if version > 4.5:
            final_positions, data = _pop_elements(data, dimension * num_shots)
            final_positions = np.reshape(final_positions, (num_shots, dimension))
        dwell_time_ns = dwell_time * 1e6
        gradient_raster_time_ns = raster_time * 1e6
        if version < 4.1:
            grad_max, data = _pop_elements(data)
        gradients, data = _pop_elements(
            data,
            dimension * num_samples_per_shot * num_shots,
        )
        # Convert gradients to T/m
        gradients = np.reshape(
            grad_max * gradients * 1e-3, (num_shots, num_samples_per_shot, dimension)
        )
        # Handle skipped samples
        if start_skip_samples > 0:
            start_location_updates = (
                np.sum(gradients[:, :start_skip_samples], axis=1) * raster_time * gamma
            )
            initial_positions += start_location_updates
            gradients = gradients[:, start_skip_samples:, :]
            num_samples_per_shot -= start_skip_samples
        if end_skip_samples > 0:
            gradients = gradients[:, :-end_skip_samples, :]
            num_samples_per_shot -= end_skip_samples
        if num_adc_samples is None:
            if read_shots:
                # Acquire one extra sample at the end of each shot in read_shots mode
                num_adc_samples = num_samples_per_shot + 1
            else:
                num_adc_samples = int(num_samples_per_shot * (raster_time / dwell_time))
        kspace_loc = np.zeros((num_shots, num_adc_samples, dimension), dtype=np.float32)
        kspace_loc[:, 0, :] = initial_positions
        adc_times = dwell_time_ns * np.arange(1, num_adc_samples)
        Q, R = divmod(adc_times, gradient_raster_time_ns)
        Q = Q.astype("int")
        if not np.all(
            np.logical_or(
                Q < num_adc_samples, np.logical_and(Q == num_adc_samples, R == 0)
            )
        ):
            warnings.warn("Binary file doesn't seem right! Proceeding anyway")
        grad_accumulated = np.cumsum(gradients, axis=1) * gradient_raster_time_ns
        for i, (q, r) in enumerate(zip(Q, R)):
            if q >= gradients.shape[1]:
                if q > gradients.shape[1]:
                    warnings.warn(
                        "Number of samples is more than what was "
                        "obtained in binary file!\n"
                        "Data will be extended"
                    )
                kspace_loc[:, i + 1, :] = (
                    initial_positions
                    + (
                        grad_accumulated[:, gradients.shape[1] - 1, :]
                        + gradients[:, gradients.shape[1] - 1, :] * r
                    )
                    * gamma
                    * 1e-6
                )
            else:
                if q == 0:
                    kspace_loc[:, i + 1, :] = (
                        initial_positions + gradients[:, q, :] * r * gamma * 1e-6
                    )
                else:
                    kspace_loc[:, i + 1, :] = (
                        initial_positions
                        + (grad_accumulated[:, q - 1, :] + gradients[:, q, :] * r)
                        * gamma
                        * 1e-6
                    )
        params = {
            "version": version,
            "dimension": dimension,
            "num_shots": num_shots,
            "num_samples_per_shot": num_samples_per_shot,
        }
        if version >= 4.1:
            params["FOV"] = fov
            params["img_size"] = img_size
            params["min_osf"] = min_osf
            params["gamma"] = gamma
            params["recon_tag"] = recon_tag
            params["TE_pos"] = TE_pos
            if version >= 4.2:
                params["timestamp"] = timestamp
        if normalize_factor is not None:
            Kmax = img_size / 2 / fov
            kspace_loc = kspace_loc / Kmax * normalize_factor
        if pre_skip > 0:
            if pre_skip >= num_samples_per_shot:
                raise ValueError(
                    "skip_first_Nsamples should be less than num_adc_samples"
                )
            oversample_factor = num_adc_samples / num_samples_per_shot
            skip_samples = pre_skip * int(oversample_factor)
            kspace_loc = kspace_loc[:, skip_samples:]
            params["num_adc_samples"] = num_adc_samples - skip_samples
        return kspace_loc, params


def read_arbgrad_rawdat(
    filename: str,
    removeOS: bool = False,
    doAverage: bool = True,
    squeeze: bool = True,
    slice_num: int | None = None,
    contrast_num: int | None = None,
    pre_skip: int = 0,
    data_type: str = "ARBGRAD_VE11C",
):  # pragma: no cover
    """Read raw data from a Siemens MRI file.

    Parameters
    ----------
    filename : str
        The path to the Siemens MRI file.
    removeOS : bool, optional
        Whether to remove the oversampling, by default False.
    doAverage : bool, optional
        Whether to average the data acquired along NAve dimension, by default True.
    squeeze : bool, optional
        Whether to squeeze the dimensions of the data, by default True.
    slice_num : int, optional
        The slice to read, by default None. This applies for 2D data.
    contrast_num: int, optional
        The contrast to read, by default None.
    pre_skip : int, optional
        Number of samples to skip from the start of each shot,
        by default 0. This is useful when we want to avoid artifacts
        from ADC switching in UTE sequences.
    data_type : str, optional
        The type of data to read, by default 'ARBGRAD_VE11C'.

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
    data, hdr, twixObj = read_siemens_rawdat(
        filename=filename,
        removeOS=removeOS,
        doAverage=doAverage,
        squeeze=squeeze,
        slice_num=slice_num,
        contrast_num=contrast_num,
    )
    if "ARBGRAD_VE11C" in data_type:
        hdr["type"] = "ARBGRAD_GRE"
        hdr["shifts"] = ()
        for s in [6, 7, 8]:
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
        if hdr["n_contrasts"] > 1:
            hdr["turboFactor"] = twixObj.search_header_for_val(
                "Phoenix", ("sFastImaging", "lTurboFactor")
            )[0]
            hdr["type"] = "ARBGRAD_MP2RAGE"
    if pre_skip > 0:
        samples_to_skip = int(hdr["oversampling_factor"] * pre_skip)
        if samples_to_skip >= hdr["n_adc_samples"]:
            raise ValueError(
                "Samples to skip should be less than n_samples in the data"
            )
        data = data[:, :, samples_to_skip:]
        hdr["n_adc_samples"] -= samples_to_skip
    return data, hdr
