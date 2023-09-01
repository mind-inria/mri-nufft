"""Basic codes for IO for trajectories."""
import warnings
import os
from typing import Tuple, Optional
import numpy as np
from datetime import datetime
from array import array

from .utils import (
    KMAX,
    DEFAULT_RASTER_TIME_MS,
    DEFAULT_GYROMAGNETIC_RATIO,
    DEFAULT_GMAX,
    DEFAULT_SMAX,
    compute_gradients,
)


def write_gradients(
    gradients: np.ndarray,
    start_positions: np.ndarray,
    grad_filename: str,
    img_size: Tuple[int, ...],
    FOV: Tuple[float, ...],
    in_out: bool = True,
    min_osf: int = 5,
    gamma: float = 42.576e3,
    version: float = 4.2,
    recon_tag: float = 1.1,
    timestamp: Optional[float] = None,
    keep_txt_file: bool = False,
):
    """Create gradient file from gradients and start positions.

    Parameters
    ----------
    gradients : np.ndarray
        Gradients. Shape (num_shots, num_samples_per_shot, dimension).
    start_positions : np.ndarray
        Start positions. Shape (num_shots, dimension).
    grad_filename : str
        Gradient filename.
    img_size : Tuple[int, ...]
        Image size.
    FOV : Tuple[float, ...]
        Field of view.
    in_out : bool, optional
        Whether it is In-Out trajectory?, by default True
    min_osf : int, optional
        Minimum oversampling factor needed at ADC, by default 5
    gamma : float, optional
        Gyromagnetic Constant, by default 42.576e3
    version : float, optional
        Trajectory versioning, by default 4.2
    recon_tag : float, optional
        Reconstruction tag for online recon, by default 1.1
    timestamp : Optional[float], optional
        Timestamp of trajectory, by default None
    keep_txt_file : bool, optional
        Whether to keep the text file used temporarily which holds data pushed to
        binary file, by default False

    """
    num_shots = gradients.shape[0]
    num_samples_per_shot = gradients.shape[1]
    dimension = start_positions.shape[-1]
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
        if not in_out:
            if np.sum(start_positions) != 0:
                warnings.warn(
                    "The start positions are not all zero for center-out trajectory"
                )
            file.write("0\n")
        else:
            file.write("0.5\n")
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
        file.write(str("0\n" * left_over))
    # Write all the k0 values
    file.write(
        "\n".join(
            " ".join([f"{iter2:5.4f}" for iter2 in iter1]) for iter1 in start_positions
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


def _pop_elements(array, num_elements=1, type="float"):
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
        return array[0].astype(type), array[1:]
    else:
        return array[0:num_elements].astype(type), array[num_elements:]


def write_trajectory(
    trajectory: np.ndarray,
    FOV: Tuple[float, ...],
    img_size: Tuple[int, ...],
    grad_filename: str,
    traj_norm_factor: float = KMAX,
    gamma: float = DEFAULT_GYROMAGNETIC_RATIO,
    raster_time: float = DEFAULT_RASTER_TIME_MS,
    check_constraints: bool = True,
    gmax: float = DEFAULT_GMAX,
    smax: float = DEFAULT_SMAX,
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
    traj_norm_factor : float, optional
        Trajectory normalization factor, by default 0.5
    gamma : float, optional
        Gyromagnetic constant in MHz, by default 42.576e3
    raster_time : float, optional
        Gradient raster time in ms, by default 0.01
    check_constraints : bool, optional
        Check scanner constraints, by default True
    gmax : float, optional
        Maximum gradient magnitude in T/m, by default 40e-3
    smax : float, optional
        Maximum slew rate in mT/m/s, by default 100e-3
    kwargs : dict, optional
        Additional arguments for writing the gradient file.
        These are arguments passed to write_gradients function above.
    """
    gradients, start_positions, _ = compute_gradients(
        trajectory,
        traj_norm_factor=traj_norm_factor,
        resolution=np.asarray(FOV) / np.asarray(img_size),
        raster_time=raster_time,
        gamma=gamma,
        check_constraints=check_constraints,
        gmax=gmax,
        smax=smax,
    )
    write_gradients(
        gradients=gradients,
        start_positions=start_positions,
        grad_filename=grad_filename,
        img_size=img_size,
        FOV=FOV,
        gamma=gamma,
        **kwargs,
    )


def read_trajectory(
    grad_filename: str,
    dwell_time: float = DEFAULT_RASTER_TIME_MS,
    num_adc_samples: int = None,
    gamma: float = DEFAULT_GYROMAGNETIC_RATIO,
    raster_time: float = DEFAULT_RASTER_TIME_MS,
    read_shots: bool = False,
    normalize_factor: float = KMAX,
):
    """Get k-space locations from gradient file.

    Parameters
    ----------
    grad_filename : str
        Gradient filename.
    dwell_time : float, optional
        Dwell time of ADC, by default 0.01
    num_adc_samples : int, optional
        Number of ADC samples, by default None
    gyromagnetic_constant : float, optional
        Gyromagnetic Constant, by default 42.576e3
    gradient_raster_time : float, optional
        Gradient raster time, by default 0.010
    read_shots : bool, optional
        Whether in read shots configuration which accepts an extra
        point at end, by default False
    normalize : float, optional
        Whether to normalize the k-space locations, by default 0.5
        When None, normalization is not done.

    Returns
    -------
    kspace_loc : np.ndarray
        K-space locations. Shape (num_shots, num_adc_samples, dimension).
    """
    dwell_time_ns = dwell_time * 1e6
    gradient_raster_time_ns = raster_time * 1e6
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
        (num_shots, num_samples_per_shot), data = _pop_elements(data, 2, type="int")
        if num_adc_samples is None:
            if read_shots:
                num_adc_samples = num_samples_per_shot + 1
            else:
                num_adc_samples = int(num_samples_per_shot * (raster_time / dwell_time))
        if version >= 4.1:
            TE, data = _pop_elements(data)
            grad_max, data = _pop_elements(data)
            recon_tag, data = _pop_elements(data)
            recon_tag = np.around(recon_tag, 2)
            left_over = 10
            if version >= 4.2:
                timestamp, data = _pop_elements(data)
                timestamp = datetime.fromtimestamp(float(timestamp))
                left_over -= 1
            _, data = _pop_elements(data, left_over)
        start_positions, data = _pop_elements(data, dimension * num_shots)
        start_positions = np.reshape(start_positions, (num_shots, dimension))
        if version < 4.1:
            grad_max, data = _pop_elements(data)
        gradients, data = _pop_elements(
            data,
            dimension * num_samples_per_shot * num_shots,
        )
        gradients = np.reshape(
            grad_max * gradients, (num_shots * num_samples_per_shot, dimension)
        )
        # Convert gradients from mT/m to T/m
        gradients = np.reshape(gradients * 1e-3, (-1, num_samples_per_shot, dimension))
        kspace_loc = np.zeros((num_shots, num_adc_samples, dimension))
        kspace_loc[:, 0, :] = start_positions
        adc_times = dwell_time_ns * np.arange(1, num_adc_samples)
        Q, R = divmod(adc_times, gradient_raster_time_ns)
        Q = Q.astype("int")
        if not np.all(
            np.logical_or(
                Q < num_adc_samples, np.logical_and(Q == num_adc_samples, R == 0)
            )
        ):
            warnings.warn("Binary file doesnt seem right! " "Proceeding anyway")
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
                    start_positions
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
                        (start_positions + gradients[:, q, :] * r) * gamma * 1e-6
                    )
                else:
                    kspace_loc[:, i + 1, :] = (
                        start_positions
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
            params["TE"] = TE
            if version >= 4.2:
                params["timestamp"] = timestamp
        if normalize_factor is not None:
            Kmax = img_size / 2 / fov
            kspace_loc = kspace_loc / Kmax * normalize_factor
        return kspace_loc, params
