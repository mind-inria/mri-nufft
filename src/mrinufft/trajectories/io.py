"""Holds functions for reading and writing trajectories from and to binary files.
These trajectories nees to follow the format specified as shown below:

|Name          |Type |Size   |Unit   |Description                                                              |
|--------------|-----|-------|-------|-------------------------------------------------------------------------|
|Version       |FLOAT|1      |n.a.   |file version this new version would be “4.0”                             |
|Dimension     |FLOAT|1      |n.a.   |2 -> 2D , 3 -> 3D                                                        |
|FOV           |FLOAT|D      |m      |FOV size (x,y,z) : z absent if 2D dimension                              | 
|Minimum OSF   |FLOAT|1      |n.a.   |Minimum OS for the trajectory                                            |
|Gamma         |FLOAT|1      |Hz/T   |For Na / MRSI imaging                                                    |
|Spokes        |FLOAT|1      |n.a.   |Number of spokes                                                         |
|Samples       |FLOAT|1      |n.a.   |Number of samples per spoke                                              |
|K-space center|FLOAT|1      |n.a.   |Relative value in the range [0-1] to define center of spokes             |
|MaxGrad       |FLOAT|1      |mT/m   |Maximum absolute gradient in all 3 (or 2) directions                     |
|recon_tag     |FLOAT|1      |n.a.   |Reconstruction tag                                    	               |
|timestamp     |FLOAT|1      |n.a.   |Time stamp when the binary is created                                    |
|Empty places  |FLOAT|9      |n.a.   |Yet unused : Default initialized with 0                                  |
|kStarts       |FLOAT|D*Nc   |1/m    |K-space location start 	                                               |
|Gradient array|FLOAT|D*Nc*Ns|unitary|Gradient trajectory expressed in the range [-1; 1] relative to MaxGrad   |




"""
import warnings
import os
from typing import Tuple, Union, Optional
import numpy as np
from datetime import datetime
from array import array



def get_grads_from_kspace_points(
    trajectory: np.ndarray,
    FOV: Tuple[float, ...],
    img_size: Tuple[int, ...],
    trajectory_normalization_factor: float = 0.5,
    gyromagnetic_constant: float = 42.576e3,
    gradient_raster_time: float = 0.01,
    check_constraints: bool = True,
    gradient_mag_max: float = 40e-3,
    slew_rate_max: float = 100e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate gradients from k-space points. Also returns start positions, slew rates and 
    allows for checking of scanner constraints.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory in k-space points. Shape (num_shots, num_samples_per_shot, dimension).
    FOV : tuple
        Field of view
    img_size : tuple
        Image size
    trajectory_normalization_factor : float, optional
        Trajectory normalization factor, by default 0.5
    gyromagnetic_constant : float, optional
        Gyromagnetic constant, by default 42.576e3
    gradient_raster_time : float, optional
        Gradient raster time, by default 0.01
    check_constraints : bool, optional
        Check scanner constraints, by default True
    gradient_mag_max : float, optional
        Maximum gradient magnitude, by default 40e-3
    slew_rate_max : float, optional
        Maximum slew rate, by default 100e-3
        
    Returns
    -------
    gradients : np.ndarray
        Gradients. Shape (num_shots-1, num_samples_per_shot, dimension).
    start_positions : np.ndarray
        Start positions. Shape (num_shots, dimension).
    slew_rate : np.ndarray
        Slew rates. Shape (num_shots-2, num_samples_per_shot, dimension).
    """
    # normalize trajectory by image size
    if trajectory_normalization_factor:
        trajectory = trajectory * np.array(img_size) / (
            2 * np.array(FOV)
        ) / trajectory_normalization_factor

    # calculate gradients and slew
    gradients = np.diff(trajectory, axis=1) / gyromagnetic_constant / gradient_raster_time
    start_positions = trajectory[:, 0, :]
    slew_rate = np.diff(gradients, axis=1) / gradient_raster_time

    # check constraints
    if check_constraints:
        if np.max(gradients) > gradient_mag_max:
            warnings.warn(
                "Gradient Maximum Maginitude overflow from Machine capabilities"
            )
        if np.max(slew_rate) > slew_rate_max:
            occurences = np.where(slew_rate > slew_rate_max)
            warnings.warn(
                "Slew Rate overflow from Machine capabilities!\n"
                "Occurences per shot : "
                + str(len(occurences[0]) / trajectory.shape[0])
                + "\n"
                "Max Value : "
                + str(np.max(np.abs(slew_rate)))
            )
    return gradients, start_positions, slew_rate


def create_gradient_file(gradients: np.ndarray, start_positions: np.ndarray, 
                         grad_filename: str, img_size: Tuple[int, ...], 
                         FOV: Tuple[float, ...], in_out: bool = True,
                         min_osf: int = 5, gyromagnetic_constant: float = 42.576e3, 
                         version: float = 4.2, recon_tag: float = 1.1, 
                         timestamp: Optional[float] = None, keep_txt_file: bool = False):
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
    gyromagnetic_constant : float, optional
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
    file = open(grad_filename + '.txt', 'w')
    if version >= 4.1:
        file.write(str(version) + '\n')
    # Write the dimension, num_samples_per_shot and num_shots
    file.write(str(dimension) + '\n')
    if version >= 4.1:
        img_size = img_size
        FOV = FOV
        if type(img_size) is int:
            img_size = (img_size,) * dimension
        if type(FOV) is float:
            FOV = (FOV,) * dimension
        for fov in FOV:
            file.write(str(fov) + '\n')
        for sz in img_size:
            file.write(str(sz) + '\n')
        file.write(str(min_osf) + '\n')
        file.write(str(gyromagnetic_constant * 1000) + '\n')
    file.write(str(num_shots) + '\n')
    file.write(str(num_samples_per_shot) + '\n')
    if version >= 4.1:
        if not in_out:
            if np.sum(start_positions) != 0:
                warnings.warn("The start positions are not all zero for center-out trajectory")
            file.write('0\n')
        else:
            file.write('0.5\n')
        # Write the maximum Gradient
        file.write(str(max_grad) + '\n')
        # Write recon Pipeline version tag
        file.write(str(recon_tag) + '\n')
        left_over = 10
        if version >= 4.2:
            # Inset datetime tag
            if timestamp is None:
                timestamp = float(datetime.now().timestamp())
            file.write(str(timestamp) + '\n')
            left_over -= 1
        file.write(str('0\n'*left_over))
    # Write all the k0 values
    file.write('\n'.join(
        ' '.join(
                ["{0:5.4f}".format(iter2) for iter2 in iter1]
            )
        for iter1 in start_positions) + '\n'
    )
    if version < 4.1:
        # Write the maximum Gradient
        file.write(str(max_grad) + '\n')
    # Normalize gradients
    gradients = gradients / max_grad
    file.write('\n'.join(
        ' '.join(
            ["{0:5.6f}".format(iter2) for iter2 in iter1]
        )
        for iter1 in gradients) + '\n'
    )
    file.close()
    y = []
    with open(grad_filename + '.txt', 'r') as txtfile:
        for line in txtfile:
            x = line.split(' ')
            for val in x:
                y.append(float(val))
    float_array = array('f', y)
    with open(grad_filename + '.bin', 'wb') as binfile:
        float_array.tofile(binfile)
    if not keep_txt_file:
        os.remove(grad_filename + '.txt')



def _pop_elements(array, num_elements=1, type='float'):
    """A function to pop elements from an array.
    Parameters
    ----------
    array : np.ndarray
        Array to pop elements from.
    num_elements : int, optional
        number of elements to pop, by default 1
    type : str, optional
        Type of the element being popped, by default 'float'
    
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
        return array[0:num_elements].astype(type), \
               array[num_elements:]
                   
                   
def get_kspace_loc_from_gradfile(grad_filename: str, dwell_time: float=0.01, num_adc_samples: int=None, 
                                 gyromagnetic_constant: float=42.576e3, gradient_raster_time: float=0.010,
                                 read_shots: bool=False):
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
        Whether in read shots configuration which accepts an extra point at end, by default False

    Returns
    -------
    kspace_loc : np.ndarray
        K-space locations. Shape (num_shots, num_adc_samples, dimension).
    """    
    dwell_time_ns = dwell_time * 1e6
    gradient_raster_time_ns = gradient_raster_time * 1e6
    with open(grad_filename, 'rb') as binfile:
        data = np.fromfile(binfile, dtype=np.float32)
        if float(data[0]) > 4:
            version, data = _pop_elements(data)
            version = np.around(version, 2)
        else:
            version = 1
        dimension, data = _pop_elements(data, type='int')
        if version >= 4.1:
            fov, data = _pop_elements(data, dimension)
            img_size, data = _pop_elements(data, dimension, type='int')
            min_osf, data = _pop_elements(data, type='int')
            gyromagnetic_constant, data = _pop_elements(data)
            gyromagnetic_constant = gyromagnetic_constant / 1000
        (num_shots,
         num_samples_per_shot), data = _pop_elements(data, 2, type='int')
        if num_adc_samples is None:
            if read_shots:
                num_adc_samples = num_samples_per_shot + 1
            else:
                num_adc_samples = int(
                    num_samples_per_shot * (gradient_raster_time / dwell_time)
                )
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
        start_positions, data = _pop_elements(data, dimension*num_shots)
        start_positions = np.reshape(start_positions, (num_shots, dimension))
        if version < 4.1:
            grad_max, data = _pop_elements(data)
        gradients, data = _pop_elements(
            data,
            dimension * num_samples_per_shot * num_shots,
        )
        gradients = np.reshape(grad_max * gradients,
                               (num_shots * num_samples_per_shot, dimension))
        # Convert gradients from mT/m to T/m
        gradients = np.reshape(
            gradients * 1e-3,
            (-1, num_samples_per_shot, dimension)
        )
        kspace_loc = np.zeros((num_shots, num_adc_samples, dimension))
        kspace_loc[:, 0, :] = start_positions
        adc_times = dwell_time_ns * np.arange(1, num_adc_samples)
        Q, R = divmod(adc_times, gradient_raster_time_ns)
        Q = Q.astype('int')
        if not np.all(
                np.logical_or(Q < num_adc_samples,
                              np.logical_and(Q == num_adc_samples, R == 0))
        ):
            warnings.warn("Binary file doesnt seem right! "
                          "Proceeding anyway")
        grad_accumulated = (np.cumsum(gradients, axis=1) *
                            gradient_raster_time_ns)
        for i, (q, r) in enumerate(zip(Q, R)):
            if q >= gradients.shape[1]:
                if q > gradients.shape[1]:
                    warnings.warn("Number of samples is more than what was "
                                  "obtained in binary file!\n"
                                  "Data will be extended")
                kspace_loc[:, i+1, :] = (
                        start_positions + (
                            grad_accumulated[:, gradients.shape[1]-1, :] +
                            gradients[:, gradients.shape[1]-1, :] * r
                        ) * gyromagnetic_constant * 1e-6
                    )
            else:
                if q == 0:
                    kspace_loc[:, i+1, :] = (
                            start_positions + gradients[:, q, :] * r * gyromagnetic_constant * 1e-6
                    )
                else:
                    kspace_loc[:, i+1, :] = (
                            start_positions + (
                                grad_accumulated[:, q-1, :] +
                                gradients[:, q, :] * r
                            ) * gyromagnetic_constant * 1e-6
                        )
        params = {
            'version': version,
            'dimension': dimension,
            'num_shots': num_shots,
            'num_samples_per_shot': num_samples_per_shot,
        }
        if version >= 4.1:
            params['FOV'] = fov
            params['img_size'] = img_size
            params['min_osf'] = min_osf
            params['gamma'] = gyromagnetic_constant
            params['recon_tag'] = recon_tag
            if version >= 4.2:
                params['timestamp'] = timestamp
        return kspace_loc, params

    
    