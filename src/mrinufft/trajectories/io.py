"""Holds functions for reading and writing trajectories from and to binary files."""
import warnings
import numpy as np


def get_grads_from_kspace_points(trajectory, FOV, img_size, trajectory_normalization_factor=0.5,
                                 gyromagnetic_constant=42.576e3, gradient_raster_time=0.01, 
                                 check_constraints=True, gradient_mag_max=40e-3,
                                 slew_rate_max=100e-3):
    """Calculate gradients from k-space points. Also returns start positions, slew rates and 
    allows for checking of scanner constraints.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory in k-space points. Shape (num_shots, num_samples_per_shot, dimension).
    FOV : float or tuple
        Field of view
    img_size : int or tuple
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
        trajectory = trajectory * np.array(img_size) / (2 * np.array(FOV)) / trajectory_normalization_factor

    # calculate gradients and slew
    gradients = np.diff(trajectory, axis=1) / gyromagnetic_constant / gradient_raster_time
    start_positions = trajectory[:, 0, :]
    slew_rate = np.diff(gradients, axis=1) / gradient_raster_time

    # check constraints
    if check_constraints:
        if np.max(gradients) > gradient_mag_max:
            warnings.warn("Gradient Maximum Maginitude overflow from Machine capabilities")
        if np.max(slew_rate) > slew_rate_max:
            occurences = np.where(slew_rate > slew_rate_max)
            warnings.warn(
                "Slew Rate overflow from Machine capabilities!\n"
                "Occurences per shot : " + str(len(occurences[0]) / trajectory.shape[0]) + "\n"
                "Max Value : " + str(np.max(np.abs(slew_rate)))
            )
    return gradients, start_positions, slew_rate


def create_gradient_file(gradients, k0, filename, keep_txt_file=False,
                         version=4.2, acq_params={}, recon_tag=1.1, timestamp=None):
    num_shots = gradients.shape[0]
    num_samples_per_shot = gradients.shape[1]
    dimension = k0.shape[-1]
    if version >= 4.1 and acq_params == {}:
        warnings.warn("For Gradient spec version of 4.1, we need acq_params, "
                      "writing binaries based on base version which wont have "
                      "FOV and img_size related information\n"
                      "This will be raised as an error in future!")
        version = 1
    # Convert gradients to mT/m
    gradients = convert_NCxNSxD_to_NCNSxD(gradients) * 1e3
    max_grad = np.max(np.abs(gradients))
    file = open(filename + '.txt', 'w')
    if version >= 4.1:
        file.write(str(version) + '\n')
    # Write the dimension, num_samples_per_shot and num_shots
    file.write(str(dimension) + '\n')
    if version >= 4.1:
        img_size = acq_params['recon_params']['img_size']
        FOV = acq_params['recon_params']['FOV']
        if type(img_size) is int:
            img_size = (img_size,) * dimension
        if type(FOV) is float:
            FOV = (FOV,) * dimension
        for fov in FOV:
            file.write(str(fov) + '\n')
        for sz in img_size:
            file.write(str(sz) + '\n')
        file.write(str(acq_params['traj_params']
                       ['oversampling_factor']) + '\n')
        file.write(str(acq_params['scan_consts']
                       ['gyromagnetic_constant'] * 1000) + '\n')
    file.write(str(num_shots) + '\n')
    file.write(str(num_samples_per_shot) + '\n')
    if version >= 4.1:
        initialization = acq_params['traj_params']['initialization']
        if initialization == 'RadialIO' or \
                initialization == 'SpiralIO' or \
                initialization == 'PappusIO':
            file.write('0.5\n')
        else:
            file.write('0\n')
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
    file.write('\n'.join(' '.join(["{0:5.4f}".format(iter2)
                                   for iter2 in iter1])
                         for iter1 in k0) + '\n')
    if version < 4.1:
        # Write the maximum Gradient
        file.write(str(max_grad) + '\n')
    # Normalize gradients
    gradients = gradients / max_grad
    file.write('\n'.join(' '.join(["{0:5.6f}".format(iter2)
                                   for iter2 in iter1])
                         for iter1 in gradients) + '\n')
    file.close()
    y = []
    with open(filename + '.txt', 'r') as txtfile:
        for line in txtfile:
            x = line.split(' ')
            for val in x:
                y.append(float(val))
    float_array = array('f', y)
    with open(filename + '.bin', 'wb') as binfile:
        float_array.tofile(binfile)
    if keep_txt_file is False:
        os.remove(filename + '.txt')


def get_kspace_loc_from_gradfile(filename, dwell_time=0.01,
                                 num_adc_samples=None, gamma=42.576e3,
                                 gradient_raster_time=0.010, verbose=0, read_shots=False):
    def _pop_elements(array, num_elements=1, type='float'):
        if num_elements == 1:
            return array[0].astype(type), array[1:]
        else:
            return array[0:num_elements].astype(type), \
                   array[num_elements:]
    dwell_time_ns = dwell_time * 1e6
    gradient_raster_time_ns = gradient_raster_time * 1e6
    with open(filename, 'rb') as binfile:
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
            gamma, data = _pop_elements(data)
            gamma = gamma / 1000
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
        k0, data = _pop_elements(data, dimension*num_shots)
        k0 = np.reshape(k0, (num_shots, dimension))
        if version < 4.1:
            grad_max, data = _pop_elements(data)
        gradients, data = _pop_elements(
            data,
            dimension * num_samples_per_shot * num_shots,
        )
        gradients = np.reshape(grad_max * gradients,
                               (num_shots * num_samples_per_shot, dimension))
        # Convert gradients from mT/m to T/m
        gradients = convert_NCNSxD_to_NCxNSxD(gradients * 1e-3,
                                              num_samples_per_shot)
        kspace_loc = np.zeros((num_shots, num_adc_samples, dimension))
        kspace_loc[:, 0, :] = k0
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
                        k0 + (
                            grad_accumulated[:, gradients.shape[1]-1, :] +
                            gradients[:, gradients.shape[1]-1, :] * r
                        ) * gamma * 1e-6
                    )
            else:
                if q == 0:
                    kspace_loc[:, i+1, :] = (
                            k0 + gradients[:, q, :] * r * gamma * 1e-6
                    )
                else:
                    kspace_loc[:, i+1, :] = (
                            k0 + (
                                grad_accumulated[:, q-1, :] +
                                gradients[:, q, :] * r
                            ) * gamma * 1e-6
                        )
        if verbose:
            loc = kspace_loc / \
                  np.max(np.max(np.abs(kspace_loc), axis=0), axis=0)
            scatter_shots(loc, num_shots='all', isnormalized=True)
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
            params['gamma'] = gamma
            params['recon_tag'] = recon_tag
            if version >= 4.2:
                params['timestamp'] = timestamp
        return kspace_loc, params

    
    