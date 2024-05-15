import numpy as np
import numpy.linalg as nl
from scipy.interpolate import CubicSpline


def patch_center_monoticity(parameters, update_shot, update_parameters, in_out=False, learning_rate=1e-1):
    single_shot = update_shot(*parameters)
    Ns = single_shot.shape[0]

    old_index = 0
    old_x_axis = np.zeros(Ns)
    x_axis = np.linspace(0, 1, Ns)

    while nl.norm(x_axis - old_x_axis) / Ns > 1e-10:
        # Determine interval to fix
        single_shot = update_shot(*parameters)
        gradient_norm = nl.norm(np.diff(single_shot[(Ns // 2) * in_out:], axis=0), axis=-1)
        arc_length = np.cumsum(gradient_norm)

        index = gradient_norm[1:] - arc_length[1:] / (1 + np.arange(1, len(gradient_norm)))
        index = np.where(index < 0, np.inf, index)
        index = max(old_index, 2 + np.argmin(index))
        start_index = (Ns // 2 + (Ns % 2) - index) * in_out
        final_index = (Ns // 2) * in_out + index

        # Balance over interval
        cbs = CubicSpline(x_axis, single_shot.T, axis=1)
        gradient_norm = nl.norm(np.diff(single_shot[start_index:final_index], axis=0), axis=-1)

        old_x_axis = np.copy(x_axis)
        new_x_axis = np.cumsum(np.diff(x_axis[start_index:final_index]) / gradient_norm)
        new_x_axis *= (x_axis[final_index - 1] - x_axis[start_index]) / new_x_axis[-1]  # Normalize
        new_x_axis += x_axis[start_index]
        x_axis[start_index + 1:final_index - 1] += (new_x_axis[:-1] - x_axis[start_index + 1:final_index - 1]) * learning_rate

        # Update loop variables
        old_index = index
        single_shot = cbs(x_axis).T
        parameters = update_parameters(single_shot, *parameters)
    return parameters