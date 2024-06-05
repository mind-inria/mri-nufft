"""Functions to improve/modify gradients."""

import numpy as np
import numpy.linalg as nl
from scipy.interpolate import CubicSpline


def patch_center_anomaly(
    shot_or_params,
    update_shot=None,
    update_parameters=None,
    in_out=False,
    learning_rate=1e-1,
):
    """Re-position samples to avoid center anomalies.

    Some trajectories behave slightly differently from expected when
    approaching definition bounds, most often the k-space center as
    for spirals in some cases.

    This function enforces non-strictly increasing monoticity of
    sample distances from the center, effectively reducing slew
    rates and smoothing gradient transitions locally.

    Shots can be updated with provided functions to keep fitting
    their strict definition, or it can be smoothed using cubic
    spline approximations.

    Parameters
    ----------
    shot_or_params : np.array, list
        Either a single shot of shape (Ns, Nd), or a list of arbitrary
        arguments used by ``update_shot`` to initialize a single shot.
    update_shot : function, optional
        Function used to initialize a single shot based on parameters
        provided by ``update_parameters``. If None, cubic splines are
        used as an approximation instead, by default None
    update_parameters : function, optional
        Function used to update shot parameters when provided in
        ``shot_or_params`` from an updated shot and parameters.
        If None, cubic spline parameterization is used instead,
        by default None
    in_out : bool, optional
        Whether the shot is going in-and-out or start from the center,
        by default False
    learning_rate : float, optional
        Learning rate used in the iterative optimization process,
        by default 1e-1

    Returns
    -------
    array_like
        N-D trajectory based on ``shot_or_params`` if a shot or
        update_shot otherwise.
    list
        Updated parameters either in the ``shot_or_params`` format
        if params, or cubic spline parameterization as an array of
        floats between 0 and 1 otherwise.
    """
    if update_shot is None or update_parameters is None:
        single_shot = shot_or_params
    else:
        parameters = shot_or_params
        single_shot = update_shot(*parameters)
    Ns = single_shot.shape[0]

    old_index = 0
    old_x_axis = np.zeros(Ns)
    x_axis = np.linspace(0, 1, Ns)

    if update_shot is None or update_parameters is None:

        def _default_update_parameters(shot, *parameters):
            return parameters

        update_parameters = _default_update_parameters
        parameters = [x_axis]
        update_shot = CubicSpline(x_axis, single_shot)

    while nl.norm(x_axis - old_x_axis) / Ns > 1e-10:
        # Determine interval to fix
        single_shot = update_shot(*parameters)
        gradient_norm = nl.norm(
            np.diff(single_shot[(Ns // 2) * in_out :], axis=0), axis=-1
        )
        arc_length = np.cumsum(gradient_norm)

        index = gradient_norm[1:] - arc_length[1:] / (
            1 + np.arange(1, len(gradient_norm))
        )
        index = np.where(index < 0, np.inf, index)
        index = max(old_index, 2 + np.argmin(index))
        start_index = (Ns // 2 + (Ns % 2) - index) * in_out
        final_index = (Ns // 2) * in_out + index

        # Balance over interval
        cbs = CubicSpline(x_axis, single_shot.T, axis=1)
        gradient_norm = nl.norm(
            np.diff(single_shot[start_index:final_index], axis=0), axis=-1
        )

        old_x_axis = np.copy(x_axis)
        new_x_axis = np.cumsum(np.diff(x_axis[start_index:final_index]) / gradient_norm)
        new_x_axis *= (x_axis[final_index - 1] - x_axis[start_index]) / new_x_axis[
            -1
        ]  # Normalize
        new_x_axis += x_axis[start_index]
        x_axis[start_index + 1 : final_index - 1] += (
            new_x_axis[:-1] - x_axis[start_index + 1 : final_index - 1]
        ) * learning_rate

        # Update loop variables
        old_index = index
        single_shot = cbs(x_axis).T
        parameters = update_parameters(single_shot, *parameters)

    single_shot = single_shot = update_shot(*parameters)
    return single_shot, parameters
