"""Slew rate ramp utilities for trajectory initialization."""

from collections.abc import Callable
from functools import wraps
import inspect

import numpy as np
from numpy.typing import NDArray

from mrinufft.trajectories.gradients import connect_gradient, min_length_connection
from mrinufft.trajectories.utils import (
    Acquisition,
    convert_gradients_to_trajectory,
    convert_trajectory_to_gradients,
    unnormalize_trajectory,
)


def _add_slew_ramp_to_traj_func(
    func: Callable,
    func_kwargs: dict,
    ramp_to_index: int,
    method: str = "lp-minslew",
    acq: Acquisition | None = None,
) -> NDArray:
    traj = func(**func_kwargs)
    acq = acq or Acquisition.default
    unnormalized_traj = unnormalize_trajectory(traj, acq=acq)
    gradients, initial_positions = convert_trajectory_to_gradients(traj, acq=acq)
    gradients_to_reach = gradients[:, ramp_to_index]
    # Calculate the number of time steps for ramps
    min_length = min_length_connection(
        kstarts=initial_positions,
        kends=unnormalized_traj[:, ramp_to_index],
        gstarts=np.zeros_like(gradients_to_reach),
        gends=gradients_to_reach,
        method=method,
        acq=acq,
    )
    # Update the Ns of the trajectory to ensure we still give
    # same Ns as users expect. We use extra 2 points as buffer.
    n_slew_ramp = min_length + 2
    func_kwargs["Ns"] -= n_slew_ramp - ramp_to_index
    new_traj = func(**func_kwargs)
    # Re-calculate the gradients
    unnormalized_traj = unnormalize_trajectory(new_traj, acq=acq)
    gradients, initial_positions = convert_trajectory_to_gradients(new_traj, acq=acq)
    gradients_to_reach = gradients[:, ramp_to_index]
    ramp_up_gradients = connect_gradient(
        kstarts=initial_positions,
        kends=unnormalized_traj[:, ramp_to_index],
        gstarts=np.zeros_like(gradients_to_reach),
        gends=gradients_to_reach,
        acq=acq,
        N=n_slew_ramp,
        method=method,
    )[:, :-1]
    ramp_up_traj = convert_gradients_to_trajectory(
        gradients=ramp_up_gradients,
        initial_positions=initial_positions,
        acq=acq,
    )
    return np.hstack([ramp_up_traj, new_traj[:, ramp_to_index:]])


def add_slew_ramp(
    func: Callable | None = None,
    ramp_to_index: int = 5,
    acq: Acquisition | None = None,
    slew_ramp_disable: bool = False,
    method: str = "lp-minslew",
) -> Callable:
    """Add slew-compatible ramps to a trajectory function.

    This decorator modifies a trajectory function to include
    slew rate ramps, ensuring that the trajectory adheres to
    the maximum slew rate and gradient amplitude constraints.
    The ramps are applied to the gradients of the trajectory
    at the specified `ramp_to_index`, which is by-default the
    index of the 5th readout sample.
    Note that this decorator does not change the length of the original
    trajectory.

    Parameters
    ----------
    func : Optional[Callable], optional
        The trajectory function to decorate. If not provided,
        the decorator can be used without arguments.
    ramp_to_index : int, optional
        The index in the trajectory where the slew ramp should be applied,
        by default 5. This is typically the index of the first readout sample.
    acq : Optional[Acquisition], optional
        An Acquisition object containing the acquisition parameters, by default None.
    slew_ramp_disable : bool, optional
        If True, the slew ramp will not be applied and the trajectory will be
        returned as is, by default False.
    method : str, optional
        The method to use for calculating the slew ramp, by default "lp-minslew".
        This can be any method supported by the `connect_gradient` function.

    Returns
    -------
    Callable
        A decorator that modifies the trajectory function to include
        slew rate ramps.

    Notes
    -----
    - The decorator modifies the trajectory function to ensure that the
        gradients at the specified `ramp_to_index` are adjusted to comply with
        the maximum slew rate and gradient amplitude constraints.
    - If `slew_ramp_disable` is set to True, the trajectory function will
        return the trajectory as is, without applying any slew ramps.
    """

    def decorator(trajectory_func):
        sig = inspect.signature(trajectory_func)

        @wraps(trajectory_func)
        def wrapped(*args, **kwargs) -> NDArray:
            # This allows users to also call the trajectory function
            # directly giving these args.
            _acq = kwargs.pop("acq", acq)
            _ramp_to_index = kwargs.pop("ramp_to_index", ramp_to_index)
            _slew_ramp_disable = kwargs.pop("slew_ramp_disable", slew_ramp_disable)
            _method = kwargs.pop("method", method)
            # Bind all args (positional and keyword)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            in_out = bound.arguments.get("in_out", False)
            if in_out or _slew_ramp_disable:
                # Send the trajectory as is for in-out trajectories
                return trajectory_func(*args, **kwargs)
            return _add_slew_ramp_to_traj_func(
                trajectory_func,
                bound.arguments,
                ramp_to_index=_ramp_to_index,
                acq=_acq,
                method=_method,
            )

        return wrapped

    if callable(func):
        return decorator(func)

    return decorator
