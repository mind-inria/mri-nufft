from mrinufft import get_operator, get_density
from mrinufft.trajectories.utils import (
    convert_trajectory_to_gradients,
    convert_gradients_to_slew_rates,
    KMAX,
    DEFAULT_RASTER_TIME,
)
import numpy as np
from typing import Tuple
from mrinufft.io import read_trajectory


def get_gridded_trajectory(
    shots: np.ndarray,
    shape: Tuple,
    osf: int = 1,
    grid_type: str = "density",
    turbo_factor: int = 176,
    backend: str = "gpunufft",
    traj_params: dict = None,
):
    """
    Compute the gridded trajectory for MRI reconstruction.

    Parameters
    ----------
    shots : ndarray
        The input array of shape (N, M), where N is the number of shots and M is the
        number of samples per shot.
    shape : tuple
        The desired shape of the gridded trajectory.
    osf : int, optional
        The oversampling factor for the gridded trajectory. Default is 1.
    grid_type : str, optional
        The type of gridded trajectory to compute. Default is "density".
        It can be one of the following:
            "density" : Get the sampling density in closest number of samples per voxel.
            Helps understand suboptimal sampling.
            "time" : Get the sampling in time, this is helpful to view and understand
            off-resonance effects.
            "inversion" : Relative inversion time at the sampling location. Needs
            turbo_factor to be set.
            "holes": Show the k-space holes within a elliosoid of the k-space.
            "gradients": Show the gradient strengths of the k-space trajectory.
            "slew": Show the slew rate of the k-space trajectory.
    turbo_factor : int, optional
        The turbo factor when sampling is with inversion. Default is 176.
    backend : str, optional
        The backend to use for gridding. Default is "gpunufft".
        Note that "gpunufft" is anyway used to get the `pipe` density internally.
    traj_params : dict, optional
        The trajectory parameters. Default is None.
        This is only needed when `grid_type` is "gradients" or "slew".

    Returns
    -------
    ndarray
        The gridded trajectory of shape `shape`.
    """
    samples = shots.reshape(-1, shots.shape[-1])
    dcomp = get_density("pipe")(samples, shape)
    grid_op = get_operator(backend)(
        samples, [sh * osf for sh in shape], density=dcomp, upsampfac=1
    )
    gridded_ones = grid_op.raw_op.adj_op(np.ones(samples.shape[0]), None, True)
    if grid_type == "density":
        return np.abs(gridded_ones).squeeze()
    elif grid_type == "time":
        data = grid_op.raw_op.adj_op(
            np.tile(np.linspace(1, 10, shots.shape[1]), (shots.shape[0],)),
            None,
            True,
        )
    elif grid_type == "inversion":
        data = grid_op.raw_op.adj_op(
            np.repeat(
                np.linspace(1, 10, turbo_factor), samples.shape[0] // turbo_factor + 1
            )[: samples.shape[0]],
            None,
            True,
        )
    elif grid_type == "holes":
        data = np.abs(gridded_ones).squeeze() == 0
        data[
            np.linalg.norm(
                np.meshgrid(*[np.linspace(-1, 1, sh) for sh in shape], indexing="ij")
            )
        ] = 0
    elif grid_type in ["gradients", "slew"]:
        gradients, initial_position = convert_trajectory_to_gradients(
            shots,
            norm_factor=KMAX,
            resolution=np.asarray(traj_params["FOV"])
            / np.asarray(traj_params["img_size"]),
            raster_time=DEFAULT_RASTER_TIME,
            gamma=traj_params["gamma"],
        )
        if grid_type == "gradients":
            data = np.hstack(
                [gradients, np.zeros((gradients.shape[0], 1, gradients.shape[2]))]
            )
        else:
            slews, _ = convert_gradients_to_slew_rates(gradients, DEFAULT_RASTER_TIME)
            data = np.hstack([slews, np.zeros((slews.shape[0], 2, slews.shape[2]))])
        data = grid_op.raw_op.adj_op(
            np.linalg.norm(data, axis=-1).flatten(), None, True
        )
    return np.squeeze(np.abs(data))  # / np.abs(gridded_ones))
