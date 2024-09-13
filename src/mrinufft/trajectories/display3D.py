from mrinufft import get_operator, get_density
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
    turbo_factor : int, optional
        The turbo factor when sampling is with inversion. Default is 176.
    backend : str, optional
        The backend to use for gridding. Default is "gpunufft".
        Note that "gpunufft" is anyway used to get the `pipe` density internally.

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
    return np.squeeze(np.abs(data) / np.abs(gridded_ones))