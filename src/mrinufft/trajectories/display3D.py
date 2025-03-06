"""Utils for displaying 3D trajectories."""

from mrinufft import get_operator, get_density
from mrinufft.trajectories.utils import (
    convert_trajectory_to_gradients,
    convert_gradients_to_slew_rates,
    KMAX,
    DEFAULT_RASTER_TIME,
)
from mrinufft.density.utils import flat_traj
import numpy as np


def get_gridded_trajectory(
    trajectory: np.ndarray,
    shape: tuple,
    grid_type: str = "density",
    osf: int = 1,
    backend: str = "gpunufft",
    traj_params: dict = None,
    turbo_factor: int = 176,
    elliptical_samp: bool = True,
    threshold: float = 1e-3,
):
    """
    Compute various trajectory characteristics onto a grid.

    This function helps in gridding a k-space sampling trajectory to a desired shape,
    allowing for easier viewing of the trajectory.
    The gridding process can be carried out to reflect the sampling density,
    sampling time, inversion time, k-space holes, gradient strengths, or slew rates.
    Please check `grid_type` parameter to know the benefits of each type of gridding.
    During the gridding process, the values corresponding to various samples within the
    same voxel get averaged.

    Parameters
    ----------
    trajectory : ndarray
        The input array of shape (N, M, D), where N is the number of shots and M is the
        number of samples per shot and D is the dimension of the trajectory (usually 3)
    shape : tuple
        The desired shape of the gridded trajectory.
    grid_type : str, optional
        The type of gridded trajectory to compute. Default is "density".
        It can be one of the following:
        "density" : Get the sampling density in closest number of samples per voxel.
            Helps understand suboptimal sampling, by showcasing regions with strong
            oversampling.
        "time" : Showcases when the k-space data is acquired in time.
            This is helpful to view and understand off-resonance effects.
            Generally, lower off-resonance effects occur when the sampling trajectory
            has smoother k-space sampling time over the k-space.
        "inversion" : Relative inversion time at the sampling location. Needs
            `turbo_factor` to be set. This is useful for analyzing the exact inversion
            time when the k-space is acquired, for sequences like MP(2)RAGE.
        "holes": Show the k-space missing coverage, or holes, within a ellipsoid of the
            k-space.
        "gradients": Show the gradient strengths of the k-space trajectory.
        "slew": Show the slew rate of the k-space trajectory.
    osf : int, optional
        The oversampling factor for the gridded trajectory. Default is 1.
    backend : str, optional
        The backend to use for gridding. Default is "gpunufft".
        Note that "gpunufft" is anyway used to get the `pipe` density internally.
    traj_params : dict, optional
        The trajectory parameters. Default is None.
        This is only needed when `grid_type` is "gradients" or "slew".
        The parameters needed include `img_size` (tuple), `FOV` (tuple in `m`),
        and `gamma` (float in kHz/T) of the sequence.
        Generally these values are stored in the header of the trajectory file.
    turbo_factor : int, optional
        The turbo factor when sampling is with inversion. Default is 176, which is
        the default turbo factor for MPRAGE acquisitions at 1mm whole
        brain acquisitions.
    elliptical_samp : bool, optional
        Whether the k-space corners should be expected to be covered
        or ignored when `grid_type` is "holes", i.e. the trajectory is an ellipsoid
        or a cuboic and whether corners should be considered as potential holes.
        Ignoring them with `True` corresponds to trajectories with spherical/elliptical
        sampling. Default is `True`.
    threshold: float, optional default 1e-3
        The threshold for the k-space holes in number of samples per voxel
        This value is set heuristically to visualize the k-space hole.

    Returns
    -------
    ndarray
        The gridded trajectory of shape `shape`.
    """
    samples = trajectory.reshape(-1, trajectory.shape[-1])
    dcomp = get_density("pipe")(trajectory, shape)
    grid_op = get_operator(backend)(
        trajectory, [sh * osf for sh in shape], density=dcomp, upsampfac=1
    )
    gridded_ones = grid_op.raw_op.adj_op(np.ones(samples.shape[0]), None, True)
    if grid_type == "density":
        return np.abs(gridded_ones).squeeze()
    elif grid_type == "time":
        data = grid_op.raw_op.adj_op(
            np.tile(np.linspace(1, 10, trajectory.shape[1]), (trajectory.shape[0],)),
            None,
            True,
        )
    elif grid_type == "inversion":
        data = grid_op.raw_op.adj_op(
            np.repeat(
                np.linspace(1, 10, turbo_factor),
                samples.shape[0] // turbo_factor + 1,
            )[: samples.shape[0]],
            None,
            True,
        )
    elif grid_type == "holes":
        data = np.abs(gridded_ones).squeeze() < threshold
        if elliptical_samp:
            # If the trajectory uses elliptical sampling, ignore the k-space holes
            # outside the ellipsoid.
            data[
                np.linalg.norm(
                    np.meshgrid(
                        *[np.linspace(-1, 1, sh) for sh in shape], indexing="ij"
                    ),
                    axis=0,
                )
                > 1
            ] = 0
    elif grid_type in ["gradients", "slew"]:
        gradients, initial_position = convert_trajectory_to_gradients(
            trajectory,
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
    return np.squeeze(np.abs(data))
