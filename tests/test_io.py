"""Test the trajectories io module."""
import numpy as np
from mrinufft.trajectories.io import get_kspace_loc_from_gradfile
from mrinufft.trajectories.utils import get_grads_from_kspace_points
from mrinufft.trajectories.trajectory2D import initialize_2D_radial
from mrinufft.trajectories.trajectory3D import initialize_3D_cones


def test_write_n_read():
    trajectories = [
        initialize_2D_radial(Nc=32, Ns=256, tilt="uniform", in_out=False).astype(np.float32),
        initialize_3D_cones(Nc=32, Ns=256, tilt="uniform", in_out=True).astype(np.float32),
    ]
    FOVs = [(0.23, 0.23), (0.23, 0.23, 0.1248)]
    img_sizes = [(256, 256), (256, 256, 128)]
    in_outs = [False, True]
    min_osfs = [2, 5]
    gammas = [42.576e3, 10e3]
    recon_tags = [1.1, 1.2]
    inputs = zip(trajectories, FOVs, img_sizes, in_outs, min_osfs, gammas, recon_tags)
    for iterate in inputs:
        trajectory, FOV, img_size, in_out, min_osf, gamma, recon_tag = iterate
        FOV = np.array(FOV).astype(np.float32)
        img_size = np.array(img_size).astype(np.float32)
        grads, k0, slews = get_grads_from_kspace_points(
            trajectory=trajectory,
            FOV=FOV,
            img_size=img_size,
            check_constraints=True,
            grad_filename="test",
            write_kwargs={
                "in_out": in_out,
                "version": 4.2,
                "min_osf": min_osf,
                "recon_tag": recon_tag,
            },
            gyromagnetic_constant=gamma,
        )
        read_trajectory, params = get_kspace_loc_from_gradfile("test.bin", read_shots=True)
        assert params['version'] == 4.2
        assert params['num_shots'] == trajectory.shape[0]
        assert params['num_samples_per_shot'] == trajectory.shape[1]-1
        assert params['TE'] == (0.5 if in_out else 0)
        assert params['gamma'] == gamma
        assert params['recon_tag'] == recon_tag
        assert params['min_osf'] == min_osf
        np.testing.assert_equal(params['FOV'], FOV)
        np.testing.assert_equal(params['img_size'], img_size)
        np.testing.assert_almost_equal(read_trajectory, trajectory, decimal=6)
