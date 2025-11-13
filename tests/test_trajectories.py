import numpy as np
from mrinufft import initialize_3D_wave_caipi
from mrinufft.trajectories.utils import Acquisition, convert_trajectory_to_gradients


def test_wave_caipi_gradients(wavegrad=8.8e-3, caipi_delta=2, nb_revolutions=11):
    """
    Simple test for 3D wave-CAIPI trajectory gradients:
      - One gradient axis (readout) should be essentially constant.
      - One of the other axes should show a sinusoidal modulation whose amplitude
        matches the requested `wavegrad`.
    """
    acq = Acquisition.default
    wavegrad = 8.8e-3
    traj = initialize_3D_wave_caipi(
        (3, 3),
        417,
        packing="square",
        shape="square",
        nb_revolutions=nb_revolutions,
        readout_axis=0,
        wavegrad=wavegrad,
        acq=acq,
        caipi_delta=caipi_delta,
    )
    G, K0 = convert_trajectory_to_gradients(traj, acq)
    G = np.asarray(G)

    # Normalize layout: get axes as a list [Gx, Gy, Gz]
    if G.ndim == 2 and G.shape[1] == 3:
        axes = G.T
    elif G.ndim == 2 and G.shape[0] == 3:
        axes = G
    else:
        axes = G.reshape(-1, 3).T

    # 1) Identify readout axis as the one with the smallest relative std dev
    rel_stds = [np.std(ax) / (np.mean(np.abs(ax)) + 1e-12) for ax in axes]
    readout_idx = int(np.argmin(rel_stds))
    assert rel_stds[readout_idx] < 1e-12, (
        f"Expected a constant readout gradient (rel std < 1e-3). "
        f"Got rel std {rel_stds[readout_idx]:.3e} on axis {readout_idx}"
    )

    # 2) Among the other two axes, find one whose max absolute amplitude matches wavegrad
    other_idxs = [i for i in range(3) if i != readout_idx]
    max_abs_vals = [np.max(np.abs(axes[i])) for i in other_idxs]
    sin_axis_idx = other_idxs[int(np.argmax(max_abs_vals))]
    found_amp = np.max(np.abs(axes[sin_axis_idx]))

    np.testing.assert_almost_equal(found_amp, wavegrad, 5)
