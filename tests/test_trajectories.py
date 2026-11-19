"""Test the trajectory initializations and tools."""

import numpy as np
import pytest
from pytest import fixture
from mrinufft.trajectories.inits.wave_caipi import initialize_3D_wave_caipi
from mrinufft.trajectories.projection import project_trajectory
from mrinufft.trajectories.utils import (
    Acquisition,
    compute_gradients_and_slew_rates,
    convert_trajectory_to_gradients,
)


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

    # 2) Among the other two axes, find one whose max amplitude matches wavegrad
    other_idxs = [i for i in range(3) if i != readout_idx]
    max_abs_vals = [np.max(np.abs(axes[i])) for i in other_idxs]
    sin_axis_idx = other_idxs[int(np.argmax(max_abs_vals))]
    found_amp = np.max(np.abs(axes[sin_axis_idx]))

    np.testing.assert_almost_equal(found_amp, wavegrad, 5)


@fixture
def violating_trajectory():
    """Small 2D trajectory, which violates the gradient constraints."""
    trajectory = np.zeros((1, 12, 2))
    trajectory[0, :, 0] = np.linspace(-0.1, 0.1, 12)
    trajectory[0, :, 1] = 0.02 * np.sin(np.linspace(0, 2 * np.pi, 12))
    return trajectory


def test_project_trajectory_is_idempotent(violating_trajectory):
    pytest.importorskip("pyproximal")
    acq = Acquisition.default

    projected = project_trajectory(
        violating_trajectory, acq, max_iter=3000, TE_pos=0, verbose=0
    )
    projected_twice = project_trajectory(
        projected, acq, max_iter=3000, TE_pos=0, verbose=0
    )

    np.testing.assert_allclose(projected_twice, projected, atol=1e-4, rtol=1e-3)


def test_project_trajectory_satisfies_constraints(violating_trajectory):
    pytest.importorskip("pyproximal")
    acq = Acquisition.default
    projected = project_trajectory(
        violating_trajectory, acq, max_iter=3000, TE_pos=0, verbose=0
    )

    gradients, slew_rates = compute_gradients_and_slew_rates(projected, acq)
    np.testing.assert_array_equal(projected[:, 0], 0)
    assert np.max(np.linalg.norm(gradients, axis=-1)) <= acq.gmax
    assert np.max(np.linalg.norm(slew_rates, axis=-1)) <= acq.smax
