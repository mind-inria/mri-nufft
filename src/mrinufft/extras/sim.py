"""Simple MR simulation module."""

import numpy as np

from typing import Sequence


def fse_simulation(
    M0: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    TE: float | Sequence[float],
    TR: float | Sequence[float],
) -> np.ndarray:
    """Perform simple analytical Fast Spin Echo simulation.

    Assume that refocusing angles are 180° and
    k-space center is sampled for echo in the Echo Train
    (e.g., spiral or radial trajectory).

    Parameters
    ----------
    M0 : np.ndarray
        Input equilibrium magnetization.
    T1 : np.ndarray
        Input T1 in [ms].
    T2 : np.ndarray
        Input T2 in [ms].
    TE : float | Sequence[float]
        Sequence Echo Time in [ms].
    TR : float | Sequence[float]
        Sequence Repetition Time in [ms].

    Returns
    -------
    signal : np.ndarray
        Simulated signal of shape (nTE*nTR, *M0).

    """
    # preprocess sequence parameters
    TE, TR = np.broadcast_arrays(np.atleast_1d(TE), np.atleast_1d(TR)[:, None])
    TE, TR = TE.ravel().astype(np.float32), TR.ravel().astype(np.float32)

    # preprocess tissue parameters
    M0, T1, T2 = np.atleast_1d(M0), np.atleast_1d(T1), np.atleast_1d(T2)
    M0, T1, T2 = M0[..., None], T1[..., None], T2[..., None]
    T1 += 1e-9
    T2 += 1e-9

    # compute signal
    signal = M0 * (1 - np.exp(-(TR - TE) / T1)) * np.exp(-TE / T2)

    # post process
    signal = signal[None, ...].swapaxes(0, -1)[..., 0]
    signal = signal.squeeze()
    if signal.size == 1:
        signal = signal.item()

    return signal
