"""Pulseq Trajectory Reader and writers.

Reads Pulseq `.seq` files to extract k-space trajectory and other parameters. It also provides functionality to create pulseq block and shape objects to
facilitate the integration of arbitrary k-space trajectories into Pulseq
sequences. Requires the `pypulseq` package to be installed.
"""

import numpy as np
import pypulseq as pp


def read_pulseq_traj(filename):
    """Extract k-space trajectory from a Pulseq sequence file.

    The sequence should be a valid Pulseq `.seq` file, with arbitrary gradient
    waveforms, which all have the same length.

    Unlike `Sequence.calculate_kspace`, this function returns the k-space
    trajectory segmented in shots ("blocks" from Pulseq), and works directly
    from the gradients waveforms stored in the `.seq` file.

    Parameters
    ----------
    filename : str
        Path to the Pulseq `.seq` file.

    Returns
    -------
    description : dict
        the [DESCRIPTION] block from the sequence file.
    np.ndarray
        The k-space trajectory as a numpy array of shape (n_shots, n_samples, 3),
        where the last dimension corresponds to the x, y, and z coordinates in k-space.

    """

    seq = pp.Sequence()
    seq.read(filename)


    gradient_waveforms = seq.waveforms()
