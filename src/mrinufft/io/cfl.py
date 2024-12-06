"""utility functions for Complex-Float data file.

This allow compatibility with BART toolbox and others.

References
----------
BART Toolbox: https://bart-doc.readthedocs.io/en/latest/data.html
"""

from pathlib import Path
import os
import mmap
import warnings
import numpy as np


def _readcfl(cfl_file, hdr_file=None):
    """Read a pair of .cfl/.hdr file to get a complex numpy array.

    Adapted from the BART python cfl library.

    Parameters
    ----------
    name : str
        Name of the file to read.

    Returns
    -------
    array : array_like
        Complex array read from the file.
    """
    basename = Path(cfl_file).with_suffix("")
    if hdr_file is None:
        hdr_file = basename.with_suffix(".hdr")
    cfl_file = basename.with_suffix(".cfl")

    # get dims from .hdr
    with open(hdr_file) as h:
        h.readline()  # skip
        line = h.readline()
    dims = [int(i) for i in line.split()]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[: np.searchsorted(dims_prod, n) + 1]

    # load data and reshape into dims
    with open(cfl_file, "rb") as d:
        a = np.fromfile(d, dtype=np.complex64, count=n)
    return a.reshape(dims, order="F")


def _writecfl(array, cfl_file, hdr_file=None):
    """Write a pair of .cfl/.hdr file representing a complex array.

    Adapted from the BART python cfl library.

    Parameters
    ----------
    name : str
        Name of the file to write.
    array : array_like
        Array to write to file.

    """
    basename = Path(cfl_file).with_suffix("")
    if hdr_file is None:
        hdr_file = basename.with_suffix(".hdr")
    cfl_file = basename.with_suffix(".cfl")

    with open(hdr_file, "w") as h:
        h.write("# Dimensions\n")
        h.write("".join(f"{i} " for i in array.shape))
        h.write("\n")

    size = np.prod(array.shape) * np.dtype(np.complex64).itemsize

    with open(cfl_file, "w+b") as d:
        os.ftruncate(d.fileno(), size)
        mm = mmap.mmap(d.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE)
        array = array.astype(np.complex64, copy=False)
        mm.write(np.ascontiguousarray(array.T))
        mm.close()


def traj2cfl(traj, shape, basename):
    """
    Export a trajectory defined in MRI-nufft to a BART compatible format.

    The trajectory will be normalized to -(FOV-1)/2 +(FOV-1)/2,
    and reshape to BART format.

    Parameters
    ----------
    traj: array_like
        trajectory array, shape (N_shot, N_points, 2 or 3)
    shape: tuple
        volume shape (FOV)

    """
    traj_ = traj * (np.array(shape) - 1)
    if traj.shape[-1] == 2:
        traj_3d = np.zeros(traj_.shape[:-1] + (3,), dtype=traj_.dtype)
        traj_3d[..., :2] = traj_
        traj_ = traj_3d
    else:
        traj_ = traj_.astype(np.complex64, copy=False)
    traj_ = traj_[None, None, ...]

    _writecfl(traj_.T, basename)


def cfl2traj(basename, shape=None):
    """Convert a trajectory BART file to a numpy array compatible with MRI-nufft.

    Parameters
    ----------
    filename: str
        Base filename for the trajectory
    shape: optional
        Shape of the Image domain.

    Returns
    -------
    np.ndarray
        MRI-NUFFT compatible trajectory of shape (n_shot, n_samples, dim)
    """
    traj_raw = _readcfl(basename)
    # Convert to float array and take only the real part
    traj = np.ascontiguousarray(traj_raw.T.view("(2,)float32")[..., 0])
    if np.all(traj[..., -1] == 0):
        warnings.warn("2D Trajectory Detected")
        traj = traj[..., :-1]
    if shape is None:
        maxs = [np.max(traj[..., i]) for i in range(traj.shape[-1])]
        mins = [np.min(traj[..., i]) for i in range(traj.shape[-1])]
        shape = np.array(maxs) - np.array(mins)
        warnings.warn(f"Estimated shape {shape}")
    else:
        shape = np.asarray(shape) - 1

    traj /= np.asarray(shape)
    return np.squeeze(traj)
