"""Interface for the BART NUFFT.

BART uses a command line interfaces, and read/writes data to files.

The file format is described here: https://bart-doc.readthedocs.io/en/latest/data.html#non-cartesian-datasets

"""

import warnings
import os
import numpy as np
import mmap
import subprocess as subp
import tempfile
from pathlib import Path

from ..base import FourierOperatorCPU, proper_trajectory

# available if return code is 0
BART_AVAILABLE = not subp.call(
    ["which", "bart"], stdout=subp.DEVNULL, stderr=subp.DEVNULL
)


class RawBartNUFFT:
    """Wrapper around BART NUFFT CLI."""

    def __init__(self, samples, shape, extra_op_args=None, extra_adj_op_args=None):
        self.samples = samples  # To normalize and send to file
        self.shape = shape
        self.shape_str = ":".join([str(s) for s in shape])
        self.shape_str += ":1" if len(shape) == 2 else ""
        self._op_args = extra_op_args or []
        self._adj_op_args = extra_adj_op_args or []
        self._temp_dir = tempfile.TemporaryDirectory()

        # Write trajectory to temp file
        tmp_path = Path(self._temp_dir.name)
        self._traj_file = tmp_path / "traj"
        self._ksp_file = tmp_path / "ksp"
        self._grid_file = tmp_path / "grid"

        traj2cfl(self.samples, self.shape, self._traj_file)

    def _tmp_file(self):
        """Return a temporary file name."""
        return os.path.join(self._temp_dir.name, next(tempfile._get_candidate_names()))

    def __del__(self):
        """Delete also the temporary files."""
        self._temp_dir.cleanup()

    def op(self, coeffs_data, grid_data):
        """Forward Operator."""
        grid_data_ = grid_data.reshape(self.shape)
        _writecfl(grid_data_, self._grid_file)
        cmd = [
            "bart",
            "nufft",
            "-d",
            self.shape_str,
            *self._op_args,
            str(self._traj_file),
            str(self._grid_file),
            str(self._ksp_file),
        ]
        try:
            subp.run(cmd, check=True, capture_output=True)
        except subp.CalledProcessError as exc:
            msg = "Failed to run BART NUFFT\n"
            msg += f"error code: {exc.returncode}\n"
            msg += "cmd: " + " ".join(cmd) + "\n"
            msg += f"stdout: {exc.output}\n"
            msg += f"stderr: {exc.stderr}"
            raise RuntimeError(msg) from exc

        ksp_raw = _readcfl(self._ksp_file)
        np.copyto(coeffs_data, ksp_raw)
        return coeffs_data

    def adj_op(self, coeffs_data, grid_data):
        """Adjoint Operator."""
        # Format grid data to cfl format, and write to file
        # Run bart nufft with argument in subprocess

        coeffs_ = coeffs_data.reshape(len(self.samples))
        _writecfl(coeffs_[None, ..., None, None, None], self._ksp_file)

        cmd = [
            "bart",
            "nufft",
            "-d",
            self.shape_str,
            "-a" if "-i" not in self._adj_op_args else "",
            *self._adj_op_args,
            str(self._traj_file),
            str(self._ksp_file),
            str(self._grid_file),
        ]
        try:
            subp.run(cmd, check=True, capture_output=True)
        except subp.CalledProcessError as exc:
            msg = "Failed to run BART NUFFT\n"
            msg += f"error code: {exc.returncode}\n"
            msg += "cmd: " + " ".join(cmd) + "\n"
            msg += f"stdout: {exc.output}\n"
            msg += f"stderr: {exc.stderr}"
            raise RuntimeError(msg) from exc

        grid_raw = _readcfl(self._grid_file)
        np.copyto(grid_data, grid_raw)
        return grid_data


class MRIBartNUFFT(FourierOperatorCPU):
    """BART implementation of MRI NUFFT transform."""

    # TODO override Data consistency function: use toepliz

    backend = "bart"
    available = BART_AVAILABLE

    def __init__(
        self,
        samples,
        shape,
        density=False,
        n_coils=1,
        n_batchs=1,
        smaps=None,
        squeeze_dims=True,
        **kwargs,
    ):
        samples_ = proper_trajectory(samples, normalize="unit")
        if density is True:
            density = False
            if getattr(kwargs, "extra_adj_op_args", None):
                kwargs["extra_adj_op_args"] += ["-i"]
            else:
                kwargs["extra_adj_op_args"] = ["-i"]

        super().__init__(
            samples_,
            shape,
            density,
            n_coils=n_coils,
            n_batchs=n_batchs,
            n_trans=1,
            smaps=smaps,
            squeeze_dims=squeeze_dims,
        )

        self.raw_op = RawBartNUFFT(samples_, shape, **kwargs)

    @property
    def norm_factor(self):
        """Normalization factor of the operator."""
        # return 1.0
        return np.sqrt(2 ** len(self.shape))


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
        for i in array.shape:
            h.write("%d " % i)
        h.write("\n")

    size = np.prod(array.shape) * np.dtype(np.complex64).itemsize

    with open(cfl_file, "w+b") as d:
        os.ftruncate(d.fileno(), size)
        mm = mmap.mmap(d.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE)
        if array.dtype != np.complex64:
            array = array.astype(np.complex64)
        mm.write(np.ascontiguousarray(array.T))
        mm.close()


def traj2cfl(traj, shape, basename):
    """
    Export a trajectory defined in MRI-nufft to a BART compatible format.

    Parameters
    ----------
    traj: array_like
        trajectory array, shape (N_shot, N_points, 2 or 3)
    shape: tuple
        volume shape (FOV)

    The trajectory will be normalized to -(FOV-1)/2 +(FOV-1)/2,
    and reshape to BART format.
    """
    traj_ = traj * (np.array(shape) - 1)
    if traj.shape[-1] == 2:
        traj_3d = np.zeros(traj_.shape[:-1] + (3,), dtype=traj_.dtype)
        traj_3d[..., :2] = traj_
        traj_ = traj_3d
    else:
        traj_ = traj_.astype(np.complex64)
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
