"""Interface for the BART NUFFT.

BART uses a command line interfaces, and read/writes data to files.

The file format is described here: https://bart-doc.readthedocs.io/en/latest/data.html#non-cartesian-datasets

"""

import os
import subprocess as subp
import tempfile
from pathlib import Path
from mrinufft._utils import proper_trajectory
from mrinufft.operators.base import FourierOperatorCPU

import numpy as np
from mrinufft.io.cfl import traj2cfl, _writecfl, _readcfl

# available if return code is 0
try:
    BART_AVAILABLE = not subp.call(
        ["which", "bart"], stdout=subp.DEVNULL, stderr=subp.DEVNULL
    )
except Exception:
    BART_AVAILABLE = False


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

        self.raw_op = RawBartNUFFT(samples_, shape, **kwargs)
        super().__init__(
            samples_,
            shape,
            density,
            n_coils=n_coils,
            n_batchs=n_batchs,
            n_trans=1,
            smaps=smaps,
            raw_op=self.raw_op,
            squeeze_dims=squeeze_dims,
        )

    @property
    def norm_factor(self):
        """Normalization factor of the operator."""
        # return 1.0
        return np.sqrt(2 ** len(self.shape))
