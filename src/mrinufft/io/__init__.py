"""Input/Output module for trajectories and data."""

from mrinufft.io.cfl import (
    cfl2traj,
    traj2cfl,
)
from mrinufft.io.nsp import (
    read_arbgrad_rawdat,
    read_trajectory,
    write_gradients,
    write_trajectory,
)
from mrinufft.io.pulseq import (
    pulseq_gre,
    read_pulseq_traj,
)
from mrinufft.io.siemens import (
    read_siemens_rawdat,
    twix2nifti_affine,
)
from mrinufft.io.utils import (
    add_phase_to_kspace_with_shifts,
    remove_extra_kspace_samples,
)

__all__ = [
    "add_phase_to_kspace_with_shifts",
    "cfl2traj",
    "nifti_affine",
    "pulseq_gre",
    "read_arbgrad_rawdat",
    "read_pulseq_traj",
    "read_siemens_rawdat",
    "read_trajectory",
    "remove_extra_kspace_samples",
    "traj2cfl",
    "write_gradients",
    "write_trajectory",
]
