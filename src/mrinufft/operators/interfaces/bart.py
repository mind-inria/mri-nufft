"""Interface for the BART NUFFT.

BART uses a command line interfaces, and read/writes data to files.

"""


import os
import numpy as np
import mmap
import subprocess


from ..base import FourierOperatorCPU

# available if return code is 0
BART_AVAILABLE = not subprocess.call(
    ["which", "bart"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)


class RawBartNUFFT:
    """Wrapper around BART NUFFT CLI."""

    def __init__(self, samples, shape):
        self.samples = samples  # To normalize and send to file
        self.shape_str = ":".join([str(s) for s in shape])

    def op(self, coeffs_data, grid_data):
        """Forward Operator."""

        # Format grid data to cfl format, and write to file
        # Run bart nufft with argument in subprocess
        #

    def adj_op(self, coeffs_data, grid_data):
        """Adjoint Operator."""


class MRIBartNUFFT(FourierOperatorCPU):
    """BART implementation of MRI NUFFT transform."""

    # TODO Use a n_jobs parameter to run multiple instances of BART ?
    # TODO Data consistency function: use toepliz
    #
    backend = "bart"
    available = BART_AVAILABLE

    def __init__(self, samples, shape, density=False, n_coils=1, smaps=None, **kwargs):
        ...


def _readcfl(name):
    """Read a pair of .cfl/.hdr file to get a complex numpy array.

    Parameters
    ----------
    name : str
        Name of the file to read.

    Returns
    -------
    array : array_like
        Complex array read from the file.

    License
    -------
    Copyright 2013-2015. The Regents of the University of California.
    Copyright 2021. Uecker Lab. University Center Göttingen.
    All rights reserved. Use of this source code is governed by
    a BSD-style license which can be found in the LICENSE file.

    Authors
    -------
    2013 Martin Uecker <uecker@eecs.berkeley.edu>
    2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
    """
    # get dims from .hdr
    with open(name + ".hdr") as h:
        h.readline()  # skip
        line = h.readline()
    dims = [int(i) for i in line.split()]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[: np.searchsorted(dims_prod, n) + 1]

    # load data and reshape into dims
    with open(name + ".cfl", "rb") as d:
        a = np.fromfile(d, dtype=np.complex64, count=n)
    return a.reshape(dims, order="F")  # column-major


def _writecfl(name, array):
    """Write a pair of .cfl/.hdr file representing a complex array.

    Parameters
    ----------
    name : str
        Name of the file to write.
    array : array_like
        Array to write to file.

    License
    -------
    Copyright 2013-2015. The Regents of the University of California.
    Copyright 2021. Uecker Lab. University Center Göttingen.
    All rights reserved. Use of this source code is governed by
    a BSD-style license which can be found in the LICENSE file.

    Authors
    -------
    2013 Martin Uecker <uecker@eecs.berkeley.edu>
    2015 Jonathan Tamir <jtamir@eecs.berkeley.edu>
    """
    with open(name + ".hdr", "w") as h:
        h.write("# Dimensions\n")
        for i in array.shape:
            h.write("%d " % i)
        h.write("\n")

    size = np.prod(array.shape) * np.dtype(np.complex64).itemsize

    with open(name + ".cfl", "a+b") as d:
        os.ftruncate(d.fileno(), size)
        mm = mmap.mmap(d.fileno(), size, flags=mmap.MAP_SHARED, prot=mmap.PROT_WRITE)
        if array.dtype != np.complex64:
            array = array.astype(np.complex64)
        mm.write(np.ascontiguousarray(array.T))
        mm.close()
