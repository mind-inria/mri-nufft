"""Provides a wrapper around the python bindings of cufinufft."""


import atexit
import sys
from ctypes import byref, c_int, c_void_p

import numpy as np

import cufinufft._cufinufft as raw_cf

from .utils import extract_columns, extract_column

# If we are shutting down python, we don't need to run __del__
#   This will avoid any shutdown gc ordering problems.
exiting = False
atexit.register(setattr, sys.modules[__name__], 'exiting', True)


def _default_opts(nufft_type, dim):
    """
    Generate a cufinufft opt struct of the dtype coresponding to plan.

    Parameters
    ----------
    finufft_type: int
        Finufft Type (`1` or `2`)
    dim: int
        Number of dimension (1, 2, 3)

    Returns
    -------
    nufft_opts structure.
    """
    nufft_opts = raw_cf.NufftOpts()

    ier = raw_cf._default_opts(nufft_type, dim, nufft_opts)

    if ier != 0:
        raise RuntimeError('Configuration not yet implemented.')

    return nufft_opts


class RawCufinufft:
    """GPU implementation of N-D non uniform Fast Fourrier Transform class.

    Parameters
    ----------
    samples : np.ndarray
        Samples in the non uniform space (K-space)
    shape : tuple
        Shape of the uniform space (Image space)
    n_trans : int
        Number of transform  executed by the plan.
    dtype : str or np.dtype
        Base dtype for the input data, default float32 (and thus complex64)
    opts1 : dict
        Extra parameters for the type 1 plan.
    opts2 : dict
        Extra parameters for the type 2 plan.

    Methods
    -------
    type1(coef, data)
        Type 1 tranform. data is updated with the result.
    type2(coef, data)
        Type 2 tranform. coef is updated with the results
    """

    def __init__(self, samples, shape,
                 n_trans=1, eps=1e-4, dtype=np.float32,
                 opts1=None, opts2=None):

        self.dtype = np.dtype(dtype)

        if self.dtype == np.float64:
            self._make_plan = raw_cf._make_plan
            self._set_pts = raw_cf._set_pts
            self._exec_plan = raw_cf._exec_plan
            self._destroy_plan = raw_cf._destroy_plan
            self.complex_dtype = np.complex128
        elif self.dtype == np.float32:
            self._make_plan = raw_cf._make_planf
            self._set_pts = raw_cf._set_ptsf
            self._exec_plan = raw_cf._exec_planf
            self._destroy_plan = raw_cf._destroy_planf
            self.complex_dtype = np.complex64
        else:
            raise TypeError("Expected np.float32 or np.float64.")

        self.ndim = len(shape)
        self.eps = float(eps)
        self.n_trans = n_trans
        self.references = []

        # We extend the mode tuple to 3D as needed,
        #   and reorder from C/python ndarray.shape style input (nZ, nY, nX)
        #   to the (F) order expected by the low level library (nX, nY, nZ).
        shape = shape[::-1] + (1,) * (3 - self.ndim)
        self.modes = (c_int * 3)(*shape)

        # setup optional parameters of the plan.
        self.opts1 = _default_opts(1, self.ndim)
        self.opts2 = _default_opts(2, self.ndim)

        for cls_opts, opts in zip([self.opts1, self.opts2], [opts1, opts2]):
            field_names = [name for name, _ in cls_opts._fields_]
            opts = {} if opts is None else opts
            # Assign field names from kwargs if they match up, otherwise error.
            for key, val in opts.items():
                if key in field_names:
                    setattr(cls_opts, key, val)
                else:
                    raise TypeError(f"Invalid option '{key}'")

        # creates the two plans for Type 1 and Type 2
        self.plan1, self.plan2 = c_void_p(None), c_void_p(None)

        ier = self._make_plan(1, self.ndim, self.modes, 1,
                              self.n_trans, eps, 1, byref(self.plan1), self.opts1)
        if ier != 0:
            raise RuntimeError("Type 1 plan initialisation failed.")

        ier = self._make_plan(2, self.ndim, self.modes, -1,
                              self.n_trans, eps, 1, byref(self.plan2), self.opts2)
        if ier != 0:
            raise RuntimeError("Type 2 plan initialisation failed")
        self._set_pts_plans(extract_column(samples, 0),
                            extract_column(samples, 1),
                            extract_column(samples, 2) if self.ndim == 3 else None)

    def _set_pts_plans(self, kx, ky, kz):
        if kx.dtype != self.dtype:
            raise TypeError("cufinufft plan.dtype and "
                            "kx dtypes do not match.")

        if ky and ky.dtype != self.dtype:
            raise TypeError("cufinufft plan.dtype and "
                            "ky dtypes do not match.")

        if kz and kz.dtype != self.dtype:
            raise TypeError("cufinufft plan.dtype and "
                            "kz dtypes do not match.")

        M = kx.size

        if ky and ky.size != M:
            raise TypeError("Number of elements in kx and ky must be equal")

        if kz and kz.size != M:
            raise TypeError("Number of elements in kx and kz must be equal")

        # Because FINUFFT/cufinufft are internally column major,
        #   we will reorder the pts axes. Reordering references
        #   save us from having to actually transpose signal data
        #   from row major (Python default) to column major.
        #   We do this by following translation:
        #     (x, None, None) ~>  (x, None, None)
        #     (x, y, None)    ~>  (y, x, None)
        #     (x, y, z)       ~>  (z, y, x)
        # Via code, we push each dimension onto a stack of axis
        fpts_axes = [kx.ptr, None, None]

        # We will also store references to these arrays.
        #   This keeps python from prematurely cleaning them up.
        self.references.append(kx)
        if ky is not None:
            fpts_axes.insert(0, ky.ptr)
            self.references.append(ky)

        if kz is not None:
            fpts_axes.insert(0, kz.ptr)
            self.references.append(kz)

        # Then take three items off the stack as our reordered axis.
        ier = self._set_pts(M, *fpts_axes[:3], 0, None, None, None, self.plan1)
        if ier != 0:
            raise RuntimeError("Error setting non-uniforms points of type1")
        ier = self._set_pts(M, *fpts_axes[:3], 0, None, None, None, self.plan2)
        if ier != 0:
            raise RuntimeError("Error setting non-uniforms points of type2")

    def type1(self, d_c, d_grid):
        """Type 1 transform, using on-gpu data."""
        ier = self._exec_plan(d_c.ptr, d_grid.ptr, self.plan1)
        if ier != 0:
            raise RuntimeError('Error executing Type 1 plan.')

    def type2(self, d_c, d_grid):
        """Type 2 transform, using on-gpu data."""
        ier = self._exec_plan(d_c.ptr, d_grid.ptr, self.plan2)
        if ier != 0:
            raise RuntimeError('Error executing Type 2 plan.')

    def __del__(self):
        """Destroy this instance's associated plan and data."""
        # If the process is exiting or we've already cleaned up plan, return.
        if exiting or (self.plan1 is None and self.plan2 is None):
            return
        ier = self._destroy_plan(self.plan2)
        if ier != 0:
            raise RuntimeError('Error destroying plan.')
        ier = self._destroy_plan(self.plan1)
        if ier != 0:
            raise RuntimeError('Error destroying plan.')

        # Reset plan to avoid double destroy.
        self.plan1, self.plan2 = None, None
        # Reset our reference.
        self.references = []
