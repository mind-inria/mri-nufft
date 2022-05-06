"""Provides a wrapper around the python bindings of cufinufft."""


import atexit
import sys
from ctypes import byref, c_int, c_void_p

import numpy as np


from .utils import check_error, is_cuda_array
from ._cufinufft import (
    NufftOpts, _default_opts,
    _make_pland, _make_planf,
    _exec_pland, _exec_planf,
    _destroy_pland, _destroy_planf,
    _set_ptsd, _set_ptsf,
)
#
# If we are shutting down python, we don't need to run __del__
#   This will avoid any shutdown gc ordering problems.
EXITING = False
atexit.register(setattr, sys.modules[__name__], 'EXITING', True)


def get_default_opts(nufft_type, dim):
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
    nufft_opts = NufftOpts()

    ier = _default_opts(nufft_type, dim, nufft_opts)
    check_error(ier, 'Configuration not yet implemented.')

    return nufft_opts


class RawCufinufft:
    """GPU implementation of N-D non uniform Fast Fourrier Transform class.

    Parameters
    ----------
    samples: np.ndarray
        Samples in the non uniform space (K-space)
    shape: tuple
        Shape of the uniform space (Image space)
    n_trans: int, default 1
        Number of transform  executed by the plan.
    dtype: str or np.dtype
        Base dtype for the input data, default float32 (and thus complex64)
    opts: dict or tuple of two dict, optional default None.
        Extra parameters for the type 1  and type 2 plan.
        It will be used to set non default argument for NufftOpts object.
    init_plans: bool default False
        If True, initialize cufinuffts plans at the end of constructor.

    Methods
    -------
    type1(coef, data)
        Type 1 tranform. data is updated with the result.
    type2(coef, data)
        Type 2 tranform. coef is updated with the results
    """

    def __init__(self, samples, shape,
                 n_trans=1, eps=1e-4, dtype=np.float32,
                 init_plans=False, opts=None):

        self.dtype = np.dtype(dtype)

        if self.dtype == np.float32:
            self.__make_plan = _make_planf
            self.__set_pts = _set_ptsf
            self.__exec_plan = _exec_planf
            self.__destroy_plan = _destroy_planf
            self.complex_dtype = np.complex64
        elif self.dtype == np.float64:
            self.__make_plan = _make_pland
            self.__set_pts = _set_ptsd
            self.__exec_plan = _exec_pland
            self.__destroy_plan = _destroy_pland
            self.complex_dtype = np.complex128
        else:
            raise TypeError("Expected np.float32.")

        if not samples.flags.f_contiguous and not is_cuda_array(samples):
            raise ValueError(
                "samples should be a f-contiguous (column major) GPUarray.")

        self.samples = samples

        self.ndim = len(shape)
        self.eps = float(eps)
        self.n_trans = n_trans

        # We extend the mode tuple to 3D as needed,
        #   and reorder from C/python ndarray.shape style input (nZ, nY, nX)
        #   to the (F) order expected by the low level library (nX, nY, nZ).
        shape = shape[::-1] + (1,) * (3 - self.ndim)
        self.modes = (c_int * 3)(*shape)

        # setup optional parameters of the plan.
        use_opts1 = get_default_opts(1, self.ndim)
        use_opts2 = get_default_opts(2, self.ndim)

        for cls_opts, opts in zip([use_opts1, use_opts2], [opts1, opts2]):
            field_names = [name for name, _ in cls_opts._fields_]
            opts = {} if opts is None else opts
            # Assign field names from kwargs if they match up, otherwise error.
            for key, val in opts.items():
                if key in field_names:
                    setattr(cls_opts, key, val)
                else:
                    raise TypeError(f"Invalid option '{key}'")

        # Easy access to the plans and opts.
        self.plans = [None, None, None]
        self.opts = [None, use_opts1, use_opts2]

        if init_plans:
            for typ in [1, 2]:
                self._make_plan(typ)
                self._set_pts(typ)

    def _make_plan(self, typ):
        if self.plans[typ] is None:
            plan = c_void_p(None)
            ier = self.__make_plan(typ, self.ndim, self.modes,
                                   1 if typ == 1 else -1,
                                   self.n_trans, self.eps, 1, byref(plan),
                                   self.opts[typ])
            check_error(ier, f"Type {typ} plan initialisation failed.")
            self.plans[typ] = plan
        else:
            raise RuntimeError(f"Type {typ} plan already exist.")

    def _set_pts(self, typ):
        if self.samples.dtype != self.dtype:
            raise TypeError("cufinufft plan.dtype and "
                            "samples dtypes do not match.")

        n_samples = len(self.samples)
        itemsize = np.dtype(self.dtype).itemsize
        ptr = self.samples.data.ptr
        fpts_axes = [None, None, None]
        # samples are column-major ordered.
        # We get the internal pointer associated with each axis.

        for i in range(self.samples.shape[1]):
            fpts_axes[i] = ptr + i * n_samples * itemsize

        ier = self.__set_pts(n_samples, *fpts_axes, 0,
                             None, None, None, self.plans[typ])
        check_error(ier, f"Error setting non-uniforms points of type{typ}")

    def _exec_plan(self, typ, c_ptr, f_ptr):
        ier = self.__exec_plan(c_ptr, f_ptr, self.plans[typ])
        check_error(ier, f'Error executing Type {typ} plan.')

    def _destroy_plan(self, typ):
        if self.plans[typ] is None:
            return None  # nothing to do.
        ier = self.__destroy_plan(self.plans[typ])
        check_error(ier, f'Error deleting Type {typ} plan.')
        self.plans[typ] = None
        return None

    def _type_exec(self, typ, d_c_ptr, d_grid_ptr):
        if self.plans[typ] is not None:
            self._exec_plan(typ, d_c_ptr, d_grid_ptr)
        else:
            self._make_plan(typ)
            self._set_pts(typ)
            self._exec_plan(typ, d_c_ptr, d_grid_ptr)
            self._destroy_plan(typ)

    # Exposed middle level interface #

    def type1(self, d_c_ptr, d_grid_ptr):
        """Type 1 transform, using on-gpu data.

        Parameters
        ----------
        d_c_ptr: int
            pointer to on-device non uniform coefficient array.

        d_grid_ptr: int
            pointer to on-device uniform grid array.
        """
        return self._type_exec(1, d_c_ptr, d_grid_ptr)

    def type2(self, d_c_ptr, d_grid_ptr):
        """
        Type 2 transform, using on-gpu data.

        Parameters
        ----------
        d_c_ptr: int
            pointer to on-device non uniform coefficient array.

        d_grid_ptr: int
            pointer to on-device uniform grid array.
        """
        return self._type_exec(2, d_c_ptr, d_grid_ptr)

    def __del__(self):
        """Destroy this instance's associated plan and data."""
        # If the process is exiting or we've already cleaned up plan, return.
        if EXITING or (self.plans[1] is None and self.plans[2] is None):
            return
        self._destroy_plan(1)
        self._destroy_plan(2)
        # Reset plan to avoid double destroy.
        self.plans = [None, None, None]
